declare const process: { env: Record<string, string | undefined> };

import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, addIssueLabel, acquireRenewableIssueMutex,
  authenticate, commentIssue, contributorLabel, deleteLease, enforceRateLimit, fetchIssue,
  getLeaseById, issueComments, json, method, patchIssue, parseVolunteerMarker, readJson,
  redis, redisGet, requireProtocol, selectedLanguages, uploadStaging, VolunteerLeaseMarker,
  replaceVolunteerMarker, signVolunteerMarker, verifyVolunteerMarker,
} from "./_lib";

export const config = { runtime: "edge" };
const FINAL_MARKER = "euroeval-volunteer-finalised:v1";
type ResultEntry = { digest: string; identity: string; path: string; warnings?: string[] };
type Receipt = { status?: string; submission_id: string; lease: any; entries: ResultEntry[]; manifest_path: string; manifest?: string };

function resultEntries(value: unknown): ResultEntry[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (typeof item !== "string") return [];
    try { const parsed = JSON.parse(item) as ResultEntry; return parsed.digest && parsed.identity && parsed.path ? [parsed] : []; } catch { return []; }
  });
}

function suffix(identity: string): string | null {
  try { const value = JSON.parse(identity) as unknown[]; return Array.isArray(value) && value.length === 4 ? JSON.stringify(value.slice(1)) : null; } catch { return null; }
}

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const identity = await authenticate(req); await enforceRateLimit(`euroeval:worker:limit:finalise:${identity.hash}`, 30, 3600);
    const body = await readJson(req, 32 * 1024); requireProtocol(body);
    if (typeof body.lease_id !== "string") throw new BrokerError(400, "lease_id is required.");
    const statusKey = `euroeval:worker:finalisation:${body.lease_id}`;
    const old = await redisGet<Receipt>(statusKey);
    if (old?.status === "ready") return json(200, { protocol_version: PROTOCOL_VERSION, status: "already_finalised", submission_id: old.submission_id });
    const active = await getLeaseById(body.lease_id);
    const lease = active || old?.lease;
    if (!lease || lease.contributor.toLowerCase() !== identity.contributor.toLowerCase()) throw new BrokerError(409, "Lease is absent or belongs to another contributor.");
    if (!active && old?.status !== "manifest_uploaded") throw new BrokerError(409, "Lease is absent, expired, or belongs to another worker.");
    if (Date.parse(lease.expires_at) <= Date.now()) throw new BrokerError(409, "Lease is absent, expired, or belongs to another worker.");
    let receipt: Receipt = old || { status: "validating", submission_id: lease.lease_id, lease, entries: [], manifest_path: `volunteer/manifests/${lease.lease_id}.json` };
    if (receipt.status !== "manifest_uploaded") {
      const raw = await redis("SMEMBERS", `euroeval:worker:results:${lease.lease_id}`);
      const entries = resultEntries(raw);
      const expected = lease.expected_scope?.identity_suffixes;
      if (!Array.isArray(expected) || !expected.length) throw new ConfigurationError("Lease has no trusted expected scope.");
      const actual = entries.map((entry) => suffix(entry.identity));
      if (actual.some((item) => item === null) || new Set(actual).size !== entries.length ||
          actual.length !== expected.length || expected.some((item: string) => !actual.includes(item))) throw new BrokerError(422, "The uploaded result set does not match the coordinator's expected scope.");
      receipt.entries = entries; receipt.status = "validating";
      await redis("SET", statusKey, JSON.stringify(receipt), "EX", String(30 * 24 * 60 * 60));
      const issue = await fetchIssue(lease.issue_number);
      const manifest = {
        protocol_version: PROTOCOL_VERSION, submission_id: receipt.submission_id, issue_number: lease.issue_number,
        verified_contributor: identity.contributor, model: { id: lease.model_id, revision: lease.model_revision },
        language: lease.language, model_profile: lease.model_profile, language_group: lease.expected_scope.language_group,
        euroeval_version: lease.euroeval_version, worker_version: lease.worker_version, image_digest: lease.image_digest,
        image_digest_provenance: "configured-required-not-runtime-attested", gpu_memory_utilisation: lease.gpu_memory_utilisation,
        selected_gpu_index: lease.selected_gpu_index, selected_gpu_uuid: lease.selected_gpu_uuid,
        expected_scope: lease.expected_scope, results: entries,
        automated_checks: { result_count: entries.length, identities_unique: true, failed_instances: 0, warnings: [...new Set([...(lease.expected_scope.warnings || []), ...entries.flatMap((entry) => entry.warnings || [])])] },
        created_at: new Date().toISOString(), issue_state: issue.state,
      };
      receipt.manifest = JSON.stringify(manifest); await uploadStaging(receipt.manifest_path, receipt.manifest);
      receipt.status = "manifest_uploaded";
      await redis("SET", statusKey, JSON.stringify(receipt), "EX", String(30 * 24 * 60 * 60));
    }
    if (Date.parse(lease.expires_at) <= Date.now()) throw new BrokerError(409, "Lease is absent, expired, or belongs to another worker.");
    const mutex = await acquireRenewableIssueMutex(lease.issue_number); if (!mutex) throw new BrokerError(409, "Issue is busy; retry finalisation.");
    try {
      const issue = await fetchIssue(lease.issue_number); const marker = parseVolunteerMarker(issue.body);
      if (!marker || !(await verifyVolunteerMarker(issue.number, marker))) throw new BrokerError(409, "The issue ownership marker is missing, unsigned, or malformed.");
      if (!selectedLanguages(issue.body).includes(lease.language)) throw new BrokerError(409, "The leased language is no longer in the issue scope.");
      const ours = marker.leases.some((item) => item.lease_id === lease.lease_id);
      const alreadySubmitted = marker.submissions?.some((item) => item.submission_id === receipt.submission_id);
      if (!ours && !alreadySubmitted) throw new BrokerError(409, "The issue no longer carries this worker's lease marker.");
      const submission = { submission_id: receipt.submission_id, language: lease.language, manifest_path: receipt.manifest_path, submitted_at: new Date().toISOString(), verified_contributor: identity.contributor, result_count: receipt.entries.length, status: "submitted" as const };
      const submissions = alreadySubmitted ? marker.submissions || [] : [...(marker.submissions || []), submission];
      const next: VolunteerLeaseMarker = { ...marker, submission: "submitted", leases: marker.leases.filter((item) => item.lease_id !== lease.lease_id), submissions };
      if (ours) {
        const signed = await signVolunteerMarker(lease.issue_number, next);
        if (Date.parse(lease.expires_at) <= Date.now()) throw new BrokerError(409, "Lease is absent, expired, or belongs to another worker.");
        await mutex.assertOwned();
        await patchIssue(lease.issue_number, replaceVolunteerMarker(issue.body || "", signed));
        const fencedIssue = await fetchIssue(lease.issue_number); const fenced = parseVolunteerMarker(fencedIssue.body);
        if (!fenced || !(await verifyVolunteerMarker(lease.issue_number, fenced)) || !fenced.submissions?.some((item) => item.submission_id === receipt.submission_id)) throw new BrokerError(409, "GitHub submission fence lost.");
      }
      const label = process.env.COMMUNITY_REVIEW_LABEL || "community-review-ready";
      if (!issue.labels?.some((item) => item.name === label)) {
        await mutex.assertOwned();
        await addIssueLabel(lease.issue_number, label);
      }
      const comments = await issueComments(lease.issue_number);
      if (!comments.some((item) => item.body?.includes(`${FINAL_MARKER} ${receipt.submission_id}`))) {
        const maintainer = process.env.COMMUNITY_MAINTAINER_LOGIN || "saattrupdan";
        const comment = `<!-- ${FINAL_MARKER} ${receipt.submission_id} -->\n@${maintainer} — community evaluation ready.\n\nContributor: **${contributorLabel(identity.contributor)}**  \nModel: **${lease.model_id}@${lease.model_revision}**  \nLanguage: **${lease.language}**  \nSubmission: **${receipt.submission_id}**  \nValidation: **${receipt.entries.length} identities; zero failed instances**`;
        await mutex.assertOwned();
        await commentIssue(lease.issue_number, comment);
      }
      await deleteLease(lease); receipt.status = "ready"; await redis("SET", statusKey, JSON.stringify(receipt), "EX", String(30 * 24 * 60 * 60));
    } finally { await mutex.release(); }
    return json(200, { protocol_version: PROTOCOL_VERSION, status: "ready", submission_id: receipt.submission_id, coverage: { language: lease.language, records: receipt.entries.length }, manifest_path: receipt.manifest_path });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to finalise submission." });
  }
}
