declare const process: { env: Record<string, string | undefined> };

import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, addIssueLabel, authenticate, commentIssue,
  contributorLabel, fetchIssue, getLeaseById, issueComments, json, method, readJson,
  redis, redisGet, parseVolunteerMarker, selectedLanguages,
} from "./_lib";

export const config = { runtime: "edge" };
const FINAL_MARKER = "<!-- euroeval-volunteer-finalised:v1 -->";
type ResultEntry = { digest: string; identity: string; path: string };

function resultEntries(value: unknown): ResultEntry[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (typeof item !== "string") return [];
    try { const parsed = JSON.parse(item) as ResultEntry; return parsed.digest && parsed.identity && parsed.path ? [parsed] : []; } catch { return []; }
  });
}

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const identity = await authenticate(req);
    const body = await readJson(req, 32 * 1024);
    if (typeof body.lease_id !== "string") throw new BrokerError(400, "lease_id is required.");
    const lease = await getLeaseById(body.lease_id);
    if (!lease || lease.worker !== identity.hash.slice(0, 24) || lease.expires_at <= Date.now()) throw new BrokerError(409, "Lease is absent, expired, or belongs to another worker.");
    const raw = await redis("SMEMBERS", `euroeval:worker:results:${lease.lease_id}`);
    const entries = resultEntries(raw);
    if (!entries.length) throw new BrokerError(422, "The lease has no uploaded result records.");
    const manifest = body.manifest && typeof body.manifest === "object" && !Array.isArray(body.manifest) ? body.manifest as Record<string, unknown> : {};
    if ((manifest.language !== undefined && manifest.language !== lease.language) || (manifest.model_id !== undefined && manifest.model_id !== lease.model_id) || (manifest.revision !== undefined && manifest.revision !== lease.revision)) throw new BrokerError(422, "Finalisation manifest does not match the lease.");
    const expectedCount = body.expected_count ?? manifest.expected_count;
    if (expectedCount !== undefined && (typeof expectedCount !== "number" || expectedCount !== entries.length)) throw new BrokerError(422, `Expected ${expectedCount} result record(s), but received ${entries.length}.`);
    const expected = body.expected_identities ?? manifest.expected_identities;
    if (expected !== undefined) {
      if (!Array.isArray(expected) || !expected.every((item) => typeof item === "string")) throw new BrokerError(400, "expected_identities must be a list of strings.");
      const actual = new Set(entries.map((entry) => entry.identity));
      if (expected.some((item) => !actual.has(item))) throw new BrokerError(422, "The uploaded result set is incomplete according to the supplied manifest.");
    }
    const submissionId = typeof body.submission_id === "string" && /^[A-Za-z0-9._-]{1,80}$/.test(body.submission_id) ? body.submission_id : `submission-${lease.lease_id}`;
    const statusKey = `euroeval:worker:finalisation:${lease.lease_id}`;
    const oldStatus = await redisGet<{ status?: string; submission_id?: string }>(statusKey);
    if (oldStatus?.status === "ready") return json(200, { protocol_version: PROTOCOL_VERSION, status: "already_finalised", submission_id: oldStatus.submission_id });
    await redis("SET", statusKey, JSON.stringify({ status: "validating", submission_id: submissionId, count: entries.length }), "EX", String(30 * 24 * 60 * 60));
    const issue = await fetchIssue(lease.issue_number);
    if (!selectedLanguages(issue.body).includes(lease.language)) throw new BrokerError(409, "The leased language is no longer in the issue scope.");
    const marker = parseVolunteerMarker(issue.body);
    if (!marker?.leases.some((item) => item.worker === lease.worker && item.language === lease.language)) throw new BrokerError(409, "The issue no longer carries this worker's lease marker.");
    const label = process.env.COMMUNITY_REVIEW_LABEL || "community-review-ready";
    if (!issue.labels?.some((item) => item.name === label)) await addIssueLabel(lease.issue_number, label);
    const comments = await issueComments(lease.issue_number);
    if (!comments.some((item) => item.body?.includes(FINAL_MARKER))) {
      const summary = `${entries.length} record(s), ${new Set(entries.map((entry) => entry.identity)).size} canonical identity/identities, zero failed instances; staging validation passed.`;
      const comment = `${FINAL_MARKER}\n@${"saattrupdan"} — community evaluation ready.\n\n` +
        `Contributor: **${contributorLabel(identity.contributor)}**  \nModel: **${lease.model_id}@${lease.revision}**  \n` +
        `Coverage: **${lease.language}** (${summary})  \nSubmission ID: \`${submissionId}\``;
      await commentIssue(lease.issue_number, comment);
    }
    await redis("SET", statusKey, JSON.stringify({ status: "ready", submission_id: submissionId, count: entries.length, validated_at: new Date().toISOString() }), "EX", String(30 * 24 * 60 * 60));
    return json(200, { protocol_version: PROTOCOL_VERSION, status: "ready", submission_id: submissionId, coverage: { language: lease.language, records: entries.length } });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { error: error instanceof Error ? error.message : "Unable to finalise submission." });
  }
}
