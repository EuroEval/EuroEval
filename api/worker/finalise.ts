declare const process: { env: Record<string, string | undefined> };

import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, addIssueLabel, authenticate, commentIssue,
  contributorLabel, deleteLease, fetchIssue, getLeaseById, issueComments, json, method, patchIssue,
  parseVolunteerMarker, readJson, redis, redisGet, replaceVolunteerMarker, requireProtocol, selectedLanguages,
  unassignIssue,
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

function trustedScope(lease: { model_id: string; language: string }): Set<string> | null {
  const raw = process.env.VOLUNTEER_EXPECTED_SCOPE_JSON;
  if (!raw) return null;
  try {
    const scope = JSON.parse(raw) as Record<string, unknown>;
    const identities = scope[`${lease.model_id}:${lease.language}`];
    if (!Array.isArray(identities) || !identities.every((item) => typeof item === "string")) return null;
    return new Set(identities as string[]);
  } catch { return null; }
}

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const identity = await authenticate(req);
    const body = await readJson(req, 32 * 1024);
    requireProtocol(body);
    if (typeof body.lease_id !== "string") throw new BrokerError(400, "lease_id is required.");
    const lease = await getLeaseById(body.lease_id);
    if (!lease || lease.worker !== identity.hash.slice(0, 24) || Date.parse(lease.expires_at) <= Date.now()) throw new BrokerError(409, "Lease is absent, expired, or belongs to another worker.");
    const raw = await redis("SMEMBERS", `euroeval:worker:results:${lease.lease_id}`);
    const entries = resultEntries(raw);
    const scope = trustedScope(lease);
    if (!scope) throw new BrokerError(503, "The coordinator has no trusted expected-scope manifest for this lease.");
    const actual = new Set(entries.map((entry) => entry.identity));
    if (actual.size !== entries.length || actual.size !== scope.size || [...scope].some((item) => !actual.has(item))) throw new BrokerError(422, "The uploaded result set does not match the coordinator's expected scope.");
    const statusKey = `euroeval:worker:finalisation:${lease.lease_id}`;
    const oldStatus = await redisGet<{ status?: string; submission_id?: string }>(statusKey);
    if (oldStatus?.status === "ready") return json(200, { protocol_version: PROTOCOL_VERSION, status: "already_finalised", submission_id: oldStatus.submission_id });
    await redis("SET", statusKey, JSON.stringify({ status: "validating", count: entries.length }), "EX", String(30 * 24 * 60 * 60));
    const issue = await fetchIssue(lease.issue_number);
    if (!selectedLanguages(issue.body).includes(lease.language)) throw new BrokerError(409, "The leased language is no longer in the issue scope.");
    const marker = parseVolunteerMarker(issue.body);
    if (!marker?.leases.some((item) => item.lease_id === lease.lease_id)) throw new BrokerError(409, "The issue no longer carries this worker's lease marker.");
    const label = process.env.COMMUNITY_REVIEW_LABEL || "community-review-ready";
    if (!issue.labels?.some((item) => item.name === label)) await addIssueLabel(lease.issue_number, label);
    const comments = await issueComments(lease.issue_number);
    if (!comments.some((item) => item.body?.includes(FINAL_MARKER))) {
      const summary = `${entries.length} canonical identities; zero failed instances; coordinator scope passed.`;
      const comment = `${FINAL_MARKER}\n@${"saattrupdan"} — community evaluation ready.\n\n` +
        `Contributor: **${contributorLabel(identity.contributor)}**  \nModel: **${lease.model_id}@${lease.model_revision}**  \nCoverage: **${lease.language}** (${summary})`;
      await commentIssue(lease.issue_number, comment);
    }
    const remaining = marker.leases.filter((item) => item.lease_id !== lease.lease_id && Date.parse(item.expires_at) > Date.now());
    await patchIssue(lease.issue_number, replaceVolunteerMarker(issue.body || "", remaining.length ? { ...marker, leases: remaining, submission: "completed" } : null));
    if (!remaining.length && issue.assignees?.some((item) => item.login === marker.coordinator)) await unassignIssue(lease.issue_number, marker.coordinator);
    await deleteLease(lease);
    await redis("SET", statusKey, JSON.stringify({ status: "ready", count: entries.length, validated_at: new Date().toISOString() }), "EX", String(30 * 24 * 60 * 60));
    return json(200, { protocol_version: PROTOCOL_VERSION, status: "ready", coverage: { language: lease.language, records: entries.length } });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to finalise submission." });
  }
}
