import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, acquireIssueMutex, authenticate,
  deleteLease, env, enforceRateLimit, fetchIssue, getLeaseById, json, method,
  parseVolunteerMarker, patchIssue, readJson, releaseIssueMutex, replaceVolunteerMarker,
  requireProtocol, unassignIssue,
} from "./_lib";

export const config = { runtime: "edge" };

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const identity = await authenticate(req); await enforceRateLimit(`euroeval:worker:limit:release:${identity.hash}`, 60, 3600);
    const body = await readJson(req, 8 * 1024); requireProtocol(body);
    if (typeof body.lease_id !== "string") throw new BrokerError(400, "lease_id is required.");
    const lease = await getLeaseById(body.lease_id);
    if (!lease || lease.contributor.toLowerCase() !== identity.contributor.toLowerCase()) throw new BrokerError(409, "Lease is absent or belongs to another contributor.");
    const mutex = await acquireIssueMutex(lease.issue_number); if (!mutex) throw new BrokerError(409, "Issue is busy; retry release.");
    try {
      const coordinator = env("WORKER_COORDINATOR_LOGIN"); const issue = await fetchIssue(lease.issue_number); const marker = parseVolunteerMarker(issue.body);
      if (marker?.leases.some((item) => item.lease_id === lease.lease_id)) {
        const remaining = marker.leases.filter((item) => item.lease_id !== lease.lease_id && Date.parse(item.expires_at) > Date.now());
        const keep = (marker.submissions?.length || marker.completed_languages?.length || remaining.length);
        if (keep) await patchIssue(lease.issue_number, replaceVolunteerMarker(issue.body || "", { ...marker, leases: remaining, submission: marker.submissions?.length ? "submitted" : "active" }));
        else { const without = (issue.body || "").replace(/<!--[\s\S]*?euroeval-volunteer-worker:v1\s+[\s\S]*?-->/i, "").replace(/\n{3,}/g, "\n\n").trimEnd() + "\n"; await patchIssue(lease.issue_number, without); }
        const after = await fetchIssue(lease.issue_number); const afterMarker = parseVolunteerMarker(after.body);
        if (keep && !afterMarker) throw new BrokerError(409, "GitHub release fence lost.");
        if (!keep && (after.body || "").includes("euroeval-volunteer-worker:v1")) throw new BrokerError(409, "GitHub release fence lost.");
        if (!keep && (after.assignees || []).some((item) => item.login === coordinator)) await unassignIssue(lease.issue_number, coordinator);
      }
      await deleteLease(lease);
    } finally { await releaseIssueMutex(lease.issue_number, mutex); }
    return json(200, { protocol_version: PROTOCOL_VERSION, status: "released", lease_id: lease.lease_id });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to release lease." });
  }
}
