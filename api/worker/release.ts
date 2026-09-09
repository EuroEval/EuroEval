import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, authenticate, fetchIssue, getLeaseById, json, method,
  patchIssue, readJson, deleteLease, replaceVolunteerMarker, parseVolunteerMarker,
  env, unassignIssue,
} from "./_lib";

export const config = { runtime: "edge" };

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const identity = await authenticate(req);
    const body = await readJson(req, 8 * 1024);
    if (typeof body.lease_id !== "string") throw new BrokerError(400, "lease_id is required.");
    const lease = await getLeaseById(body.lease_id);
    if (!lease || lease.worker !== identity.hash.slice(0, 24)) throw new BrokerError(409, "Lease is absent or belongs to another worker.");
    await deleteLease(lease);
    const coordinator = env("WORKER_COORDINATOR_LOGIN");
    const issue = await fetchIssue(lease.issue_number);
    const marker = parseVolunteerMarker(issue.body);
    if (marker) {
      const remaining = marker.leases.filter((item) => !(item.worker === lease.worker && item.language === lease.language));
      const live = remaining.filter((item) => item.expires_at > Date.now());
      // Re-fetch immediately before changing the body, preserving a competing lease.
      const latest = await fetchIssue(lease.issue_number);
      const latestMarker = parseVolunteerMarker(latest.body);
      if (latestMarker && latestMarker.leases.some((item) => item.worker === lease.worker && item.language === lease.language)) {
        const latestRemaining = latestMarker.leases.filter((item) => !(item.worker === lease.worker && item.language === lease.language) && item.expires_at > Date.now());
        await patchIssue(lease.issue_number, replaceVolunteerMarker(latest.body || "", latestRemaining.length ? { winner: latestMarker.winner, leases: latestRemaining } : null));
        if (!latestRemaining.length && (latest.assignees || []).some((item) => item.login === coordinator)) await unassignIssue(lease.issue_number, coordinator);
      }
    }
    return json(200, { protocol_version: PROTOCOL_VERSION, status: "released", lease_id: lease.lease_id });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { error: error instanceof Error ? error.message : "Unable to release lease." });
  }
}
