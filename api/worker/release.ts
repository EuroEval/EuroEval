import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, authenticate, deleteLease, env, fetchIssue,
  getLeaseById, json, method, parseVolunteerMarker, patchIssue, readJson, replaceVolunteerMarker,
  requireProtocol, unassignIssue,
} from "./_lib";

export const config = { runtime: "edge" };

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const identity = await authenticate(req);
    const body = await readJson(req, 8 * 1024);
    requireProtocol(body);
    if (typeof body.lease_id !== "string") throw new BrokerError(400, "lease_id is required.");
    const lease = await getLeaseById(body.lease_id);
    if (!lease || lease.worker !== identity.hash.slice(0, 24)) throw new BrokerError(409, "Lease is absent or belongs to another worker.");
    const coordinator = env("WORKER_COORDINATOR_LOGIN");
    const issue = await fetchIssue(lease.issue_number);
    const marker = parseVolunteerMarker(issue.body);
    if (marker && marker.leases.some((item) => item.lease_id === lease.lease_id)) {
      const remaining = marker.leases.filter((item) => item.lease_id !== lease.lease_id && Date.parse(item.expires_at) > Date.now());
      await patchIssue(lease.issue_number, replaceVolunteerMarker(issue.body || "", remaining.length ? { ...marker, leases: remaining, submission: "active" } : null));
      if (!remaining.length && (issue.assignees || []).some((item) => item.login === coordinator)) await unassignIssue(lease.issue_number, coordinator);
    }
    // Do not report success if the GitHub fence failed. Keeping Redis state lets
    // a retry finish cleanup instead of making local state falsely look released.
    await deleteLease(lease);
    return json(200, { protocol_version: PROTOCOL_VERSION, status: "released", lease_id: lease.lease_id });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to release lease." });
  }
}
