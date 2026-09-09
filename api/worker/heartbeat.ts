import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, authenticate, fetchIssue, getLeaseById,
  json, leaseTtl, method, parseVolunteerMarker, patchIssue, readJson, requireProtocol,
  replaceVolunteerMarker, saveLease,
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
    if (!lease || lease.worker !== identity.hash.slice(0, 24) || Date.parse(lease.expires_at) <= Date.now()) throw new BrokerError(409, "Lease is absent, expired, or belongs to another worker.");
    const oldExpiry = Date.parse(lease.expires_at);
    lease.expires_at = new Date(Date.now() + leaseTtl() * 1000).toISOString();
    if (!(await saveLease(lease))) throw new BrokerError(409, "Lease was replaced or expired before it could be renewed.");
    // GitHub is deliberately written at most once per half-TTL. Redis is the
    // heartbeat source of truth; this keeps issue ownership recoverable without
    // turning a long evaluation into a GitHub write storm.
    if (oldExpiry - Date.now() < leaseTtl() * 500) {
      const issue = await fetchIssue(lease.issue_number);
      const marker = parseVolunteerMarker(issue.body);
      if (!marker || !marker.leases.some((item) => item.lease_id === lease.lease_id)) throw new BrokerError(409, "GitHub ownership marker was lost.");
      const leases = marker.leases.map((item) => item.lease_id === lease.lease_id ? { ...item, expires_at: lease.expires_at } : item);
      await patchIssue(lease.issue_number, replaceVolunteerMarker(issue.body || "", { ...marker, leases }));
    }
    return json(200, { protocol_version: PROTOCOL_VERSION, lease_id: lease.lease_id, expires_at: lease.expires_at });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to renew lease." });
  }
}
