import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, acquireIssueMutex, authenticate,
  enforceRateLimit, fetchIssue, getLeaseById, json, leaseTtl, method, parseVolunteerMarker,
  patchIssue, readJson, requireProtocol, replaceVolunteerMarker, releaseIssueMutex,
  saveLease, signVolunteerMarker, verifyVolunteerMarker,
} from "./_lib.ts";

export const config = { runtime: "edge" };

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const identity = await authenticate(req); await enforceRateLimit(`euroeval:worker:limit:heartbeat:${identity.hash}`, 240, 3600);
    const body = await readJson(req, 8 * 1024); requireProtocol(body);
    if (typeof body.lease_id !== "string") throw new BrokerError(400, "lease_id is required.");
    const lease = await getLeaseById(body.lease_id);
    if (!lease || lease.contributor.toLowerCase() !== identity.contributor.toLowerCase() || Date.parse(lease.expires_at) <= Date.now()) throw new BrokerError(409, "Lease is absent, expired, or belongs to another contributor.");
    const mutex = await acquireIssueMutex(lease.issue_number);
    if (!mutex) throw new BrokerError(409, "Issue is busy; retry heartbeat.");
    try {
      const issue = await fetchIssue(lease.issue_number);
      const marker = parseVolunteerMarker(issue.body);
      const markerLease = marker?.leases.find((item) => item.lease_id === lease.lease_id);
      if (!marker || !markerLease || !(await verifyVolunteerMarker(issue.number, marker))) throw new BrokerError(409, "GitHub ownership marker is missing, unsigned, or malformed.");
      const expiresAt = new Date(Date.now() + leaseTtl() * 1000).toISOString();
      const nextMarker = {
        ...marker,
        leases: marker.leases.map((item) => item.lease_id === lease.lease_id ? { ...item, expires_at: expiresAt } : item),
      };
      const signedMarker = await signVolunteerMarker(lease.issue_number, nextMarker);
      await patchIssue(lease.issue_number, replaceVolunteerMarker(issue.body || "", signedMarker));
      const fencedIssue = await fetchIssue(lease.issue_number);
      const fenced = parseVolunteerMarker(fencedIssue.body);
      if (!fenced || !(await verifyVolunteerMarker(lease.issue_number, fenced)) ||
          !fenced.leases.some((item) => item.lease_id === lease.lease_id && item.expires_at === expiresAt)) {
        throw new BrokerError(409, "GitHub heartbeat fence lost.");
      }
      lease.expires_at = expiresAt;
      if (!(await saveLease(lease))) throw new BrokerError(409, "Lease was replaced or expired before it could be renewed.");
      return json(200, { protocol_version: PROTOCOL_VERSION, lease_id: lease.lease_id, expires_at: expiresAt });
    } finally { await releaseIssueMutex(lease.issue_number, mutex); }
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to renew lease." });
  }
}
