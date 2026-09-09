import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, acquireIssueMutex, authenticate,
  enforceRateLimit, fetchIssue, getLeaseById, json, leaseTtl, method, parseVolunteerMarker,
  patchIssue, readJson, requireProtocol, replaceVolunteerMarker, releaseIssueMutex, saveLease,
} from "./_lib";

export const config = { runtime: "edge" };

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const identity = await authenticate(req); await enforceRateLimit(`euroeval:worker:limit:heartbeat:${identity.hash}`, 240, 3600);
    const body = await readJson(req, 8 * 1024); requireProtocol(body);
    if (typeof body.lease_id !== "string") throw new BrokerError(400, "lease_id is required.");
    const lease = await getLeaseById(body.lease_id);
    if (!lease || lease.contributor.toLowerCase() !== identity.contributor.toLowerCase() || Date.parse(lease.expires_at) <= Date.now()) throw new BrokerError(409, "Lease is absent, expired, or belongs to another contributor.");
    const markerIssue = await fetchIssue(lease.issue_number); const marker = parseVolunteerMarker(markerIssue.body);
    const markerLease = marker?.leases.find((item) => item.lease_id === lease.lease_id);
    if (!marker || !markerLease) throw new BrokerError(409, "GitHub ownership marker was lost.");
    const markerExpiry = Date.parse(markerLease.expires_at);
    lease.expires_at = new Date(Date.now() + leaseTtl() * 1000).toISOString();
    if (!(await saveLease(lease))) throw new BrokerError(409, "Lease was replaced or expired before it could be renewed.");
    // The issue marker is a recovery fence. Extend from its own expiry, rather
    // than from the newly renewed Redis expiry, so it cannot drift indefinitely.
    if (markerExpiry - Date.now() < leaseTtl() * 500) {
      const mutex = await acquireIssueMutex(lease.issue_number);
      if (!mutex) throw new BrokerError(409, "Issue is busy; retry heartbeat.");
      try {
        const current = await fetchIssue(lease.issue_number); const currentMarker = parseVolunteerMarker(current.body);
        const currentLease = currentMarker?.leases.find((item) => item.lease_id === lease.lease_id);
        if (!currentMarker || !currentLease) throw new BrokerError(409, "GitHub ownership marker was lost.");
        const newExpiry = new Date(Math.max(Date.parse(currentLease.expires_at), Date.now()) + leaseTtl() * 1000).toISOString();
        await patchIssue(lease.issue_number, replaceVolunteerMarker(current.body || "", { ...currentMarker, leases: currentMarker.leases.map((item) => item.lease_id === lease.lease_id ? { ...item, expires_at: newExpiry } : item) }));
      } finally { await releaseIssueMutex(lease.issue_number, mutex); }
    }
    return json(200, { protocol_version: PROTOCOL_VERSION, lease_id: lease.lease_id, expires_at: lease.expires_at });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to renew lease." });
  }
}
