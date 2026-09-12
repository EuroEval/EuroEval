import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, acquireRenewableIssueMutex, authenticate,
  brokerErrorBody, deleteLease, enforceRateLimit, fetchIssue, getLeaseById, json, method,
  parseVolunteerMarker, patchIssue, readJson, replaceVolunteerMarker, requireAssignee, VOLUNTEER_MARKER_RE,
  requireProtocol, signVolunteerMarker, unassignIssue, verifyVolunteerMarker,
} from "./_lib.ts";

export const config = { runtime: "edge" };

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const identity = await authenticate(req); await enforceRateLimit(`euroeval:worker:limit:release:${identity.hash}`, 60, 3600);
    const body = await readJson(req, 8 * 1024); requireProtocol(body);
    if (typeof body.lease_id !== "string") throw new BrokerError(400, "lease_id is required.");
    const lease = await getLeaseById(body.lease_id);
    if (lease?.released && lease.contributor.toLowerCase() === identity.contributor.toLowerCase()) {
      return json(200, { protocol_version: PROTOCOL_VERSION, status: "released", lease_id: lease.lease_id });
    }
    if (!lease || lease.released || lease.contributor.toLowerCase() !== identity.contributor.toLowerCase() ||
        Date.parse(lease.expires_at) <= Date.now()) throw new BrokerError(409, "Lease is absent, expired, or belongs to another contributor.");
    const mutex = await acquireRenewableIssueMutex(lease.issue_number); if (!mutex) throw new BrokerError(409, "Issue is busy; retry release.");
    try {
      const issue = await fetchIssue(lease.issue_number);
      await requireAssignee(issue, lease.contributor);
      const marker = parseVolunteerMarker(issue.body);
      if (!marker || !(await verifyVolunteerMarker(lease.issue_number, marker))) throw new BrokerError(409, "GitHub ownership marker is missing, unsigned, or malformed.");
      if (!marker.leases.some((item) => item.lease_id === lease.lease_id)) throw new BrokerError(409, "GitHub ownership marker no longer carries this lease.");
      const remaining = marker.leases.filter((item) => item.lease_id !== lease.lease_id);
      const contributor = lease.contributor.toLowerCase();
      const retained = remaining.some((item) => Date.parse(item.expires_at) > Date.now() && item.contributor.toLowerCase() === contributor) ||
        (marker.submissions || []).some((item) => ["submitted", "accepted"].includes(item.status) && item.verified_contributor.toLowerCase() === contributor);
      const keep = Boolean(marker.submissions?.length || marker.completed_languages?.length || remaining.length);
      if (keep) {
        const nextMarker = {
          ...marker,
          submission: marker.submissions?.length ? "submitted" as const : "active" as const,
          leases: remaining,
        };
        const signed = await signVolunteerMarker(lease.issue_number, nextMarker);
        if (Date.parse(lease.expires_at) <= Date.now()) {
          throw new BrokerError(409, "Lease is absent, expired, or belongs to another contributor.");
        }
        await mutex.assertOwned();
        await patchIssue(lease.issue_number, replaceVolunteerMarker(issue.body || "", signed));
        const after = await fetchIssue(lease.issue_number); const afterMarker = parseVolunteerMarker(after.body);
        if (!afterMarker || !(await verifyVolunteerMarker(lease.issue_number, afterMarker)) ||
            afterMarker.leases.some((item) => item.lease_id === lease.lease_id)) throw new BrokerError(409, "GitHub release fence lost.");
        await requireAssignee(after, lease.contributor);
      } else {
        const without = (issue.body || "").replace(VOLUNTEER_MARKER_RE, "").replace(/\n{3,}/g, "\n\n").trimEnd() + "\n";
        if (Date.parse(lease.expires_at) <= Date.now()) {
          throw new BrokerError(409, "Lease is absent, expired, or belongs to another contributor.");
        }
        await mutex.assertOwned();
        await patchIssue(lease.issue_number, without);
        const after = await fetchIssue(lease.issue_number);
        if (VOLUNTEER_MARKER_RE.test(after.body || "")) throw new BrokerError(409, "GitHub release fence lost.");
        await requireAssignee(after, lease.contributor);
      }
      if (!retained) {
        const current = await fetchIssue(lease.issue_number);
        await requireAssignee(current, lease.contributor);
        const assigned = (current.assignees || []).find((item) => item.login.toLowerCase() === contributor);
        if (assigned) {
          await mutex.assertOwned();
          await unassignIssue(lease.issue_number, assigned.login);
          const afterAssignment = await fetchIssue(lease.issue_number);
          if ((afterAssignment.assignees || []).some((item) => item.login.toLowerCase() === contributor)) {
            throw new BrokerError(409, "GitHub release fence lost.");
          }
        }
      }
      await deleteLease(lease);
    } finally { await mutex.release(); }
    return json(200, { protocol_version: PROTOCOL_VERSION, status: "released", lease_id: lease.lease_id });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, brokerErrorBody(error, "Unable to release lease."));
  }
}
