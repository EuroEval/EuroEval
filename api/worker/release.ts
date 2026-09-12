import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, acquireRenewableIssueMutex, authenticate,
  brokerErrorBody, deleteLease, enforceRateLimit, fetchIssue, getLeaseById, hasIssueAssignee, json, method,
  parseVolunteerMarker, patchIssue, readJson, replaceVolunteerMarker, VOLUNTEER_MARKER_RE,
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
    if (!lease || lease.contributor.toLowerCase() !== identity.contributor.toLowerCase()) {
      throw new BrokerError(409, "Lease is absent, expired, or belongs to another contributor.");
    }
    const mutex = await acquireRenewableIssueMutex(lease.issue_number); if (!mutex) throw new BrokerError(409, "Issue is busy; retry release.");
    try {
      const issue = await fetchIssue(lease.issue_number);
      const marker = parseVolunteerMarker(issue.body);
      if (VOLUNTEER_MARKER_RE.test(issue.body || "") &&
          (!marker || !(await verifyVolunteerMarker(lease.issue_number, marker)))) {
        throw new BrokerError(409, "GitHub ownership marker is missing, unsigned, or malformed.");
      }
      const contributor = lease.contributor.toLowerCase();
      const markerContainsLease = marker?.leases.some((item) => item.lease_id === lease.lease_id) ?? false;
      if (markerContainsLease && !hasIssueAssignee(issue, lease.contributor)) {
        throw new BrokerError(409,
          "The lease contributor is no longer assigned to this issue.", "lease_assignment_lost");
      }
      let current = issue;
      let currentMarker = marker;
      if (marker && markerContainsLease) {
        const remaining = marker.leases.filter((item) => item.lease_id !== lease.lease_id);
        const retained = remaining.some((item) => Date.parse(item.expires_at) > Date.now() &&
            item.contributor.toLowerCase() === contributor) ||
          (marker.submissions || []).some((item) => ["submitted", "accepted"].includes(item.status) &&
            item.verified_contributor.toLowerCase() === contributor);
        const nextMarker = {
          ...marker,
          submission: marker.submissions?.length ? "submitted" as const : "active" as const,
          leases: remaining,
        };
        const signed = await signVolunteerMarker(lease.issue_number, nextMarker);
        await mutex.assertOwned();
        await patchIssue(lease.issue_number, replaceVolunteerMarker(issue.body || "", signed));
        current = await fetchIssue(lease.issue_number);
        currentMarker = parseVolunteerMarker(current.body);
        if (!currentMarker || !(await verifyVolunteerMarker(lease.issue_number, currentMarker)) ||
            currentMarker.leases.some((item) => item.lease_id === lease.lease_id)) {
          throw new BrokerError(409, "GitHub release fence lost.");
        }
        if (!retained) {
          const expected = new Set<string>();
          for (const item of currentMarker.leases) {
            if (Date.parse(item.expires_at) > Date.now()) expected.add(item.contributor.toLowerCase());
          }
          for (const item of currentMarker.submissions || []) {
            if (["submitted", "accepted"].includes(item.status)) {
              expected.add(item.verified_contributor.toLowerCase());
            }
          }
          const actual = new Set((current.assignees || []).map((item) => item.login.toLowerCase()));
          if (actual.has(contributor) &&
              (actual.size !== expected.size + 1 ||
               [...actual].some((login) => login !== contributor && !expected.has(login)))) {
            throw new BrokerError(409, "GitHub release assignment changed; retry release.");
          }
        }
      }
      current = await fetchIssue(lease.issue_number);
      currentMarker = parseVolunteerMarker(current.body);
      if (VOLUNTEER_MARKER_RE.test(current.body || "") &&
          (!currentMarker || !(await verifyVolunteerMarker(lease.issue_number, currentMarker)))) {
        throw new BrokerError(409, "GitHub ownership marker is missing, unsigned, or malformed.");
      }
      const assigned = (current.assignees || []).find(
        (item) => item.login.toLowerCase() === contributor,
      );
      const hasOtherAssignee = (current.assignees || []).some(
        (item) => item.login.toLowerCase() !== contributor,
      );
      const retained = currentMarker ?
        currentMarker.leases.some((item) => Date.parse(item.expires_at) > Date.now() &&
          item.contributor.toLowerCase() === contributor) ||
        (currentMarker.submissions || []).some((item) => ["submitted", "accepted"].includes(item.status) &&
          item.verified_contributor.toLowerCase() === contributor) : false;
      if (assigned && !retained && !hasOtherAssignee) {
        await mutex.assertOwned();
        await unassignIssue(lease.issue_number, assigned.login);
        current = await fetchIssue(lease.issue_number);
        if ((current.assignees || []).some((item) => item.login.toLowerCase() === contributor)) {
          throw new BrokerError(409, "GitHub release fence lost.");
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
