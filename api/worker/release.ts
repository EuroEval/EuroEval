import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, acquireRenewableIssueMutex, authenticate,
  brokerErrorBody, deleteLease, enforceRateLimit, fetchIssue, getLeaseById, json, method,
  parseVolunteerMarker, patchIssue, readJson, replaceVolunteerMarker, VOLUNTEER_MARKER_RE,
  requireProtocol, signVolunteerMarker, unassignIssue, verifyVolunteerMarker,
  reconcilePublicAssignments,
} from "./_lib.js";
import type { VolunteerLeaseMarker } from "./_lib.js";

export const config = { runtime: "edge" };

function expectedAssignees(marker: VolunteerLeaseMarker): Set<string> {
  const expected = new Set(marker.leases.filter((item) => item.ownership_mode !== "public").map((item) => item.contributor.toLowerCase()));
  for (const item of marker.submissions || []) {
    if (["submitted", "accepted"].includes(item.status) && item.ownership_mode !== "public") {
      expected.add(item.verified_contributor.toLowerCase());
    }
  }
  return expected;
}

function assigneesMatch(
  assignees: Array<{ login: string }> | undefined,
  expected: Set<string>,
  allowAdditional = false,
): boolean {
  const actual = new Set((assignees || []).map((item) => item.login.toLowerCase()));
  return [...expected].every((login) => actual.has(login)) &&
    (allowAdditional || actual.size === expected.size);
}

function hasPublicOwnership(marker: VolunteerLeaseMarker): boolean {
  return marker.leases.some((lease) => lease.ownership_mode === "public") ||
    (marker.submissions || []).some((submission) => submission.ownership_mode === "public") ||
    !!marker.public_assignees?.length;
}

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
        Date.parse(lease.expires_at) <= Date.now()) {
      throw new BrokerError(409, "Lease is absent, expired, or belongs to another contributor.");
    }
    const mutex = await acquireRenewableIssueMutex(lease.issue_number);
    if (!mutex) throw new BrokerError(409, "Issue is busy; retry release.");
    try {
      let current = await fetchIssue(lease.issue_number);
      let marker = parseVolunteerMarker(current.body);
      if (!marker) {
        if (VOLUNTEER_MARKER_RE.test(current.body || "")) {
          throw new BrokerError(409, "GitHub ownership marker is missing, unsigned, or malformed.");
        }
        // Legacy release retries may have completed the GitHub mutation before
        // Redis cleanup. Public leases require their signed marker as ownership proof.
        if (lease.ownership_mode === "public") {
          throw new BrokerError(409, "GitHub ownership marker is missing, unsigned, or malformed.");
        }
        await deleteLease(lease);
        return json(200, { protocol_version: PROTOCOL_VERSION, status: "released", lease_id: lease.lease_id });
      }
      if (!(await verifyVolunteerMarker(lease.issue_number, marker))) {
        throw new BrokerError(409, "GitHub ownership marker is missing, unsigned, or malformed.");
      }
      if (!marker.leases.some((item) => item.lease_id === lease.lease_id)) {
        // The marker transition committed before Redis cleanup. Preserve any
        // other leases, submissions, or history and only finish our cleanup.
        await deleteLease(lease);
        return json(200, { protocol_version: PROTOCOL_VERSION, status: "released", lease_id: lease.lease_id });
      }
      const contributor = lease.contributor.toLowerCase();
      const ownedMarkerLease = marker.leases.find((item) => item.lease_id === lease.lease_id);
      if (!ownedMarkerLease || ownedMarkerLease.contributor.toLowerCase() !== contributor ||
          (ownedMarkerLease.ownership_mode || "assignee") !== (lease.ownership_mode || "assignee")) {
        throw new BrokerError(409, "GitHub release ownership fence lost.");
      }
      const markerExpected = expectedAssignees(marker);
      const retained = marker.leases.some((item) => item.lease_id !== lease.lease_id &&
          item.ownership_mode !== "public" && item.contributor.toLowerCase() === contributor) ||
        (marker.submissions || []).some((item) => ["submitted", "accepted"].includes(item.status) &&
          item.ownership_mode !== "public" && item.verified_contributor.toLowerCase() === contributor);
      const withoutContributor = new Set(markerExpected);
      if (!retained) withoutContributor.delete(contributor);
      if (!assigneesMatch(current.assignees, markerExpected, hasPublicOwnership(marker)) &&
          (retained || !assigneesMatch(current.assignees, withoutContributor, hasPublicOwnership(marker)))) {
        throw new BrokerError(409, "GitHub release assignment changed; retry release.");
      }

      const assigned = (current.assignees || []).find(
        (item) => item.login.toLowerCase() === contributor,
      );
      if (lease.ownership_mode !== "public" && assigned && !retained) {
        await mutex.assertOwned();
        try {
          await unassignIssue(lease.issue_number, assigned.login);
        } catch (error) {
          const observed = await fetchIssue(lease.issue_number).catch(() => null);
          if (!observed || (observed.assignees || []).some(
            (item) => item.login.toLowerCase() === contributor,
          )) throw error;
          current = observed;
        }
        current = await fetchIssue(lease.issue_number);
        marker = parseVolunteerMarker(current.body);
        if (!marker || !(await verifyVolunteerMarker(lease.issue_number, marker)) ||
            !marker.leases.some((item) => item.lease_id === lease.lease_id) ||
            (current.assignees || []).some((item) => item.login.toLowerCase() === contributor)) {
          throw new BrokerError(409, "GitHub release fence lost.");
        }
      }

      current = await fetchIssue(lease.issue_number);
      marker = parseVolunteerMarker(current.body);
      if (!marker || !(await verifyVolunteerMarker(lease.issue_number, marker)) ||
          !marker.leases.some((item) => item.lease_id === lease.lease_id)) {
        throw new BrokerError(409, "GitHub release fence lost.");
      }
      const remaining = marker.leases.filter((item) => item.lease_id !== lease.lease_id);
      const nextExpected = expectedAssignees({ ...marker, leases: remaining });
      if (!assigneesMatch(current.assignees, nextExpected, hasPublicOwnership(marker))) {
        throw new BrokerError(409, "GitHub release assignment changed; retry release.");
      }
      const nextMarker = remaining.length || marker.submissions?.length || marker.completed_languages?.length ||
          marker.public_assignees?.length
        ? { ...marker, submission: marker.submissions?.length ? "submitted" as const : "active" as const, leases: remaining }
        : null;
      await mutex.assertOwned();
      try {
        const signed = nextMarker && await signVolunteerMarker(lease.issue_number, nextMarker);
        await patchIssue(lease.issue_number, replaceVolunteerMarker(current.body || "", signed));
      } catch (error) {
        const observed = await fetchIssue(lease.issue_number).catch(() => null);
        const observedMarker = observed && parseVolunteerMarker(observed.body);
        if (!observed || observedMarker?.leases.some((item) => item.lease_id === lease.lease_id)) throw error;
        current = observed;
      }
      current = await fetchIssue(lease.issue_number);
      marker = parseVolunteerMarker(current.body);
      if (marker?.leases.some((item) => item.lease_id === lease.lease_id) ||
          marker && !(await verifyVolunteerMarker(lease.issue_number, marker))) {
        throw new BrokerError(409, "GitHub release fence lost.");
      }
      const finalExpected = marker ? expectedAssignees(marker) : new Set<string>();
      if (!assigneesMatch(current.assignees, finalExpected, marker ? hasPublicOwnership(marker) : false)) {
        throw new BrokerError(409, "GitHub release assignment changed; retry release.");
      }
      if (marker) await reconcilePublicAssignments(current, marker, () => mutex.assertOwned());
      await deleteLease(lease);
    } finally { await mutex.release(); }
    return json(200, { protocol_version: PROTOCOL_VERSION, status: "released", lease_id: lease.lease_id });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, brokerErrorBody(error, "Unable to release lease."));
  }
}
