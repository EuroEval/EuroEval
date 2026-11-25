import { fetchIssue, patchIssue, unassignIssue, reconcilePublicAssignments } from "./github.js";
import { getLeaseForIssue, reclaimExpiredLease, acquireRenewableIssueMutex } from "./redis.js";
import { signVolunteerMarker, verifyVolunteerMarker } from "./protocol.js";
import { parseVolunteerMarker, volunteerAssigneesMatch, replaceVolunteerMarker } from "./markers.js";
import type { VolunteerLeaseMarker } from "./protocol.js";

export async function recoverExpiredOwnership(
  issue: Awaited<ReturnType<typeof fetchIssue>>,
  marker: VolunteerLeaseMarker,
  mutex: NonNullable<Awaited<ReturnType<typeof acquireRenewableIssueMutex>>>,
): Promise<{ issue: Awaited<ReturnType<typeof fetchIssue>>; marker: VolunteerLeaseMarker } | null> {
  let current = issue;
  let currentMarker: VolunteerLeaseMarker | null = marker;
  while (true) {
    const expired = currentMarker.leases.filter((lease) => Date.parse(lease.expires_at) <= Date.now());
    if (!expired.length) return { issue: current, marker: currentMarker };
    const expiredLease = expired[0];
    const contributor = expiredLease.contributor.toLowerCase();
    const redisLease = await getLeaseForIssue(current.number, expiredLease.language);
    if (redisLease && (redisLease.lease_id !== expiredLease.lease_id ||
        redisLease.contributor.toLowerCase() !== contributor ||
        !Number.isFinite(Date.parse(redisLease.expires_at)) ||
        Date.parse(redisLease.expires_at) > Date.now())) return null;
    if (!volunteerAssigneesMatch(current.assignees, currentMarker, Date.now(), true, true)) return null;
    const retained = currentMarker.leases.some(
      (lease) => lease.lease_id !== expiredLease.lease_id && lease.ownership_mode !== "public" &&
        lease.contributor.toLowerCase() === contributor,
    ) || (currentMarker.submissions || []).some(
      (submission) => ["submitted", "accepted"].includes(submission.status) &&
        submission.ownership_mode !== "public" &&
        submission.verified_contributor.toLowerCase() === contributor,
    );

    const assigned = (current.assignees || []).find(
      (assignee) => assignee.login.toLowerCase() === contributor,
    );
    if (expiredLease.ownership_mode !== "public" && assigned && !retained) {
      await mutex.assertOwned();
      try {
        await unassignIssue(current.number, assigned.login);
      } catch (error) {
        const observed = await fetchIssue(current.number).catch(() => null);
        if (!observed || (observed.assignees || []).some(
          (assignee) => assignee.login.toLowerCase() === contributor,
        )) throw error;
        current = observed;
      }
      current = await fetchIssue(current.number);
      currentMarker = parseVolunteerMarker(current.body);
      if (!currentMarker || !(await verifyVolunteerMarker(current.number, currentMarker))) return null;
      if (currentMarker.leases.some((lease) => lease.lease_id === expiredLease.lease_id) &&
          (current.assignees || []).some((assignee) => assignee.login.toLowerCase() === contributor)) return null;
      continue;
    }

    await mutex.assertOwned();
    const signed = await signVolunteerMarker(current.number, {
      ...currentMarker,
      submission: ["accepted", "rejected"].includes(currentMarker.submission) ? currentMarker.submission
        : currentMarker.submissions?.length ? "submitted" : "active",
      leases: currentMarker.leases.filter((lease) => lease.lease_id !== expiredLease.lease_id),
    });
    try {
      await patchIssue(current.number, replaceVolunteerMarker(current.body || "", signed));
    } catch (error) {
      const observed = await fetchIssue(current.number).catch(() => null);
      const observedMarker = observed && parseVolunteerMarker(observed.body);
      if (!observed || !observedMarker || observedMarker.leases.some(
        (lease) => lease.lease_id === expiredLease.lease_id,
      )) throw error;
      current = observed;
      currentMarker = observedMarker;
    }
    current = await fetchIssue(current.number);
    currentMarker = parseVolunteerMarker(current.body);
    if (!currentMarker || !(await verifyVolunteerMarker(current.number, currentMarker))) return null;
    if (currentMarker.leases.some((lease) => lease.lease_id === expiredLease.lease_id)) return null;
    const reconciled = await reconcilePublicAssignments(current, currentMarker, () => mutex.assertOwned());
    current = reconciled.issue;
    currentMarker = reconciled.marker;

    const redisAfterRecovery = await getLeaseForIssue(current.number, expiredLease.language);
    if (redisAfterRecovery && (redisAfterRecovery.lease_id !== expiredLease.lease_id ||
        !Number.isFinite(Date.parse(redisAfterRecovery.expires_at)) ||
        Date.parse(redisAfterRecovery.expires_at) > Date.now())) return null;
    if (redisAfterRecovery) {
      await mutex.assertOwned();
      if (!(await reclaimExpiredLease(redisAfterRecovery))) {
        const remaining = await getLeaseForIssue(current.number, expiredLease.language);
        if (remaining) return null;
      }
    }
  }
}


export async function recoverQueueOwnership(issueNumber: number) {
  const mutex = await acquireRenewableIssueMutex(issueNumber);
  if (!mutex) return null;
  try {
    const issue = await fetchIssue(issueNumber);
    const marker = parseVolunteerMarker(issue.body);
    if (issue.state !== "open" || !marker || !(await verifyVolunteerMarker(issueNumber, marker))) return null;
    const recovered = await recoverExpiredOwnership(issue, marker, mutex);
    if (!recovered) return null;
    return await reconcilePublicAssignments(recovered.issue, recovered.marker, () => mutex.assertOwned());
  } finally { await mutex.release(); }
}
