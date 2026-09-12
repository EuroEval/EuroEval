declare const process: { env: Record<string, string | undefined> };

import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, addIssueLabel, acquireRenewableIssueMutex,
  brokerErrorBody, commentIssue, completePromotionReservation, fetchIssue, getPromotionReservation,
  issueComments, json, method, parsePromotionRecords,
  parseVolunteerMarker, patchIssue, promotionSecret, readJson,
  releaseResultReservations,
  removeIssueLabel, replaceVolunteerMarker, requireProtocol,
  savePromotionReservation, selectedLanguages, signVolunteerMarker, unassignIssue,
  verifyVolunteerMarker,
} from "./_lib.ts";
import type {
  PromotionRecord, PromotionReservation, VolunteerLeaseMarker,
} from "./_lib.ts";

export const config = { runtime: "edge" };
export const PROMOTION_MARKER = "euroeval-volunteer-promotion:v1";

export interface PromotionPlan {
  marker: VolunteerLeaseMarker;
  complete: boolean;
  removeReviewLabel: boolean;
  releaseContributor: string | null;
}

export function promotionPlan(
  marker: VolunteerLeaseMarker,
  selected: string[],
  submissionId: string,
  outcome: "accepted" | "rejected",
  now = Date.now(),
): PromotionPlan {
  const current = marker.submissions?.find((item) => item.submission_id === submissionId);
  if (!current) throw new BrokerError(404, "Submission is not present on this issue.");
  if (current.status !== "submitted" && current.status !== outcome) {
    throw new BrokerError(409, "Submission has already completed a different terminal transition.");
  }
  const submissions = (marker.submissions || []).map((item) =>
    item.submission_id === submissionId ? { ...item, status: outcome } : item);
  const acceptedLanguages = new Set(submissions
    .filter((item) => item.status === "accepted").map((item) => item.language));
  const complete = selected.length > 0 && selected.length === acceptedLanguages.size &&
    selected.every((language) => acceptedLanguages.has(language));
  const active = marker.leases.some((lease) => Date.parse(lease.expires_at) > now);
  const submitted = submissions.some((item) => item.status === "submitted");
  const state = complete ? "accepted" : submitted ? "submitted" : active ? "active" :
    outcome === "rejected" ? "rejected" : "active";
  const next: typeof marker = {
    ...marker, submission: state as typeof marker.submission, submissions,
    completed_languages: [...acceptedLanguages].sort(),
  };
  const target = current.verified_contributor.toLowerCase();
  const contributorRetained = marker.leases.some((lease) =>
    Date.parse(lease.expires_at) > now && lease.contributor.toLowerCase() === target) ||
    submissions.some((item) => ["submitted", "accepted"].includes(item.status) &&
      item.verified_contributor.toLowerCase() === target);
  const releaseContributor = outcome === "rejected" && !contributorRetained ?
    current.verified_contributor : null;
  return {
    marker: next, complete, removeReviewLabel: !submitted,
    releaseContributor,
  };
}

function sameRecords(a: PromotionRecord[], b: PromotionRecord[]): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

async function releaseSubmissionReservations(reservation: PromotionReservation): Promise<void> {
  if (!await releaseResultReservations(reservation.records, reservation.submission_id)) {
    throw new BrokerError(409, "A result reservation could not be safely released.");
  }
}

async function terminalReservation(reservation: PromotionReservation): Promise<void> {
  if (reservation.outcome === "accepted") {
    if (!(await completePromotionReservation(reservation))) {
      throw new BrokerError(409, "Promotion reservation was replaced during promotion.");
    }
    return;
  }
  await savePromotionReservation(reservation, true);
}

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    promotionSecret(req);
    const body = await readJson(req, 128 * 1024); requireProtocol(body);
    if (!Number.isSafeInteger(body.issue_number) || typeof body.submission_id !== "string" ||
        !body.submission_id || (body.outcome !== "accepted" && body.outcome !== "rejected") ||
        typeof body.reservation_token !== "string" || !body.reservation_token || !Array.isArray(body.records)) {
      throw new BrokerError(400, "issue_number, submission_id, outcome, reservation_token, and records are required.");
    }
    const issueNumber = body.issue_number as number;
    const submissionId = body.submission_id as string;
    const outcome = body.outcome as "accepted" | "rejected";
    if (typeof body.decision_digest !== "string" || !/^[0-9a-f]{64}$/.test(body.decision_digest)) {
      throw new BrokerError(400, "decision_digest must be a SHA256 digest.");
    }
    const decisionDigest = body.decision_digest;
    const requestedRecords = parsePromotionRecords(body.records);
    const reservation = await getPromotionReservation(issueNumber, submissionId);
    if (!reservation || reservation.token !== body.reservation_token || reservation.outcome !== outcome ||
        reservation.decision_digest !== decisionDigest || !sameRecords(reservation.records, requestedRecords)) {
      throw new BrokerError(409, "Promotion reservation is absent, expired, or does not match.");
    }
    const mutex = await acquireRenewableIssueMutex(issueNumber);
    if (!mutex) throw new BrokerError(409, "Issue is busy; retry promotion.");
    try {
      const issue = await fetchIssue(issueNumber);
      const marker = parseVolunteerMarker(issue.body);
      if (!marker || !(await verifyVolunteerMarker(issueNumber, marker))) throw new BrokerError(409, "The issue ownership marker is missing, unsigned, or malformed.");
      const plan = promotionPlan(marker, selectedLanguages(issue.body), submissionId, outcome);
      // replaceVolunteerMarker also strips any legacy immutable credit marker.
      // Assignees, not this marker, are the mutable credit source.
      let promotedBody = issue.body || "";
      // Rejection must not make the language claimable while any identity is
      // still reserved. Keep the signed marker submitted if cleanup fails.
      if (outcome === "rejected") await releaseSubmissionReservations(reservation);
      const signedMarker = await signVolunteerMarker(issueNumber, plan.marker);
      promotedBody = replaceVolunteerMarker(promotedBody, signedMarker);
      await mutex.assertOwned();
      await patchIssue(issueNumber, promotedBody);
      const fencedIssue = await fetchIssue(issueNumber);
      const fenced = parseVolunteerMarker(fencedIssue.body);
      if (!fenced || !(await verifyVolunteerMarker(issueNumber, fenced)) ||
          fenced.submissions?.find((item) => item.submission_id === submissionId)?.status !== outcome) {
        throw new BrokerError(409, "Promotion fence lost.");
      }
      const reviewLabel = process.env.COMMUNITY_REVIEW_LABEL || "community-review-ready";
      const resultsLabel = process.env.RESULTS_READY_LABEL || "results-ready";
      if (plan.complete && !fencedIssue.labels?.some((item) => item.name === resultsLabel)) {
        await mutex.assertOwned();
        await addIssueLabel(issueNumber, resultsLabel);
      }
      if (plan.removeReviewLabel) {
        await mutex.assertOwned();
        await removeIssueLabel(issueNumber, reviewLabel);
      }
      if (plan.releaseContributor) {
        const currentIssue = await fetchIssue(issueNumber);
        const contributor = plan.releaseContributor.toLowerCase();
        const assigned = (currentIssue.assignees || []).find((item) => item.login.toLowerCase() === contributor);
        if (assigned) {
          await mutex.assertOwned();
          await unassignIssue(issueNumber, assigned.login);
          const afterAssignment = await fetchIssue(issueNumber);
          if ((afterAssignment.assignees || []).some((item) => item.login.toLowerCase() === contributor)) {
            throw new BrokerError(409, "Promotion assignment fence lost.");
          }
        }
      }
      const comments = await issueComments(issueNumber);
      if (!comments.some((item) => item.body?.includes(`${PROMOTION_MARKER} ${submissionId}`))) {
        const current = plan.marker.submissions?.find((item) => item.submission_id === submissionId);
        await mutex.assertOwned();
        await commentIssue(issueNumber, `<!-- ${PROMOTION_MARKER} ${submissionId} -->\nCommunity submission **${submissionId}** was **${outcome}**.\n\nManifest: \`${current?.manifest_path}\``);
      }
      await terminalReservation(reservation);
      return json(200, { protocol_version: PROTOCOL_VERSION, status: outcome, submission_id: submissionId,
        decision_reviewer: reservation.decision_reviewer,
        decision_created_at: reservation.decision_created_at,
        complete: plan.complete, decision_digest: decisionDigest });
    } finally { await mutex.release(); }
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, brokerErrorBody(error, "Unable to promote submission."));
  }
}
