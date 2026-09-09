declare const process: { env: Record<string, string | undefined> };

import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, addIssueLabel, acquireIssueMutex,
  commentIssue, completePromotionReservation, env, fetchIssue, getPromotionReservation,
  issueComments, json, method, parseFinalCredit, parsePromotionRecords,
  parseVolunteerMarker, patchIssue, promotionSecret, readJson, redis, releaseIssueMutex,
  releaseResultReservation,
  removeIssueLabel, replaceVolunteerMarker, requireProtocol,
  savePromotionReservation, selectedLanguages, signFinalCredit, signVolunteerMarker, unassignIssue,
  verifyFinalCredit, verifyVolunteerMarker, sha256,
} from "./_lib.ts";
import type {
  FinalCredit, PromotionRecord, PromotionReservation, VolunteerLeaseMarker, VolunteerSubmission,
} from "./_lib.ts";

export const config = { runtime: "edge" };
export const PROMOTION_MARKER = "euroeval-volunteer-promotion:v1";

export interface PromotionPlan {
  marker: VolunteerLeaseMarker;
  complete: boolean;
  winner: string | null;
  acceptedCounts: Array<{ login: string; count: number }>;
  removeReviewLabel: boolean;
  releaseCoordinator: boolean;
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
  const counts = new Map<string, number>();
  for (const item of submissions.filter((item) => item.status === "accepted")) {
    counts.set(item.verified_contributor, (counts.get(item.verified_contributor) || 0) + item.result_count);
  }
  const acceptedCounts = [...counts.entries()].map(([login, count]) => ({ login, count }))
    .sort((a, b) => b.count - a.count || a.login.toLowerCase().localeCompare(b.login.toLowerCase()) || a.login.localeCompare(b.login));
  return {
    marker: next, complete, winner: complete ? largestAcceptedShare(submissions) : null,
    acceptedCounts, removeReviewLabel: !submitted,
    releaseCoordinator: complete || outcome === "rejected" && !active && !submitted,
  };
}

export function largestAcceptedShare(submissions: VolunteerSubmission[] | undefined): string | null {
  const shares = new Map<string, { login: string; count: number }>();
  for (const item of submissions || []) {
    if (item.status !== "accepted") continue;
    const key = item.verified_contributor.toLowerCase();
    const previous = shares.get(key);
    shares.set(key, { login: previous?.login || item.verified_contributor, count: (previous?.count || 0) + item.result_count });
  }
  return [...shares.values()].sort((left, right) => right.count - left.count || left.login.toLowerCase().localeCompare(right.login.toLowerCase()) || left.login.localeCompare(right.login))[0]?.login || null;
}

function sameRecords(a: PromotionRecord[], b: PromotionRecord[]): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

async function releaseSubmissionReservations(reservation: PromotionReservation): Promise<void> {
  const raw = await redis("SMEMBERS", `euroeval:worker:reservations:${reservation.submission_id}`);
  const records = reservation.records;
  for (const record of records) {
    await releaseResultReservation(
      `euroeval:worker:record-identity:${await sha256(record.identity)}`,
      record.digest,
      reservation.submission_id,
    );
  }
  if (Array.isArray(raw)) await redis("DEL", `euroeval:worker:reservations:${reservation.submission_id}`);
}

function validCredit(credit: FinalCredit | null, plan: PromotionPlan): boolean {
  return !!credit && credit.winner === plan.winner &&
    JSON.stringify(credit.accepted_counts) === JSON.stringify(plan.acceptedCounts) &&
    JSON.stringify(credit.completed_languages) === JSON.stringify(plan.marker.completed_languages || []);
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
    const requestedRecords = parsePromotionRecords(body.records);
    const reservation = await getPromotionReservation(issueNumber, submissionId);
    if (!reservation || reservation.token !== body.reservation_token || reservation.outcome !== outcome ||
        !sameRecords(reservation.records, requestedRecords)) {
      throw new BrokerError(409, "Promotion reservation is absent, expired, or does not match.");
    }
    const mutex = await acquireIssueMutex(issueNumber);
    if (!mutex) throw new BrokerError(409, "Issue is busy; retry promotion.");
    try {
      const issue = await fetchIssue(issueNumber);
      const marker = parseVolunteerMarker(issue.body);
      if (!marker || !(await verifyVolunteerMarker(issueNumber, marker))) throw new BrokerError(409, "The issue ownership marker is missing, unsigned, or malformed.");
      const plan = promotionPlan(marker, selectedLanguages(issue.body), submissionId, outcome);
      let promotedBody = issue.body || "";
      const oldCredit = parseFinalCredit(promotedBody);
      let credit: FinalCredit | null = null;
      if (plan.complete && plan.winner) {
        credit = oldCredit && validCredit(oldCredit, plan) && await verifyFinalCredit(issueNumber, oldCredit)
          ? oldCredit
          : await signFinalCredit(issueNumber, { version: 1, immutable: true, winner: plan.winner, accepted_counts: plan.acceptedCounts, completed_languages: plan.marker.completed_languages || [] });
      }
      const signedMarker = await signVolunteerMarker(issueNumber, plan.marker);
      promotedBody = replaceVolunteerMarker(promotedBody, signedMarker);
      if (credit) promotedBody += `<!-- euroeval-volunteer-credit:v1 ${JSON.stringify(credit)} -->\n`;
      await patchIssue(issueNumber, promotedBody);
      const fencedIssue = await fetchIssue(issueNumber);
      const fenced = parseVolunteerMarker(fencedIssue.body);
      if (!fenced || !(await verifyVolunteerMarker(issueNumber, fenced)) ||
          fenced.submissions?.find((item) => item.submission_id === submissionId)?.status !== outcome ||
          plan.complete && !validCredit(parseFinalCredit(fencedIssue.body), plan)) {
        throw new BrokerError(409, "Promotion fence lost.");
      }
      if (outcome === "rejected") await releaseSubmissionReservations(reservation);
      const reviewLabel = process.env.COMMUNITY_REVIEW_LABEL || "community-review-ready";
      const resultsLabel = process.env.RESULTS_READY_LABEL || "results-ready";
      if (plan.complete && !fencedIssue.labels?.some((item) => item.name === resultsLabel)) await addIssueLabel(issueNumber, resultsLabel);
      if (plan.removeReviewLabel) await removeIssueLabel(issueNumber, reviewLabel);
      if (plan.releaseCoordinator) await unassignIssue(issueNumber, env("WORKER_COORDINATOR_LOGIN"));
      const comments = await issueComments(issueNumber);
      if (!comments.some((item) => item.body?.includes(`${PROMOTION_MARKER} ${submissionId}`))) {
        const current = plan.marker.submissions?.find((item) => item.submission_id === submissionId);
        await commentIssue(issueNumber, `<!-- ${PROMOTION_MARKER} ${submissionId} -->\nCommunity submission **${submissionId}** was **${outcome}**.\n\nManifest: \`${current?.manifest_path}\``);
      }
      await terminalReservation(reservation);
      return json(200, { protocol_version: PROTOCOL_VERSION, status: outcome, submission_id: submissionId, complete: plan.complete, winner: plan.winner });
    } finally { await releaseIssueMutex(issueNumber, mutex); }
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to promote submission." });
  }
}
