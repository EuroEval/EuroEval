declare const process: { env: Record<string, string | undefined> };

import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, acquireIssueMutex,
  addIssueLabel, commentIssue, env, fetchIssue, issueComments, json, method,
  parseVolunteerMarker, patchIssue, promotionSecret, readJson, releaseIssueMutex,
  removeIssueLabel, replaceVolunteerMarker, requireProtocol, selectedLanguages,
  unassignIssue, signVolunteerMarker, verifyVolunteerMarker, signFinalCredit,
  redis, releaseResultReservation, sha256,
} from "./_lib.ts";
import type { VolunteerLeaseMarker, VolunteerSubmission } from "./_lib.ts";

export const config = { runtime: "edge" };
const PROMOTION_MARKER = "euroeval-volunteer-promotion:v1";
const CREDIT_MARKER = "euroeval-volunteer-credit:v1";

export interface PromotionPlan {
  marker: VolunteerLeaseMarker;
  complete: boolean;
  winner: string | null;
  acceptedCounts: Array<{ login: string; count: number }>;
  removeReviewLabel: boolean;
  releaseCoordinator: boolean;
}

/** Derive the monotonic issue lifecycle from one reviewed submission. */
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
  const complete = selected.length > 0 && selected.every((language) => acceptedLanguages.has(language));
  const active = marker.leases.some((lease) => Date.parse(lease.expires_at) > now);
  const submitted = submissions.some((item) => item.status === "submitted");
  const state = complete ? "accepted" : submitted ? "submitted" : active ? "active" :
    outcome === "rejected" ? "rejected" : "active";
  const next: VolunteerLeaseMarker = {
    ...marker,
    submission: state,
    submissions,
    completed_languages: [...acceptedLanguages].sort(),
  };
  const counts = new Map<string, number>();
  for (const item of submissions.filter((item) => item.status === "accepted")) {
    counts.set(item.verified_contributor, (counts.get(item.verified_contributor) || 0) + item.result_count);
  }
  const acceptedCounts = [...counts.entries()].map(([login, count]) => ({ login, count }))
    .sort((a, b) => b.count - a.count || a.login.toLowerCase().localeCompare(b.login.toLowerCase()) || a.login.localeCompare(b.login));
  return {
    marker: next,
    complete,
    winner: complete ? largestAcceptedShare(submissions) : null,
    acceptedCounts,
    removeReviewLabel: !submitted,
    releaseCoordinator: complete || outcome === "rejected" && !active && !submitted,
  };
}

/** Select the contributor with most server-counted accepted results. */
export function largestAcceptedShare(submissions: VolunteerSubmission[]): string | null {
  const shares = new Map<string, { login: string; count: number }>();
  for (const item of submissions) {
    if (item.status !== "accepted") continue;
    const key = item.verified_contributor.toLowerCase();
    const previous = shares.get(key);
    shares.set(key, {
      login: previous?.login || item.verified_contributor,
      count: (previous?.count || 0) + item.result_count,
    });
  }
  return [...shares.values()].sort((left, right) =>
    right.count - left.count ||
    left.login.toLowerCase().localeCompare(right.login.toLowerCase()) ||
    left.login.localeCompare(right.login))[0]?.login || null;
}

/** Authenticate and apply a resumable accepted/rejected transition. */
async function releaseSubmissionReservations(submissionId: string): Promise<void> {
  const raw = await redis("SMEMBERS", `euroeval:worker:reservations:${submissionId}`);
  if (!Array.isArray(raw)) return;
  for (const encoded of raw) {
    if (typeof encoded !== "string") continue;
    try {
      const item = JSON.parse(encoded) as { identity?: string; digest?: string };
      if (typeof item.identity === "string" && typeof item.digest === "string") {
        await releaseResultReservation(`euroeval:worker:record-identity:${await sha256(item.identity)}`, item.digest, submissionId);
      }
    } catch { /* An audit entry must not abort the rest of the reclamation. */ }
  }
  await redis("DEL", `euroeval:worker:reservations:${submissionId}`);
}

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    promotionSecret(req);
    const body = await readJson(req, 16 * 1024); requireProtocol(body);
    if (!Number.isSafeInteger(body.issue_number) || typeof body.submission_id !== "string" ||
        !body.submission_id || body.outcome !== "accepted" && body.outcome !== "rejected") {
      throw new BrokerError(400, "issue_number, submission_id, and an accepted/rejected outcome are required.");
    }
    const issueNumber = body.issue_number as number;
    const submissionId = body.submission_id as string;
    const outcome = body.outcome as "accepted" | "rejected";
    const mutex = await acquireIssueMutex(issueNumber);
    if (!mutex) throw new BrokerError(409, "Issue is busy; retry promotion.");
    try {
      const issue = await fetchIssue(issueNumber);
      const marker = parseVolunteerMarker(issue.body);
      if (!marker || !(await verifyVolunteerMarker(issueNumber, marker))) throw new BrokerError(409, "The issue ownership marker is missing, unsigned, or malformed.");
      const plan = promotionPlan(marker, selectedLanguages(issue.body), submissionId, outcome);
      const signedMarker = await signVolunteerMarker(issueNumber, plan.marker);
      let promotedBody = replaceVolunteerMarker(issue.body || "", signedMarker);
      if (plan.complete && plan.winner) {
        const credit = await signFinalCredit(issueNumber, { version: 1, immutable: true, winner: plan.winner, accepted_counts: plan.acceptedCounts, completed_languages: plan.marker.completed_languages || [] });
        promotedBody += `<!-- ${CREDIT_MARKER} ${JSON.stringify(credit)} -->\n`;
      }
      await patchIssue(issueNumber, promotedBody);
      const fencedIssue = await fetchIssue(issueNumber);
      const fenced = parseVolunteerMarker(fencedIssue.body);
      if (!fenced || !(await verifyVolunteerMarker(issueNumber, fenced)) || fenced.submissions?.find((item) => item.submission_id === submissionId)?.status !== outcome ||
          plan.complete && !fencedIssue.body?.includes(CREDIT_MARKER)) {
        throw new BrokerError(409, "Promotion fence lost.");
      }

      if (outcome === "rejected") await releaseSubmissionReservations(submissionId);
      const reviewLabel = process.env.COMMUNITY_REVIEW_LABEL || "community-review-ready";
      const resultsLabel = process.env.RESULTS_READY_LABEL || "results-ready";
      if (plan.complete && !fencedIssue.labels?.some((item) => item.name === resultsLabel)) {
        await addIssueLabel(issueNumber, resultsLabel);
      }
      if (plan.removeReviewLabel) await removeIssueLabel(issueNumber, reviewLabel);
      if (plan.releaseCoordinator) await unassignIssue(issueNumber, env("WORKER_COORDINATOR_LOGIN"));

      const comments = await issueComments(issueNumber);
      if (!comments.some((item) => item.body?.includes(`${PROMOTION_MARKER} ${submissionId}`))) {
        const current = plan.marker.submissions?.find((item) => item.submission_id === submissionId);
        await commentIssue(issueNumber, `<!-- ${PROMOTION_MARKER} ${submissionId} -->\nCommunity submission **${submissionId}** was **${outcome}**.\n\nManifest: \`${current?.manifest_path}\``);
      }
      return json(200, {
        protocol_version: PROTOCOL_VERSION,
        status: outcome,
        submission_id: submissionId,
        complete: plan.complete,
        winner: plan.winner,
      });
    } finally { await releaseIssueMutex(issueNumber, mutex); }
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to promote submission." });
  }
}
