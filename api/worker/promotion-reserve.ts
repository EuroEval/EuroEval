import {
  bindPromotionDecision,
  BrokerError,
  ConfigurationError,
  PROTOCOL_VERSION,
  PROMOTION_RESERVATION_TTL,
  acquireIssueMutex,
  fetchIssue,
  getPromotionReservation,
  json,
  method,
  parsePromotionRecords,
  promotionReservationKey,
  promotionSecret,
  randomToken,
  readJson,
  redisSet,
  reservePromotionReservation,
  savePromotionReservation,
  releaseIssueMutex,
  requireProtocol,
  parseVolunteerMarker,
  verifyVolunteerMarker,
} from "./_lib.js";
import type { PromotionReservation } from "./_lib.js";
import { canonicalPublicationBucket } from "./_lib/publication.js";

export const config = { runtime: "edge" };

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    promotionSecret(req);
    const body = await readJson(req, 128 * 1024); requireProtocol(body);
    if (!Number.isSafeInteger(body.issue_number) || (body.issue_number as number) < 1 ||
        typeof body.submission_id !== "string" || !body.submission_id ||
        (body.outcome !== "accepted" && body.outcome !== "rejected")) {
      throw new BrokerError(400, "issue_number, submission_id, and outcome are required.");
    }
    const issueNumber = body.issue_number as number;
    const submissionId = body.submission_id as string;
    const outcome = body.outcome as "accepted" | "rejected";
    const requested = parsePromotionRecords(body.records);
    const requestedToken = body.reservation_token;
    const requestedDigest = body.decision_digest;
    if (requestedDigest !== undefined &&
        (typeof requestedDigest !== "string" || !/^[0-9a-f]{64}$/.test(requestedDigest))) {
      throw new BrokerError(400, "decision_digest must be a SHA256 digest.");
    }
    const requestedReviewer = body.reviewer;
    if (requestedReviewer !== undefined &&
        (typeof requestedReviewer !== "string" || !requestedReviewer.trim())) {
      throw new BrokerError(400, "reviewer must be a non-empty reviewer login.");
    }
    const durableReviewer = body.decision_reviewer;
    const durableCreatedAt = body.decision_created_at;
    if ((durableReviewer !== undefined &&
         (typeof durableReviewer !== "string" || !durableReviewer.trim())) ||
        (durableCreatedAt !== undefined &&
         (typeof durableCreatedAt !== "string" || !/^\d{4}-\d{2}-\d{2}T.*Z$/.test(durableCreatedAt)))) {
      throw new BrokerError(400, "Original decision metadata is invalid.");
    }
    if ((durableReviewer === undefined) !== (durableCreatedAt === undefined)) {
      throw new BrokerError(400, "Original decision metadata must be supplied together.");
    }
    if (body.decision_nonce !== undefined) {
      throw new BrokerError(400, "decision_nonce is no longer accepted; use reviewer metadata.");
    }
    const mutex = await acquireIssueMutex(issueNumber);
    if (!mutex) throw new BrokerError(409, "Issue is busy; retry promotion reservation.");
    try {
      const issue = await fetchIssue(issueNumber);
      const marker = parseVolunteerMarker(issue.body);
      if (!marker || !(await verifyVolunteerMarker(issueNumber, marker))) {
        throw new BrokerError(409, "The issue ownership marker is missing, unsigned, or malformed.");
      }
      const submission = marker.submissions?.find((item) => item.submission_id === submissionId);
      if (!submission) throw new BrokerError(404, "Submission is not present on this issue.");
      if (submission.status !== "submitted" && submission.status !== outcome) {
        throw new BrokerError(409, "Submission has already completed a different terminal transition.");
      }
      const existing = await getPromotionReservation(issueNumber, submissionId);
      const publicationBucket = body.publication_bucket;
      const publicationOwner = body.publication_owner;
      if (outcome === "accepted" && existing?.status !== "terminal" &&
          (publicationBucket !== canonicalPublicationBucket() || typeof publicationOwner !== "string" ||
           !/^[A-Za-z0-9_-]{24,256}$/.test(publicationOwner))) {
        throw new BrokerError(400, "Accepted promotions require publication_owner and publication_bucket.");
      }
      if (requestedToken === undefined && requestedReviewer === undefined &&
          !(requestedDigest !== undefined && existing !== undefined) &&
          !(requestedDigest !== undefined && durableReviewer !== undefined)) {
        throw new BrokerError(400, "reviewer or original decision metadata is required.");
      }
      if (existing) {
        if (existing.outcome !== outcome) throw new BrokerError(409, "A different outcome is already reserved.");
        if (JSON.stringify(existing.records) !== JSON.stringify(requested)) throw new BrokerError(409, "Reservation evidence differs.");
        if (requestedToken !== undefined && requestedToken !== existing.token) {
          throw new BrokerError(409, "Promotion reservation token differs.");
        }
        if (existing.decision_digest !== undefined &&
            requestedDigest !== existing.decision_digest) {
          throw new BrokerError(409, "Decision digest is required to match the reservation.");
        }
        if (existing.status === "terminal" && requestedDigest !== existing.decision_digest) {
          throw new BrokerError(409, "Terminal reservation decision digest differs.");
        }
        const stable: PromotionReservation = {
          ...existing,
          ...(outcome === "accepted" && existing.status !== "terminal" ? {
            publication_bucket: publicationBucket as string, publication_owner: publicationOwner as string,
          } : {}),
          decision_reviewer: existing.decision_reviewer ||
            (typeof requestedReviewer === "string" ? requestedReviewer.trim() : "legacy"),
          decision_created_at: existing.decision_created_at || new Date().toISOString(),
        };
        if (existing.status === "terminal") {
          return json(200, { protocol_version: PROTOCOL_VERSION, status: existing.status,
            token: stable.token, decision_reviewer: stable.decision_reviewer,
            decision_created_at: stable.decision_created_at,
            decision_digest: stable.decision_digest, expires_in: 30 * 24 * 60 * 60 });
        }
        const result = outcome === "accepted"
          ? await reservePromotionReservation(stable)
          : (await savePromotionReservation(stable, false), "reserved");
        if (result === "busy") throw new BrokerError(409, "A canonical result is being promoted; retry.");
        if (result === "conflict") throw new BrokerError(409, "A canonical result has a different digest.");
        if (requestedDigest !== undefined) {
          const bound = await bindPromotionDecision(stable, requestedDigest);
          if (bound === "mismatch") throw new BrokerError(409, "Decision digest is already bound differently.");
          if (bound !== "bound") throw new BrokerError(409, "Promotion reservation expired before decision binding.");
        }
        return json(200, { protocol_version: PROTOCOL_VERSION, status: "reserved", token: stable.token,
          decision_reviewer: stable.decision_reviewer, decision_created_at: stable.decision_created_at,
          decision_digest: requestedDigest || stable.decision_digest, expires_in: PROMOTION_RESERVATION_TTL });
      }
      if (requestedToken !== undefined) throw new BrokerError(409, "Promotion reservation is absent; retry.");
      if (requestedDigest !== undefined &&
          (durableReviewer === undefined || durableCreatedAt === undefined)) {
        throw new BrokerError(409, "Original decision metadata is required to recover an expired reservation.");
      }
      const reservation: PromotionReservation = {
        issue_number: issueNumber, submission_id: submissionId, outcome, records: requested,
        token: randomToken(),
        decision_reviewer: typeof durableReviewer === "string"
          ? durableReviewer.trim() : (requestedReviewer as string).trim(),
        decision_created_at: typeof durableCreatedAt === "string"
          ? durableCreatedAt : new Date().toISOString(),
        ...(requestedDigest !== undefined ? { decision_digest: requestedDigest } : {}),
        status: "reserved",
        ...(outcome === "accepted" ? {
          publication_bucket: publicationBucket as string, publication_owner: publicationOwner as string,
        } : {}),
      };
      if (outcome === "accepted") {
        const result = await reservePromotionReservation(reservation);
        if (result === "busy") throw new BrokerError(409, "A canonical result is being promoted; retry.");
        if (result === "conflict") throw new BrokerError(409, "A canonical result has a different digest.");
      } else if (!await redisSet(promotionReservationKey(issueNumber, submissionId), JSON.stringify(reservation), PROMOTION_RESERVATION_TTL, true)) {
        throw new BrokerError(409, "A concurrent promotion reservation won; retry.");
      }
      return json(201, { protocol_version: PROTOCOL_VERSION, status: "reserved", token: reservation.token,
        decision_reviewer: reservation.decision_reviewer, decision_created_at: reservation.decision_created_at,
        ...(reservation.decision_digest ? { decision_digest: reservation.decision_digest } : {}),
        expires_in: PROMOTION_RESERVATION_TTL });
    } finally { await releaseIssueMutex(issueNumber, mutex); }
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to reserve promotion." });
  }
}
