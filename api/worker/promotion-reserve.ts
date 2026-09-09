import {
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
} from "./_lib.ts";
import type { PromotionReservation } from "./_lib.ts";

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
    const requestedDecisionNonce = body.decision_nonce;
    if (requestedDecisionNonce !== undefined && (typeof requestedDecisionNonce !== "string" || !requestedDecisionNonce)) {
      throw new BrokerError(400, "decision_nonce must be a non-empty broker-issued value.");
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
      if (existing) {
        if (existing.outcome !== outcome) throw new BrokerError(409, "A different outcome is already reserved.");
        if (JSON.stringify(existing.records) !== JSON.stringify(requested)) throw new BrokerError(409, "Reservation evidence differs.");
        if (requestedToken !== undefined && requestedToken !== existing.token) {
          throw new BrokerError(409, "Promotion reservation token differs.");
        }
        const existingDecisionNonce = existing.decision_nonce || existing.token;
        if (requestedDecisionNonce !== undefined && requestedDecisionNonce !== existingDecisionNonce) {
          throw new BrokerError(409, "Promotion decision metadata differs.");
        }
        // Older reservations predate decision_nonce. Their broker-issued token
        // is already stable, so use it as a compatibility nonce and persist the
        // normalized shape before returning it to the caller.
        const stable = existing.decision_nonce ? existing : { ...existing, decision_nonce: existing.token };
        if (existing.status === "terminal") {
          return json(200, { protocol_version: PROTOCOL_VERSION, status: existing.status, token: stable.token,
            decision_nonce: stable.decision_nonce, expires_in: 30 * 24 * 60 * 60 });
        }
        const result = outcome === "accepted"
          ? await reservePromotionReservation(stable)
          : (await savePromotionReservation(stable, false), "reserved");
        if (result === "busy") throw new BrokerError(409, "A canonical result is being promoted; retry.");
        if (result === "conflict") throw new BrokerError(409, "A canonical result has a different digest.");
        return json(200, { protocol_version: PROTOCOL_VERSION, status: "reserved", token: stable.token,
          decision_nonce: stable.decision_nonce, expires_in: PROMOTION_RESERVATION_TTL });
      }
      if (requestedToken !== undefined || requestedDecisionNonce !== undefined) throw new BrokerError(409, "Promotion reservation is absent; retry.");
      const reservation: PromotionReservation = {
        issue_number: issueNumber, submission_id: submissionId, outcome, records: requested,
        token: randomToken(), decision_nonce: randomToken(), status: "reserved",
      };
      if (outcome === "accepted") {
        const result = await reservePromotionReservation(reservation);
        if (result === "busy") throw new BrokerError(409, "A canonical result is being promoted; retry.");
        if (result === "conflict") throw new BrokerError(409, "A canonical result has a different digest.");
      } else if (!await redisSet(promotionReservationKey(issueNumber, submissionId), JSON.stringify(reservation), PROMOTION_RESERVATION_TTL, true)) {
        throw new BrokerError(409, "A concurrent promotion reservation won; retry.");
      }
      return json(201, { protocol_version: PROTOCOL_VERSION, status: "reserved", token: reservation.token,
        decision_nonce: reservation.decision_nonce, expires_in: PROMOTION_RESERVATION_TTL });
    } finally { await releaseIssueMutex(issueNumber, mutex); }
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to reserve promotion." });
  }
}
