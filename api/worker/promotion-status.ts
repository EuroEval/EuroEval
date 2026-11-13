import { promotionSecret } from "./_lib/auth.js";
import { BrokerError, json, method, PROTOCOL_VERSION, readJson, requireProtocol } from "./_lib/protocol.js";
import { getPromotionReservation, parsePromotionRecords } from "./_lib/promotion.js";

export const config = { runtime: "edge" };

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req);
  if (rejected) return rejected;
  try {
    promotionSecret(req);
    const body = await readJson(req, 128 * 1024);
    requireProtocol(body);
    if (!Number.isSafeInteger(body.issue_number) || (body.issue_number as number) < 1 ||
        typeof body.submission_id !== "string" || !body.submission_id ||
        (body.outcome !== "accepted" && body.outcome !== "rejected") ||
        typeof body.decision_digest !== "string" || !/^[0-9a-f]{64}$/.test(body.decision_digest)) {
      throw new BrokerError(400, "issue_number, submission_id, outcome, and decision_digest are required.");
    }
    const issueNumber = body.issue_number as number;
    const submissionId = body.submission_id as string;
    const outcome = body.outcome as "accepted" | "rejected";
    const decisionDigest = body.decision_digest;
    const records = parsePromotionRecords(body.records);
    const reservation = await getPromotionReservation(issueNumber, submissionId);
    if (!reservation) {
      return json(200, { protocol_version: PROTOCOL_VERSION, status: "absent" });
    }
    let reservedRecords: ReturnType<typeof parsePromotionRecords>;
    try {
      reservedRecords = parsePromotionRecords(reservation.records);
    } catch {
      throw new BrokerError(409, "Promotion reservation evidence differs.");
    }
    if (reservation.issue_number !== issueNumber || reservation.submission_id !== submissionId ||
        reservation.outcome !== outcome || reservation.decision_digest !== decisionDigest ||
        JSON.stringify(reservedRecords) !== JSON.stringify(records) ||
        (reservation.status !== "reserved" && reservation.status !== "terminal")) {
      throw new BrokerError(409, "Promotion reservation evidence differs.");
    }
    return json(200, {
      protocol_version: PROTOCOL_VERSION,
      status: reservation.status,
      decision_digest: reservation.decision_digest,
    });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to read promotion status." });
  }
}
