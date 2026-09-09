import {
  BrokerError,
  ConfigurationError,
  PROTOCOL_VERSION,
  acquireIssueMutex,
  fetchIssue,
  getPromotionReservation,
  json,
  method,
  promotionReservationKey,
  promotionSecret,
  randomToken,
  readJson,
  redisSet,
  savePromotionReservation,
  releaseIssueMutex,
  requireProtocol,
  parseVolunteerMarker,
  verifyVolunteerMarker,
} from "./_lib.ts";
import type { PromotionRecord, PromotionReservation } from "./_lib.ts";

export const config = { runtime: "edge" };

function records(value: unknown): PromotionRecord[] {
  if (!Array.isArray(value) || !value.length || value.length > 100_000) {
    throw new BrokerError(400, "records must be a non-empty array.");
  }
  const result = value.map((item) => {
    if (!item || typeof item !== "object") throw new BrokerError(400, "Invalid reservation record.");
    const record = item as Record<string, unknown>;
    if (typeof record.identity !== "string" || !record.identity ||
        typeof record.digest !== "string" || !/^[0-9a-f]{64}$/.test(record.digest)) {
      throw new BrokerError(400, "Reservation records require an identity and SHA256 digest.");
    }
    return { identity: record.identity, digest: record.digest };
  });
  const keys = result.map((item) => item.identity).sort();
  if (new Set(keys).size !== keys.length) throw new BrokerError(409, "Reservation identities are not unique.");
  return result.sort((a, b) => a.identity.localeCompare(b.identity));
}

function sameRecords(a: PromotionRecord[], b: PromotionRecord[]): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

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
    const requested = records(body.records);
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
        if (!sameRecords(existing.records, requested)) throw new BrokerError(409, "Reservation evidence differs.");
        await savePromotionReservation(existing, existing.status === "terminal");
        return json(200, { protocol_version: PROTOCOL_VERSION, status: existing.status, token: existing.token, expires_in: existing.status === "terminal" ? 30 * 24 * 60 * 60 : 15 * 60 });
      }
      const reservation: PromotionReservation = {
        issue_number: issueNumber, submission_id: submissionId, outcome, records: requested,
        token: randomToken(), status: "reserved",
      };
      if (!await redisSet(promotionReservationKey(issueNumber, submissionId), JSON.stringify(reservation), 15 * 60, true)) {
        throw new BrokerError(409, "A concurrent promotion reservation won; retry.");
      }
      return json(201, { protocol_version: PROTOCOL_VERSION, status: "reserved", token: reservation.token, expires_in: 15 * 60 });
    } finally { await releaseIssueMutex(issueNumber, mutex); }
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to reserve promotion." });
  }
}
