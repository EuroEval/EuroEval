import {
  BrokerError,
  ConfigurationError,
  PROTOCOL_VERSION,
  authenticate,
  brokerErrorBody,
  canaryIdentity,
  commitCanaryReceipt,
  enforceRateLimit,
  findCanaryLease,
  getCanaryReceipt,
  json,
  method,
  readJson,
  releaseCanary,
  requireCanaryReservation,
  requireProtocol,
  sha256,
  storedCanaryPaths,
  validateCanaryEvidence,
  validateCanaryObservationBinding,
} from "./_lib.js";
import type { CanaryReceipt } from "./_lib.js";
import { uploadCanaryEvidence } from "./_lib/canary-storage.js";
const MAX_BODY = 512 * 1024;

export async function fetch(req: Request): Promise<Response> {
  const rejected = method(req);
  if (rejected) return rejected;
  try {
    const worker = await authenticate(req);
    await enforceRateLimit(`euroeval:worker:limit:canary-evidence:${worker.hash}`, 20, 3600);
    const body = await readJson(req, MAX_BODY);
    requireProtocol(body);
    for (const field of [
      "lease_id", "reservation_id", "model_id", "model_revision", "corpus_revision",
      "corpus_sha256", "evidence_json", "digest",
    ]) if (typeof body[field] !== "string") throw new BrokerError(400, `${field} is required.`);
    const lease = await findCanaryLease(body.lease_id as string);
    if (!lease || lease.contributor.toLowerCase() !== worker.contributor.toLowerCase()) {
      throw new BrokerError(409, "Lease is absent or belongs to another contributor.");
    }
    const instruction = lease.contamination_canary;
    if (
      !instruction || instruction.status !== "required" ||
      instruction.reservation_id !== body.reservation_id || body.model_id !== lease.model_id ||
      body.model_revision !== lease.model_revision ||
      body.corpus_revision !== instruction.corpus_revision ||
      body.corpus_sha256 !== instruction.corpus_sha256
    ) throw new BrokerError(409, "Evidence does not match the canary lease.");
    const digest = await sha256(body.evidence_json as string);
    if (digest !== body.digest) throw new BrokerError(422, "Canary evidence digest is invalid.");
    const identity = await canaryIdentity(lease.model_id, lease.model_revision);
    const existing = await getCanaryReceipt(identity);
    if (existing) {
      return existing.digest === digest
        ? json(200, { protocol_version: PROTOCOL_VERSION, status: "duplicate", receipt: digest })
        : json(409, { error: "Different canary evidence already exists; manual review is required." });
    }
    let parsed: unknown;
    try { parsed = JSON.parse(body.evidence_json as string); }
    catch { throw new BrokerError(422, "Canary evidence must be valid JSON."); }
    validateCanaryEvidence(parsed, { modelId: lease.model_id, revision: lease.model_revision });
    await validateCanaryObservationBinding(parsed);
    if (
      instruction.reason === "encoder" &&
      ((parsed as Record<string, unknown>).status !== "not_applicable" ||
        (parsed as Record<string, unknown>).reason !== "encoder")
    ) throw new BrokerError(422, "Encoder canary evidence must be not_applicable/encoder.");
    if ((parsed as Record<string, unknown>).status === "failed") {
      await releaseCanary(lease);
      return json(202, {
        protocol_version: PROTOCOL_VERSION,
        status: "retryable_failure_not_stored",
        receipt: digest,
      });
    }
    const path = `v1/${identity}/${digest}.json`;
    const stored = await storedCanaryPaths(identity);
    if (stored.includes(path)) {
      return json(200, {
        protocol_version: PROTOCOL_VERSION,
        status: "duplicate",
        receipt: digest,
      });
    }
    if (stored.length) {
      throw new BrokerError(409, "Different canary evidence already exists; manual review is required.");
    }
    const { reservation } = await requireCanaryReservation(
      lease.model_id, lease.model_revision, lease.lease_id, body.reservation_id as string,
    );
    await uploadCanaryEvidence(identity, digest, `${body.evidence_json as string}\n`);
    const receipt: CanaryReceipt = {
      identity,
      lease_id: lease.lease_id,
      reservation_id: body.reservation_id as string,
      digest,
      accepted_at: new Date().toISOString(),
      path,
    };
    const outcome = await commitCanaryReceipt(identity, reservation, receipt);
    if (outcome === "conflict") {
      return json(409, { error: "Different canary evidence already exists; manual review is required." });
    }
    if (outcome === "lost") throw new BrokerError(409, "Canary reservation expired before acknowledgement.");
    return json(outcome === "accepted" ? 201 : 200, {
      protocol_version: PROTOCOL_VERSION,
      status: outcome,
      receipt: digest,
    });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, brokerErrorBody(error, "Unable to accept canary evidence."));
  }
}

export default fetch;
