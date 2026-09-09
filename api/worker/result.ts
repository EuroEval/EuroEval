import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, authenticate, enforceRateLimit,
  getLeaseById, json, method, readJson, sha256, redis, redisGet,
  reserveResultIdentity, completeResultIdentity, abortResultIdentity,
  uploadStaging, validateRecord, requireProtocol,
} from "./_lib";

export const config = { runtime: "edge" };
const MAX_RESULT_BODY = 2 * 1024 * 1024;
type StoredIdentity = { digest: string; lease_id: string; status: "uploading" | "uploaded"; issue_number: number; language: string; path: string; warnings?: string[] };

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const identity = await authenticate(req); await enforceRateLimit(`euroeval:worker:limit:result:${identity.hash}`, 500, 3600);
    const body = await readJson(req, MAX_RESULT_BODY); requireProtocol(body);
    const required = ["lease_id", "issue_number", "language", "model_id", "model_revision", "euroeval_version", "image_digest", "worker_version", "record_json", "digest"];
    for (const field of required) if (body[field] === undefined) throw new BrokerError(400, `${field} is required.`);
    if (typeof body.lease_id !== "string" || typeof body.language !== "string" || typeof body.model_id !== "string" || typeof body.model_revision !== "string" || typeof body.euroeval_version !== "string" || typeof body.image_digest !== "string" || typeof body.worker_version !== "string" || typeof body.record_json !== "string" || typeof body.digest !== "string" || typeof body.issue_number !== "number" || !Number.isSafeInteger(body.issue_number)) throw new BrokerError(400, "Result contract fields have invalid types.");
    const lease = await getLeaseById(body.lease_id);
    if (!lease || lease.contributor.toLowerCase() !== identity.contributor.toLowerCase() || Date.parse(lease.expires_at) <= Date.now()) throw new BrokerError(409, "Lease is absent, expired, or belongs to another contributor.");
    if (body.issue_number !== lease.issue_number || body.language !== lease.language || body.model_id !== lease.model_id || body.model_revision !== lease.model_revision || body.euroeval_version !== lease.euroeval_version || body.image_digest !== lease.image_digest || body.worker_version !== lease.worker_version) throw new BrokerError(422, "Result contract does not exactly match the active lease.");
    const digest = await sha256(body.record_json);
    if (body.digest !== digest) throw new BrokerError(422, "digest does not match the UTF-8 record_json bytes.");
    let record: unknown; try { record = JSON.parse(body.record_json); } catch { throw new BrokerError(422, "record_json must be valid JSON."); }
    const checked = validateRecord(record, { modelId: lease.model_id, revision: lease.model_revision, language: lease.language, euroevalVersion: lease.euroeval_version });
    const identityKey = `euroeval:worker:record-identity:${await sha256(checked.identity)}`;
    const existing = await redisGet<StoredIdentity>(identityKey);
    const path = `volunteer/submissions/${lease.lease_id}/results/${digest}.json`;
    if (existing && existing.digest !== digest) throw new BrokerError(409, "This canonical record identity already has a different digest.");
    if (existing?.status === "uploaded") {
      await redis("SADD", `euroeval:worker:results:${lease.lease_id}`, JSON.stringify({ digest, identity: checked.identity, path: existing.path || path, warnings: existing.warnings || checked.warnings }));
      await redis("SADD", `euroeval:worker:reservations:${lease.lease_id}`, JSON.stringify({ digest, identity: checked.identity }));
      return json(200, { protocol_version: PROTOCOL_VERSION, status: "duplicate", digest, identity: checked.identity, path: existing.path || path, warnings: existing.warnings || checked.warnings });
    }
    const reservation: StoredIdentity = { digest, lease_id: lease.lease_id, status: "uploading", issue_number: lease.issue_number, language: lease.language, path, warnings: checked.warnings };
    const claim = await reserveResultIdentity(identityKey, JSON.stringify(reservation), digest, lease.lease_id, 24 * 60 * 60);
    if (claim === "busy") throw new BrokerError(409, "Another submission for this identity is in progress; retry later.");
    if (claim === "duplicate") throw new BrokerError(409, "This canonical record identity was already accepted by another lease.");
    try {
      // The worker's exact JSON text is the durable representation.
      await uploadStaging(path, body.record_json);
      reservation.status = "uploaded";
      if (!(await completeResultIdentity(identityKey, JSON.stringify(reservation), digest, lease.lease_id, 30 * 24 * 60 * 60))) throw new BrokerError(409, "Result reservation was replaced during upload; retry safely.");
      await redis("SADD", `euroeval:worker:results:${lease.lease_id}`, JSON.stringify({ digest, identity: checked.identity, path, warnings: checked.warnings }));
      await redis("SADD", `euroeval:worker:reservations:${lease.lease_id}`, JSON.stringify({ digest, identity: checked.identity }));
      await redis("EXPIRE", `euroeval:worker:results:${lease.lease_id}`, String(30 * 24 * 60 * 60));
      await redis("EXPIRE", `euroeval:worker:reservations:${lease.lease_id}`, String(30 * 24 * 60 * 60));
    } catch (error) { await abortResultIdentity(identityKey, digest, lease.lease_id).catch(() => undefined); throw error; }
    return json(201, { protocol_version: PROTOCOL_VERSION, status: "uploaded", digest, identity: checked.identity, path, warnings: checked.warnings });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to store result." });
  }
}
