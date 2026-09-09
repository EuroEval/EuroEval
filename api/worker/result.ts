import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, authenticate, getLeaseById, json, method, readJson,
  recordDigest, redis, redisGet, redisSet, uploadStaging, validateRecord, requireProtocol,
} from "./_lib";

export const config = { runtime: "edge" };
const MAX_RESULT_BODY = 2 * 1024 * 1024;

type StoredIdentity = { digest: string; lease_id: string; status: "uploading" | "uploaded"; issue_number: number; language: string };

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const identity = await authenticate(req);
    const body = await readJson(req, MAX_RESULT_BODY);
    requireProtocol(body);
    const required = ["lease_id", "issue_number", "language", "model_id", "model_revision", "euroeval_version", "image_digest", "worker_version", "record", "digest"];
    for (const field of required) if (body[field] === undefined) throw new BrokerError(400, `${field} is required.`);
    if (typeof body.lease_id !== "string" || typeof body.language !== "string" || typeof body.model_id !== "string" || typeof body.model_revision !== "string" || typeof body.euroeval_version !== "string" || typeof body.image_digest !== "string" || typeof body.worker_version !== "string" || typeof body.digest !== "string" || typeof body.issue_number !== "number" || !Number.isSafeInteger(body.issue_number)) throw new BrokerError(400, "Result contract fields have invalid types.");
    const lease = await getLeaseById(body.lease_id);
    if (!lease || lease.worker !== identity.hash.slice(0, 24) || Date.parse(lease.expires_at) <= Date.now()) throw new BrokerError(409, "Lease is absent, expired, or belongs to another worker.");
    if (body.issue_number !== lease.issue_number || body.language !== lease.language || body.model_id !== lease.model_id || body.model_revision !== lease.model_revision || body.euroeval_version !== lease.euroeval_version || body.image_digest !== lease.image_digest || body.worker_version !== lease.worker_version) throw new BrokerError(422, "Result contract does not exactly match the active lease.");
    const checked = validateRecord(body.record, { modelId: lease.model_id, revision: lease.model_revision, language: lease.language });
    const digest = await recordDigest(body.record);
    if (body.digest !== digest) throw new BrokerError(422, "digest does not match the canonical record JSON.");
    const identityKey = `euroeval:worker:record-identity:${await recordDigest(checked.identity)}`;
    const existing = await redisGet<StoredIdentity>(identityKey);
    if (existing && existing.digest !== digest) throw new BrokerError(409, "This canonical record identity already has a different digest.");
    if (existing?.status === "uploaded") {
      await redis("SADD", `euroeval:worker:results:${lease.lease_id}`, JSON.stringify({ digest, identity: checked.identity, path: `volunteer/${lease.issue_number}/${lease.language}/${digest}.json` }));
      return json(200, { protocol_version: PROTOCOL_VERSION, status: "duplicate", digest, identity: checked.identity });
    }
    const reservation: StoredIdentity = { digest, lease_id: lease.lease_id, status: "uploading", issue_number: lease.issue_number, language: lease.language };
    if (!existing && !(await redisSet(identityKey, JSON.stringify(reservation), 24 * 60 * 60, true))) throw new BrokerError(409, "Another submission for this identity is in progress; retry later.");
    if (existing?.status === "uploading" && existing.lease_id !== lease.lease_id) throw new BrokerError(409, "Another submission for this identity is in progress; retry later.");
    const path = `volunteer/${lease.issue_number}/${lease.language}/${digest}.json`;
    try {
      await uploadStaging(path, JSON.stringify(body.record));
      reservation.status = "uploaded";
      await redisSet(identityKey, JSON.stringify(reservation), 30 * 24 * 60 * 60);
      await redis("SADD", `euroeval:worker:results:${lease.lease_id}`, JSON.stringify({ digest, identity: checked.identity, path }));
      await redis("EXPIRE", `euroeval:worker:results:${lease.lease_id}`, String(30 * 24 * 60 * 60));
    } catch (error) {
      await redis("DEL", identityKey).catch(() => undefined);
      throw error;
    }
    return json(201, { protocol_version: PROTOCOL_VERSION, status: "uploaded", digest, identity: checked.identity, path });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { error: error instanceof Error ? error.message : "Unable to store result." });
  }
}
