import { fetchWithRetry } from "./http.js";
import {
  BrokerError,
  ConfigurationError,
  canonicalJson,
  optionalEnv,
  randomToken,
  sha256,
} from "./protocol.js";
import type { Lease } from "./protocol.js";
import { getLeaseById, leaseTtl, redis, redisGet } from "./redis.js";

export const CANARY_PROTOCOL_VERSION = "private-completion-canary/v1" as const;
export const CANARY_CORPUS_ID = "EuroEval/watermark-audit" as const;
export const CANARY_CORPUS_REVISION = "16d468bbacc284c912a8598a392239af2387ca53" as const;
export const CANARY_CORPUS_SHA256 = "37258fb324cf400bba4bd57cda430a73393928f5e09506594c3678adc38ff324" as const;
export const CANARY_EVIDENCE_SCHEMA = "contamination-canary-evidence/v1" as const;
export const CANARY_NORMALISER_VERSION = "first-two-words-nfc/v1" as const;
export const CANARY_GENERATION_VERSION = "greedy-continuation/v1" as const;

export type CanaryLease = {
  status: "required" | "cached" | "reserved" | "not_applicable";
  protocol_version: typeof CANARY_PROTOCOL_VERSION;
  corpus_revision: typeof CANARY_CORPUS_REVISION;
  corpus_sha256: typeof CANARY_CORPUS_SHA256;
  reason?: string;
  reservation_id?: string;
};

type Reservation = {
  identity: string;
  lease_id: string;
  reservation_id: string;
};

export type CanaryReceipt = {
  identity: string;
  lease_id: string;
  reservation_id: string;
  digest: string;
  accepted_at: string;
  path: string;
};

const receiptKey = (identity: string): string => `euroeval:worker:canary:receipt:${identity}`;
const reservationKey = (identity: string): string => `euroeval:worker:canary:reservation:${identity}`;
let canaryCorpus: string | null = null;

export async function findCanaryLease(leaseId: string): Promise<Lease | null> {
  const active = await getLeaseById(leaseId);
  if (active) return active;
  const finalised = await redisGet<{ status?: string; lease?: Lease }>(
    `euroeval:worker:finalisation:${leaseId}`,
  );
  return finalised?.status === "ready" && finalised.lease ? finalised.lease : null;
}

export async function storedCanaryPaths(identity: string): Promise<string[]> {
  const bucket = optionalEnv("HF_CANARY_EVIDENCE_BUCKET", "");
  const token = optionalEnv("HF_TOKEN", "");
  if (!/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(bucket) || !token) {
    throw new ConfigurationError("HF_CANARY_EVIDENCE_BUCKET and HF_TOKEN are required.");
  }
  const prefix = encodeURIComponent(`v1/${identity}/`);
  const response = await fetchWithRetry(
    `https://huggingface.co/api/buckets/${bucket}/tree/${prefix}?recursive=true`,
    { headers: { accept: "application/json", authorization: `Bearer ${token}` } },
  );
  if (!response.ok) {
    throw new BrokerError(502, `Canary evidence lookup failed with HTTP ${response.status}.`);
  }
  const entries = await response.json() as unknown;
  if (!Array.isArray(entries)) throw new BrokerError(502, "Canary evidence lookup returned invalid data.");
  return entries.flatMap((entry) => {
    if (!entry || typeof entry !== "object") return [];
    const item = entry as Record<string, unknown>;
    return item.type === "file" && typeof item.path === "string" && item.path.endsWith(".json")
      ? [item.path]
      : [];
  });
}

export async function canaryIdentity(modelId: string, revision: string): Promise<string> {
  return sha256(canonicalJson({
    model_id: modelId.trim().replace(/\/$/, "").toLowerCase(),
    model_revision: revision.trim() || "main",
    protocol_version: CANARY_PROTOCOL_VERSION,
    corpus_revision: CANARY_CORPUS_REVISION,
    corpus_sha256: CANARY_CORPUS_SHA256,
  }));
}

export async function reserveCanary(
  modelId: string,
  revision: string,
  modelType: string,
  leaseId: string,
): Promise<CanaryLease> {
  const common = {
    protocol_version: CANARY_PROTOCOL_VERSION,
    corpus_revision: CANARY_CORPUS_REVISION,
    corpus_sha256: CANARY_CORPUS_SHA256,
  } as const;
  if (optionalEnv("CONTAMINATION_CANARY_ENABLED", "0") !== "1") {
    return { ...common, status: "not_applicable", reason: "disabled" };
  }
  const identity = await canaryIdentity(modelId, revision);
  if (await redisGet<CanaryReceipt>(receiptKey(identity))) return { ...common, status: "cached" };
  if ((await storedCanaryPaths(identity)).length) return { ...common, status: "cached" };
  const reservation: Reservation = {
    identity,
    lease_id: leaseId,
    reservation_id: randomToken(18),
  };
  const encoded = JSON.stringify(reservation);
  const result = await redis(
    "EVAL",
    `if redis.call('GET',KEYS[2]) then return 'receipt' end
     local current=redis.call('GET',KEYS[1])
     if not current then
       redis.call('SET',KEYS[1],ARGV[1],'EX',ARGV[2]); return ARGV[1]
     end
     local item=cjson.decode(current)
     if item.identity == ARGV[3] and item.lease_id == ARGV[4] then
       redis.call('EXPIRE',KEYS[1],ARGV[2]); return current
     end
     return nil`,
    "2",
    reservationKey(identity),
    receiptKey(identity),
    encoded,
    String(leaseTtl()),
    identity,
    leaseId,
  );
  if (result === "receipt") return { ...common, status: "cached" };
  if (typeof result !== "string") return { ...common, status: "reserved" };
  const held = JSON.parse(result) as Reservation;
  if (held.identity !== identity || held.lease_id !== leaseId) {
    return { ...common, status: "reserved" };
  }
  return {
    ...common,
    status: "required",
    ...(modelType === "generative" ? {} : { reason: "encoder" }),
    reservation_id: held.reservation_id,
  };
}

export async function preserveCanaryReservation(lease: {
  model_id: string;
  model_revision: string;
  lease_id: string;
  contamination_canary?: CanaryLease;
}): Promise<void> {
  const instruction = lease.contamination_canary;
  if (!instruction || instruction.status !== "required" || !instruction.reservation_id) return;
  const identity = await canaryIdentity(lease.model_id, lease.model_revision);
  await redis(
    "EVAL",
    `local current=redis.call('GET',KEYS[1])
     if not current then return 0 end
     local item=cjson.decode(current)
     if item.lease_id == ARGV[1] and item.reservation_id == ARGV[2] then
       return redis.call('EXPIRE',KEYS[1],ARGV[3])
     end
     return 0`,
    "1",
    reservationKey(identity),
    lease.lease_id,
    instruction.reservation_id,
    String(7 * 24 * 60 * 60),
  );
}

export async function releaseCanary(lease: {
  model_id: string;
  model_revision: string;
  lease_id: string;
  contamination_canary?: CanaryLease;
}): Promise<void> {
  const instruction = lease.contamination_canary;
  if (!instruction || instruction.status !== "required" || !instruction.reservation_id) return;
  const identity = await canaryIdentity(lease.model_id, lease.model_revision);
  await redis(
    "EVAL",
    `local current=redis.call('GET',KEYS[1])
     if not current then return 0 end
     local item=cjson.decode(current)
     if item.lease_id == ARGV[1] and item.reservation_id == ARGV[2] then
       return redis.call('DEL',KEYS[1])
     end
     return 0`,
    "1",
    reservationKey(identity),
    lease.lease_id,
    instruction.reservation_id,
  );
}

export async function requireCanaryReservation(
  modelId: string,
  revision: string,
  leaseId: string,
  reservationId: string,
): Promise<{ identity: string; reservation: Reservation }> {
  const identity = await canaryIdentity(modelId, revision);
  const reservation = await redisGet<Reservation>(reservationKey(identity));
  if (
    !reservation ||
    reservation.identity !== identity ||
    reservation.lease_id !== leaseId ||
    reservation.reservation_id !== reservationId
  ) {
    throw new BrokerError(409, "Canary reservation is absent or belongs to another lease.");
  }
  return { identity, reservation };
}

export async function fetchCanaryCorpus(): Promise<string> {
  if (canaryCorpus !== null) return canaryCorpus;
  const token = optionalEnv("HF_TOKEN", "");
  if (!token) throw new ConfigurationError("HF_TOKEN is required for the private canary corpus.");
  const url = `https://huggingface.co/datasets/${CANARY_CORPUS_ID}/resolve/${CANARY_CORPUS_REVISION}/test.jsonl`;
  const response = await fetchWithRetry(url, {
    headers: { authorization: `Bearer ${token}`, accept: "application/x-ndjson" },
  });
  if (!response.ok) throw new BrokerError(502, `Canary corpus fetch failed with HTTP ${response.status}.`);
  const content = await response.text();
  if (await sha256(content) !== CANARY_CORPUS_SHA256) {
    throw new ConfigurationError("Private canary corpus does not match its frozen digest.");
  }
  const lines = content.trimEnd().split("\n");
  if (lines.length !== 256) throw new ConfigurationError("Private canary corpus must contain 256 rows.");
  canaryCorpus = content;
  return content;
}

export async function validateCanaryObservationBinding(value: unknown): Promise<void> {
  const evidence = value as Record<string, unknown>;
  if (evidence.status !== "collected") return;
  const observations = evidence.observations as Array<Record<string, unknown>>;
  const lines = (await fetchCanaryCorpus()).trimEnd().split("\n");
  for (let index = 0; index < lines.length; index += 1) {
    let row: unknown;
    try { row = JSON.parse(lines[index]); }
    catch { throw new ConfigurationError("Private canary corpus contains invalid JSON."); }
    if (!row || typeof row !== "object" || Array.isArray(row)) {
      throw new ConfigurationError("Private canary corpus row is invalid.");
    }
    const item = row as Record<string, unknown>;
    const match = typeof item.text === "string"
      ? item.text.match(/^([\s\S]+ referred to) ([a-z]+) ([a-z]+)\.$/)
      : null;
    if (
      Object.keys(item).length !== 2 || typeof item.row_id !== "string" || !match ||
      observations[index].row_id !== item.row_id ||
      observations[index].prompt_sha256 !== await sha256(match[1])
    ) throw new BrokerError(422, "Canary observations do not match the frozen corpus.");
  }
}

export function validateCanaryEvidence(
  value: unknown,
  expected: { modelId: string; revision: string },
): void {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new BrokerError(422, "Canary evidence must be an object.");
  }
  const evidence = value as Record<string, unknown>;
  const fields = [
    "schema_version", "protocol_version", "normaliser_version", "generation_version",
    "model_id", "requested_revision", "resolved_revision", "identity_kind", "backend",
    "corpus_id", "corpus_revision", "corpus_sha256", "row_count", "status", "reason",
    "observations",
  ];
  if (Object.keys(evidence).some((field) => !fields.includes(field)) || fields.some((field) => !(field in evidence))) {
    throw new BrokerError(422, "Canary evidence contains undeclared or missing fields.");
  }
  if (
    evidence.schema_version !== CANARY_EVIDENCE_SCHEMA ||
    evidence.protocol_version !== CANARY_PROTOCOL_VERSION ||
    evidence.normaliser_version !== CANARY_NORMALISER_VERSION ||
    evidence.generation_version !== CANARY_GENERATION_VERSION ||
    evidence.model_id !== expected.modelId ||
    evidence.requested_revision !== expected.revision ||
    evidence.resolved_revision !== expected.revision ||
    evidence.identity_kind !== "immutable" ||
    evidence.corpus_id !== CANARY_CORPUS_ID ||
    evidence.corpus_revision !== CANARY_CORPUS_REVISION ||
    evidence.corpus_sha256 !== CANARY_CORPUS_SHA256 ||
    evidence.row_count !== 256 ||
    !["collected", "not_applicable", "unsupported", "failed"].includes(String(evidence.status)) ||
    typeof evidence.backend !== "string" || !evidence.backend
  ) {
    throw new BrokerError(422, "Canary evidence identity or protocol is invalid.");
  }
  if (!Array.isArray(evidence.observations)) throw new BrokerError(422, "Canary observations must be an array.");
  if (evidence.status !== "collected") {
    const validReason =
      (evidence.status === "not_applicable" && evidence.reason === "encoder") ||
      (evidence.status === "unsupported" && evidence.reason === "backend_unsupported") ||
      (evidence.status === "failed" && [
        "corpus_unavailable", "generation_failed", "incomplete_generation",
      ].includes(String(evidence.reason)));
    if (evidence.observations.length || !validReason) {
      throw new BrokerError(422, "Non-collected canary evidence is malformed.");
    }
    return;
  }
  if (evidence.reason !== null || evidence.observations.length !== 256) {
    throw new BrokerError(422, "Collected canary evidence must contain 256 rows.");
  }
  const rowIds = new Set<string>();
  for (const item of evidence.observations) {
    if (!item || typeof item !== "object" || Array.isArray(item)) throw new BrokerError(422, "Canary observation is invalid.");
    const row = item as Record<string, unknown>;
    if (
      Object.keys(row).some((field) => !["row_id", "prompt_sha256", "normalised_completion"].includes(field)) ||
      Object.keys(row).length !== 3 || typeof row.row_id !== "string" || !row.row_id || rowIds.has(row.row_id) ||
      typeof row.prompt_sha256 !== "string" || !/^[0-9a-f]{64}$/.test(row.prompt_sha256) ||
      typeof row.normalised_completion !== "string" ||
      new TextEncoder().encode(row.normalised_completion).length > 256 ||
      !/^(?:\p{L}+(?:[-']\p{L}+)*(?: \p{L}+(?:[-']\p{L}+)*)?)?$/u.test(row.normalised_completion)
    ) throw new BrokerError(422, "Canary observation is invalid.");
    rowIds.add(row.row_id);
  }
}

export async function getCanaryReceipt(
  identity: string,
): Promise<CanaryReceipt | null> {
  return redisGet<CanaryReceipt>(receiptKey(identity));
}

export async function commitCanaryReceipt(
  identity: string,
  reservation: Reservation,
  receipt: CanaryReceipt,
): Promise<"accepted" | "duplicate" | "conflict" | "lost"> {
  const result = await redis(
    "EVAL",
    `local receipt=redis.call('GET',KEYS[2])
     if receipt then
       local item=cjson.decode(receipt)
       if item.digest == ARGV[1] then return 'duplicate' end
       return 'conflict'
     end
     local current=redis.call('GET',KEYS[1])
     if not current then return 'lost' end
     local held=cjson.decode(current)
     if held.identity ~= ARGV[2] or held.lease_id ~= ARGV[3] or held.reservation_id ~= ARGV[4] then return 'lost' end
     redis.call('SET',KEYS[2],ARGV[5],'EX',ARGV[6]); redis.call('DEL',KEYS[1]); return 'accepted'`,
    "2",
    reservationKey(identity),
    receiptKey(identity),
    receipt.digest,
    identity,
    reservation.lease_id,
    reservation.reservation_id,
    JSON.stringify(receipt),
    String(365 * 24 * 60 * 60),
  );
  return ["accepted", "duplicate", "conflict"].includes(String(result))
    ? result as "accepted" | "duplicate" | "conflict"
    : "lost";
}
