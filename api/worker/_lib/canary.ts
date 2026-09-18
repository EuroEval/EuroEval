import { fetchWithRetry } from "./http.js";
import {
  BrokerError,
  ConfigurationError,
  env,
  sha256,
} from "./protocol.js";

export const CANARY_PROTOCOL_VERSION = "private-completion-canary/v1" as const;
export const CANARY_CORPUS_ID = "EuroEval/watermark-audit" as const;
export const CANARY_CORPUS_REVISION = "16d468bbacc284c912a8598a392239af2387ca53" as const;
export const CANARY_CORPUS_SHA256 = "37258fb324cf400bba4bd57cda430a73393928f5e09506594c3678adc38ff324" as const;
export const CANARY_EVIDENCE_SCHEMA = "contamination-canary-evidence/v1" as const;
export const CANARY_NORMALISER_VERSION = "first-two-words-nfc/v1" as const;
export const CANARY_GENERATION_VERSION = "greedy-continuation/v1" as const;
export const CANARY_RESULT_DATASET = "contamination-canary" as const;

export type CanaryLease = {
  status: "required" | "not_applicable";
  protocol_version: typeof CANARY_PROTOCOL_VERSION;
  corpus_revision: typeof CANARY_CORPUS_REVISION;
  corpus_sha256: typeof CANARY_CORPUS_SHA256;
  reason?: string;
};

let canaryCorpus: string | null = null;

export async function reserveCanary(
  _modelId: string,
  _revision: string,
  modelType: string,
  _leaseId: string,
): Promise<CanaryLease> {
  const common = {
    protocol_version: CANARY_PROTOCOL_VERSION,
    corpus_revision: CANARY_CORPUS_REVISION,
    corpus_sha256: CANARY_CORPUS_SHA256,
  } as const;
  return {
    ...common,
    status: "required",
    ...(modelType === "generative" ? {} : { reason: "encoder" }),
  };
}

export async function fetchCanaryCorpus(): Promise<string> {
  if (canaryCorpus !== null) return canaryCorpus;
  const token = env("HF_TOKEN");
  const url = `https://huggingface.co/datasets/${CANARY_CORPUS_ID}/resolve/${CANARY_CORPUS_REVISION}/test.jsonl`;
  const response = await fetchWithRetry(url, {
    headers: { authorization: `Bearer ${token}`, accept: "application/x-ndjson" },
  });
  if (!response.ok) {
    throw new BrokerError(502, `Canary corpus fetch failed with HTTP ${response.status}.`);
  }
  const content = await response.text();
  if (await sha256(content) !== CANARY_CORPUS_SHA256) {
    throw new ConfigurationError("Private canary corpus does not match its frozen digest.");
  }
  const lines = content.trimEnd().split("\n");
  if (lines.length !== 256) {
    throw new ConfigurationError("Private canary corpus must contain 256 rows.");
  }
  canaryCorpus = content;
  return content;
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
  if (
    Object.keys(evidence).some((field) => !fields.includes(field)) ||
    fields.some((field) => !(field in evidence))
  ) {
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
  if (!Array.isArray(evidence.observations)) {
    throw new BrokerError(422, "Canary observations must be an array.");
  }
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
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      throw new BrokerError(422, "Canary observation is invalid.");
    }
    const row = item as Record<string, unknown>;
    if (
      Object.keys(row).some((field) => ![
        "row_id", "prompt_sha256", "normalised_completion",
      ].includes(field)) ||
      Object.keys(row).length !== 3 || typeof row.row_id !== "string" || !row.row_id ||
      rowIds.has(row.row_id) || typeof row.prompt_sha256 !== "string" ||
      !/^[0-9a-f]{64}$/.test(row.prompt_sha256) ||
      typeof row.normalised_completion !== "string" ||
      new TextEncoder().encode(row.normalised_completion).length > 256 ||
      !/^(?:\p{L}+(?:[-']\p{L}+)*(?: \p{L}+(?:[-']\p{L}+)*)?)?$/u.test(row.normalised_completion)
    ) {
      throw new BrokerError(422, "Canary observation is invalid.");
    }
    rowIds.add(row.row_id);
  }
}
