import assert from "node:assert/strict";
import test from "node:test";

import {
  CANARY_CORPUS_ID,
  CANARY_CORPUS_REVISION,
  CANARY_CORPUS_SHA256,
  CANARY_EVIDENCE_SCHEMA,
  CANARY_GENERATION_VERSION,
  CANARY_NORMALISER_VERSION,
  CANARY_PROTOCOL_VERSION,
  reserveCanary,
  validateCanaryEvidence,
} from "../../../api/worker/_lib/canary.ts";
import { fetch as corpusEndpoint } from "../../../api/worker/canary-corpus.ts";
import { validateRecord } from "../../../api/worker/_lib/eee.ts";

const revision = "a".repeat(40);

function evidence() {
  return {
    schema_version: CANARY_EVIDENCE_SCHEMA,
    protocol_version: CANARY_PROTOCOL_VERSION,
    normaliser_version: CANARY_NORMALISER_VERSION,
    generation_version: CANARY_GENERATION_VERSION,
    model_id: "org/model",
    requested_revision: revision,
    resolved_revision: revision,
    identity_kind: "immutable",
    backend: "vllm:base",
    corpus_id: CANARY_CORPUS_ID,
    corpus_revision: CANARY_CORPUS_REVISION,
    corpus_sha256: CANARY_CORPUS_SHA256,
    row_count: 256,
    status: "collected",
    reason: null,
    observations: Array.from({ length: 256 }, (_, index) => ({
      row_id: `row-${String(index).padStart(3, "0")}`,
      prompt_sha256: "b".repeat(64),
      normalised_completion: "amber forest",
    })),
  };
}

test("canary evidence accepts only the plaintext-free bounded contract", () => {
  const value = evidence();
  validateCanaryEvidence(value, { modelId: "org/model", revision });
  assert.throws(
    () => validateCanaryEvidence({ ...value, secret: "forbidden" }, { modelId: "org/model", revision }),
    /undeclared or missing/,
  );
  assert.throws(
    () => validateCanaryEvidence({ ...value, observations: value.observations.map((item, index) => index ? item : { ...item, normalised_completion: "three words here" }) }, { modelId: "org/model", revision }),
    /observation is invalid/,
  );
});

test("ordinary EEE validation accepts embedded string evidence", () => {
  const record = {
    schema_version: "0.2.1",
    model_info: { id: "org/model", name: `org/model@${revision}` },
    eval_library: {
      name: "euroeval",
      version: "18.1.0",
      additional_details: {
        dataset: "contamination-canary-da",
        task: "contamination-detection",
        languages: JSON.stringify(["da"]),
        validation_split: null,
        few_shot: null,
        raw_results: "[]",
        contamination_canary_evidence: JSON.stringify(evidence()),
      },
    },
    evaluation_results: [{
      evaluation_name: "test_collection_success",
      source_data: { dataset_name: "contamination-canary-da" },
      metric_config: { lower_is_better: false, score_type: "continuous", min_score: 0, max_score: 100 },
      score_details: { score: 100, details: { num_failed_instances: "0" } },
    }],
  };
  const checked = validateRecord(record, {
    modelId: "org/model",
    revision,
    language: "da",
    euroevalVersion: "18.1.0",
  });
  assert.equal(JSON.parse(checked.identity)[1], "contamination-canary-da");
  assert.throws(
    () => validateRecord(record, {
      modelId: "org/model",
      revision,
      language: "da",
      euroevalVersion: "18.1.0",
      modelType: "encoder",
    }),
    /encoder canary evidence must be not_applicable\/encoder/,
  );
});

test("encoder evidence uses the typed not-applicable result", () => {
  const value = {
    ...evidence(),
    backend: "hf",
    status: "not_applicable",
    reason: "encoder",
    observations: [],
  };
  validateCanaryEvidence(value, { modelId: "org/model", revision });
});

test("leases require collection without a separate evidence reservation", async () => {
  const decoder = await reserveCanary("org/model", revision, "generative", "lease");
  const encoder = await reserveCanary("org/model", revision, "encoder", "lease");
  assert.equal(decoder.status, "required");
  assert.equal("reservation_id" in decoder, false);
  assert.equal(encoder.status, "required");
  assert.equal(encoder.reason, "encoder");
});

test("canary corpus endpoint rejects unsupported methods before authentication", async () => {
  const response = await corpusEndpoint(new Request("https://example.test", { method: "GET" }));
  assert.equal(response.status, 405);
});
