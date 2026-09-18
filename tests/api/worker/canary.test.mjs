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
  findCanaryLease,
  storedCanaryPaths,
  validateCanaryEvidence,
} from "../../../api/worker/_lib/canary.ts";
import { fetch as corpusEndpoint } from "../../../api/worker/canary-corpus.ts";
import { fetch as evidenceEndpoint } from "../../../api/worker/canary-evidence.ts";

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
      row_id: `row-${index}`,
      prompt_sha256: index.toString(16).padStart(64, "0"),
      normalised_completion: "plain words",
    })),
  };
}

test("canary evidence accepts only the plaintext-free bounded contract", () => {
  const value = evidence();
  assert.doesNotThrow(() => validateCanaryEvidence(value, {
    modelId: "org/model",
    revision,
  }));
  assert.throws(
    () => validateCanaryEvidence({ ...value, secret: "forbidden" }, {
      modelId: "org/model",
      revision,
    }),
    /undeclared/,
  );
  assert.throws(
    () => validateCanaryEvidence({
      ...value,
      observations: [...value.observations.slice(0, 255), value.observations[0]],
    }, { modelId: "org/model", revision }),
    /invalid/,
  );
  assert.throws(
    () => validateCanaryEvidence({
      ...value,
      observations: value.observations.map((item, index) => index
        ? item
        : { ...item, normalised_completion: "three plaintext words" }),
    }, { modelId: "org/model", revision }),
    /invalid/,
  );
  assert.throws(
    () => validateCanaryEvidence({
      ...value,
      status: "failed",
      reason: "raw provider exception text",
      observations: [],
    }, { modelId: "org/model", revision }),
    /malformed/,
  );
  assert.throws(
    () => validateCanaryEvidence({ ...value, identity_kind: "mutable" }, {
      modelId: "org/model",
      revision,
    }),
    /identity or protocol/,
  );
});

test("encoder evidence uses the typed not-applicable result", () => {
  const value = {
    ...evidence(),
    status: "not_applicable",
    reason: "encoder",
    observations: [],
  };
  assert.doesNotThrow(() => validateCanaryEvidence(value, {
    modelId: "org/model",
    revision,
  }));
});

test("finalised lease tombstone authorises delayed evidence retry", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.UPSTASH_REDIS_REST_URL = "https://redis.test";
  process.env.UPSTASH_REDIS_REST_TOKEN = "redis-token";
  const lease = {
    lease_id: "lease", contributor: "alice", model_id: "org/model",
    model_revision: revision,
  };
  globalThis.fetch = async (_input, init) => {
    const command = JSON.parse(init.body);
    const value = command[1] === "euroeval:worker:finalisation:lease"
      ? JSON.stringify({ status: "ready", lease })
      : null;
    return Response.json({ result: value });
  };
  try {
    assert.deepEqual(await findCanaryLease("lease"), lease);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

test("durable bucket evidence survives Redis receipt loss", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.HF_CANARY_EVIDENCE_BUCKET = "EuroEval/private-evidence";
  process.env.HF_TOKEN = "token";
  globalThis.fetch = async (input) => {
    assert.match(String(input), /\/api\/buckets\/EuroEval\/private-evidence\/tree\//);
    return Response.json([
      { type: "file", path: `v1/identity/${"a".repeat(64)}.json` },
    ]);
  };
  try {
    assert.deepEqual(await storedCanaryPaths("identity"), [
      `v1/identity/${"a".repeat(64)}.json`,
    ]);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

test("canary endpoints reject unsupported methods before authentication", async () => {
  for (const endpoint of [corpusEndpoint, evidenceEndpoint]) {
    const response = await endpoint(new Request("https://example.test", { method: "GET" }));
    assert.equal(response.status, 405);
  }
});
