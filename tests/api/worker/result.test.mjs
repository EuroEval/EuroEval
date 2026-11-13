import test from "node:test";
import assert from "node:assert/strict";
import { fetch as resultFetch } from "../../../api/worker/result.ts";
import { signVolunteerMarker, sha256 } from "../../../api/worker/_lib.ts";
import { withRedisRestBridge } from "./redis-bridge.mjs";
import { CANARY_CORPUS_ID, CANARY_CORPUS_REVISION, CANARY_CORPUS_SHA256,
  CANARY_EVIDENCE_SCHEMA, CANARY_GENERATION_VERSION, CANARY_NORMALISER_VERSION,
  CANARY_PROTOCOL_VERSION } from "../../../api/worker/_lib/canary.ts";

test("result rejects unsupported HTTP methods", async () => {
  const response = await resultFetch(new Request("https://euroeval.test/api/worker/result", {
    method: "GET",
  }));

  assert.equal(response.status, 405);
  assert.deepEqual(await response.json(), { error: "Method not allowed" });
});

const revision = "a".repeat(40);

function resultRecord({ canary = false, revisionValue } = {}) {
  const modelInfo = { id: "org/model", ...(revisionValue === undefined ? {} : { revision: revisionValue }) };
  const details = { dataset: canary ? "contamination-canary-da" : "dataset", task: canary
    ? "contamination-detection" : "classification", raw_results: [], languages: ["da"],
  validation_split: true, few_shot: true };
  if (canary) details.contamination_canary_evidence = {
    schema_version: CANARY_EVIDENCE_SCHEMA, protocol_version: CANARY_PROTOCOL_VERSION,
    normaliser_version: CANARY_NORMALISER_VERSION, generation_version: CANARY_GENERATION_VERSION,
    model_id: "org/model", requested_revision: revision, resolved_revision: revision,
    identity_kind: "immutable", backend: "vllm:base", corpus_id: CANARY_CORPUS_ID,
    corpus_revision: CANARY_CORPUS_REVISION, corpus_sha256: CANARY_CORPUS_SHA256,
    row_count: 256, status: "collected", reason: null,
    observations: Array.from({ length: 256 }, (_, index) => ({ row_id: `row-${String(index).padStart(3, "0")}`,
      prompt_sha256: "b".repeat(64), normalised_completion: "amber forest" })),
  };
  return { schema_version: "0.3.0", model_info: modelInfo,
    eval_library: { name: "euroeval", version: "1.0.0", additional_details: details },
    evaluation_results: [{ evaluation_name: "accuracy", source_data: { dataset_name: details.dataset },
      metric_config: { lower_is_better: false }, score_details: { score: 0.5 } }] };
}

async function submitRecord({ canary = false, recordRevision, requiredFlag, forgedBody = false } = {}) {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  Object.assign(process.env, { VOLUNTEER_MARKER_SECRET: "marker-secret", GITHUB_TOKEN: "github-token",
    HF_STAGING_BUCKET: "EuroEval/staging", HF_TOKEN: "hf_test-token" });
  const leaseId = "revision-compatibility-lease";
  const language = "da";
  const lease = { lease_id: leaseId, issue_number: 12, language, worker: "worker", contributor: "alice",
    model_id: "org/model", model_revision: revision, euroeval_version: "1.0.0", image_digest: "sha256:image",
    worker_version: "1.0.0", model_type: "generative", expires_at: new Date(Date.now() + 60_000).toISOString(),
    ...(requiredFlag === undefined ? {} : { record_revision_required: requiredFlag }) };
  const marker = await signVolunteerMarker(12, { protocol_version: "volunteer-worker/v1",
    coordinator: "coordinator", submission: "active", leases: [{ lease_id: leaseId, language,
      worker: "worker", contributor: "alice", expires_at: lease.expires_at }] });
  const issue = { number: 12, state: "open", body: `request\n\n<!-- euroeval-volunteer-worker:v1 ${JSON.stringify(marker)} -->`,
    assignees: [{ login: "alice" }] };
  const record = resultRecord({ canary, revisionValue: recordRevision });
  const recordJson = JSON.stringify(record);
  const payload = { protocol_version: "volunteer-worker/v1", lease_id: leaseId, issue_number: 12,
    language, model_id: "org/model", model_revision: revision, euroeval_version: "1.0.0",
    image_digest: "sha256:image", worker_version: "1.0.0", record_json: recordJson,
    digest: await sha256(recordJson), ...(forgedBody ? { record_revision_required: false, capabilities: [] } : {}) };
  const credential = "test-worker-credential";
  const credentialHash = await sha256(credential);
  const priorBytes = new TextEncoder().encode(recordJson);
  try {
    return await withRedisRestBridge(async () => {
      const redisFetch = globalThis.fetch;
      const redisSet = async (key, value) => {
        await redisFetch("https://redis.test", { method: "POST", body: JSON.stringify(["SET", key, value]) });
      };
      await redisSet(`euroeval:worker:credential:${credentialHash}`, JSON.stringify({ contributor: "alice" }));
      await redisSet(`euroeval:worker:lease-id:${leaseId}`, JSON.stringify(lease));
      globalThis.fetch = async (input, init = {}) => {
        const url = String(input); const method = init.method || "GET";
        if (url === "https://redis.test") return redisFetch(input, init);
        if (url.endsWith("/issues/12") && method === "GET") return Response.json(issue);
        if (url.includes("xet-write-token")) return Response.json({ accessToken: "xet-token",
          casUrl: "https://cas.test", exp: Math.floor(Date.now() / 1000) + 3600 });
        if (url.endsWith("/batch")) return Response.json({ failed: [] });
        if (url.includes("huggingface.co/api/buckets/")) return Response.json({ private: true });
        if (url.includes("cas.test")) return method === "GET" ?
          new Response(null, { status: 404 }) : Response.json({ ok: true });
        if (url.includes("huggingface.co")) return Response.json({ failed: [] });
        throw new Error(`Unexpected request: ${method} ${url}`);
      };
      const response = await resultFetch(new Request("https://euroeval.test/api/worker/result", {
        method: "POST", headers: { authorization: `Bearer ${credential}`, "content-type": "application/json" },
        body: JSON.stringify(payload),
      }));
      const identity = JSON.stringify(["org/model", canary ? "contamination-canary-da" : "dataset", true, true]);
      const identityKey = `euroeval:worker:record-identity:${await sha256(identity)}`;
      const storedResponse = await redisFetch("https://redis.test", {
        method: "POST", body: JSON.stringify(["GET", identityKey]),
      });
      const storedIdentity = (await storedResponse.json()).result;
      assert.deepEqual([...new TextEncoder().encode(recordJson)], [...priorBytes]);
      return { response, recordJson, leaseId, storedIdentity: JSON.parse(storedIdentity) };
    });
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
}

test("result handler allows legacy missing revisions only for compatible leases", async () => {
  for (const canary of [false, true]) {
    for (const requiredFlag of [false, undefined]) {
      const { response, recordJson, storedIdentity } = await submitRecord({ canary, requiredFlag });
      assert.equal(response.status, 201, await response.clone().text());
      const body = await response.json();
      assert.equal(body.status, "uploaded");
      assert.equal(body.digest, await sha256(recordJson));
      assert.ok(body.warnings.includes("Legacy record is missing model_info.revision"));
      assert.equal(storedIdentity.digest, body.digest);
      assert.deepEqual(storedIdentity.warnings, body.warnings);
    }
  }
});

test("result handler enforces modern revision requirements against caller forgeries", async () => {
  for (const recordRevision of [undefined, null, "wrong-revision"]) {
    const { response } = await submitRecord({ requiredFlag: true, recordRevision, forgedBody: true });
    assert.equal(response.status, 422);
    assert.match((await response.json()).error, /model_info\.revision/i);
  }
  const { response } = await submitRecord({ requiredFlag: false, recordRevision: null });
  assert.equal(response.status, 422);
  assert.match((await response.json()).error, /model_info\.revision/i);
});

test("result handler accepts a modern record with the lease revision", async () => {
  const { response } = await submitRecord({ requiredFlag: true, recordRevision: revision });
  assert.equal(response.status, 201, await response.clone().text());
  const body = await response.json();
  assert.equal(body.status, "uploaded");
  assert.deepEqual(body.warnings, []);
});
