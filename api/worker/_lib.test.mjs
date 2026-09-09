import test from "node:test";
import assert from "node:assert/strict";
import { expectedScope, extractModelId, parsePromotionRecords, parseVolunteerMarker, PROMOTION_RESERVATION_TTL, replaceVolunteerMarker, selectedLanguages, validateRecord } from "./_lib.ts";
import { fitsGpu, selectedGpu } from "./_lib/model.ts";
import { reclaimExpiredLease, releaseResultReservations, reserveResultIdentity } from "./_lib/redis.ts";
import { promotionIdentityKey, reservePromotionReservation } from "./_lib/promotion.ts";

test("parses the queue model and language checkboxes", () => {
  const body = "### Model ID\n\norg/model\n\n- [x] Greek\n- [ ] Albanian\n";
  assert.equal(extractModelId("ignored", body), "org/model");
  assert.deepEqual(selectedLanguages(body), ["el"]);
});

test("strictly parses the shared ownership marker", () => {
  assert.equal(parseVolunteerMarker("<!-- euroeval-volunteer-worker:v1 not-json -->"), null);
  const marker = { protocol_version: "volunteer-worker/v1", coordinator: "coordinator", submission: "active", leases: [{ lease_id: "lease", language: "da", worker: "abc", contributor: "contributor", expires_at: new Date(Date.now() + 1000).toISOString() }] };
  const body = replaceVolunteerMarker("queue", marker);
  assert.deepEqual(parseVolunteerMarker(body), marker);
});

test("submission markers require verified server-derived counts", () => {
  const base = { protocol_version: "volunteer-worker/v1", coordinator: "coordinator", submission: "submitted", leases: [] };
  const entry = { submission_id: "one", language: "da", manifest_path: "volunteer/manifests/one.json", submitted_at: "2026-09-06T10:00:00Z", verified_contributor: "alice", result_count: 3, status: "submitted" };
  const retry = { ...entry, submission_id: "two", status: "rejected" };
  const body = replaceVolunteerMarker("queue", { ...base, submissions: [entry, retry] });
  assert.equal(parseVolunteerMarker(body).submissions.length, 2);
  const invalid = replaceVolunteerMarker("queue", { ...base, submissions: [{ ...entry, result_count: 0 }] });
  assert.equal(parseVolunteerMarker(invalid), null);
});

test("validates canonical model identity and score bounds", () => {
  const record = {
    schema_version: "0.2.1",
    model_info: { id: "org/model", revision: "deadbeef" },
    eval_library: { name: "euroeval", version: "0.2.1", additional_details: { dataset: "dataset", task: "task", language: "da", languages: '["da"]', raw_results: "[]", few_shot: false, validation_split: false, num_failed_instances: 0 } },
    evaluation_results: [{ evaluation_name: "accuracy", source_data: { dataset_name: "dataset" }, metric_config: { lower_is_better: false, min_score: 0, max_score: 100 }, score_details: { score: 85 } }],
  };
  assert.equal(validateRecord(record, { modelId: "org/model", revision: "deadbeef", language: "da" }).failed, 0);
  const outOfRange = validateRecord({ ...record, evaluation_results: [{ ...record.evaluation_results[0], score_details: { score: 101 } }] }, { modelId: "org/model", revision: "deadbeef", language: "da" });
  assert.match(outOfRange.warnings[0], /outside declared metric bounds/);
  const negative = validateRecord({ ...record, evaluation_results: [{ ...record.evaluation_results[0], score_details: { score: -1 } }] }, { modelId: "org/model", revision: "deadbeef", language: "da" });
  assert.equal(negative.failed, 0);
  assert.throws(() => validateRecord({ ...record, evaluation_results: [{ ...record.evaluation_results[0], score_details: { score: 85, details: { failed_instances: "[1]" } } }] }, { modelId: "org/model", revision: "deadbeef", language: "da" }));
});

test("promotion reservations outlive long uploads", () => {
  assert.ok(PROMOTION_RESERVATION_TTL >= 24 * 60 * 60);
});

test("promotion evidence binds safe canonical paths to digests", () => {
  const records = parsePromotionRecords([
    { identity: '["org/model","dataset",false,true]', canonical_path: "org_model/dataset__test__fewshot.json", digest: "a".repeat(64) },
  ]);
  assert.equal(records[0].canonical_path, "org_model/dataset__test__fewshot.json");
  assert.throws(() => parsePromotionRecords([
    { identity: '["org/model","dataset",false,true]', canonical_path: "../results.json", digest: "a".repeat(64) },
  ]), /safe path/);
  assert.throws(() => parsePromotionRecords([
    { identity: '["org/model","dataset",false,true]', canonical_path: "org_model/dataset__test__fewshot.json", digest: "not-a-digest" },
  ]), /SHA256/);
  assert.throws(() => parsePromotionRecords([
    { identity: '["org/model","dataset",false,true]', canonical_path: "org_model/dataset__test__fewshot.json", digest: "a".repeat(64) },
    { identity: '["org_model","dataset",false,true]', canonical_path: "org_model/dataset__test__fewshot.json", digest: "a".repeat(64) },
  ]), /not unique/);
});

test("generated trusted scopes are exact-language and versioned", () => {
  const scope = expectedScope("18.0.0.dev", "bert", "da");
  assert.equal(scope.language, "da");
  assert.equal(scope.policy_version, "volunteer-scope/18.0.0.dev0");
  assert.ok(scope.identity_suffixes.length > 0);
});

test("trusted exact-language policy owns the lease language group", () => {
  const original = process.env.VOLUNTEER_SCOPE_POLICY_JSON;
  process.env.VOLUNTEER_SCOPE_POLICY_JSON = JSON.stringify({ policy_version: "test-policy", policies: [{
    euroeval_version: "1.0.0", model_profile: "bert", language: "da", language_group: "policy-da",
    identity_suffixes: [JSON.stringify(["dataset", false, true])],
  }] });
  try {
    const scope = expectedScope("1.0.0", "bert", "da");
    assert.equal(scope.language, "da");
    assert.equal(scope.language_group, "policy-da");
  } finally {
    if (original === undefined) delete process.env.VOLUNTEER_SCOPE_POLICY_JSON;
    else process.env.VOLUNTEER_SCOPE_POLICY_JSON = original;
  }
});

test("fits only the explicitly selected GPU", () => {
  const model = { id: "org/model", revision: "r", config: {}, weight_bytes: 300, repo_bytes: 500, model_profile: "llama" };
  const hardware = {
    free_disk_bytes: 500, gpu_memory_utilisation: 0.8, selected_gpu_index: 1, selected_gpu_uuid: "GPU-1",
    gpus: [{ index: 0, name: "a", uuid: "GPU-0", free_memory_bytes: 100, total_memory_bytes: 100 },
      { index: 1, name: "b", uuid: "GPU-1", free_memory_bytes: 600, total_memory_bytes: 600 }],
  };
  assert.equal(selectedGpu(hardware)?.uuid, "GPU-1");
  assert.equal(fitsGpu(model, hardware), true);
  assert.equal(fitsGpu(model, { ...hardware, selected_gpu_index: 0, selected_gpu_uuid: "GPU-0" }), false);
  assert.equal(fitsGpu(model, { ...hardware, selected_gpu_uuid: "wrong" }), false);
  assert.equal(fitsGpu(model, { ...hardware, selected_gpu_index: 9 }), false);
});

test("fits a model on one reported GPU and requires repository disk", () => {
  const model = { id: "org/model", revision: "r", config: {}, weight_bytes: 100, repo_bytes: 500, model_profile: "llama" };
  const hardware = { free_disk_bytes: 500, gpus: [{ name: "a", uuid: "1", free_memory_bytes: 50, total_memory_bytes: 50 }, { name: "b", uuid: "2", free_memory_bytes: 135, total_memory_bytes: 135 }] };
  assert.equal(fitsGpu(model, hardware), true);
  assert.equal(fitsGpu(model, { ...hardware, free_disk_bytes: 499 }), false);
});

test("result reservation tracking is atomic with identity ownership", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.UPSTASH_REDIS_REST_URL = "https://redis.test";
  process.env.UPSTASH_REDIS_REST_TOKEN = "redis-token";
  const commands = [];
  globalThis.fetch = async (_input, init) => {
    commands.push(JSON.parse(init.body));
    return Response.json({ result: "reserved" });
  };
  try {
    assert.equal(await reserveResultIdentity("identity-key", "{}", "a".repeat(64), "lease", 60,
      "reservations-key", JSON.stringify({ identity: "identity", digest: "a".repeat(64) })), "reserved");
    assert.equal(commands[0][0], "EVAL");
    assert.equal(commands[0][2], "2");
    assert.match(commands[0][1], /SISMEMBER/);
    assert.match(commands[0][1], /SCARD/);
    assert.match(commands[0][1], /SADD/);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

test("expired lease cleanup fences every reservation owner and digest", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.UPSTASH_REDIS_REST_URL = "https://redis.test";
  process.env.UPSTASH_REDIS_REST_TOKEN = "redis-token";
  const records = [{ identity: JSON.stringify(["org/model", "one", false, true]), digest: "a".repeat(64) }];
  const commands = [];
  globalThis.fetch = async (_input, init) => {
    const command = JSON.parse(init.body); commands.push(command);
    return Response.json({ result: command[0] === "SMEMBERS" ? [JSON.stringify(records[0])] : 1 });
  };
  const lease = { issue_number: 12, language: "da", lease_id: "lease", expires_at: "2020-01-01T00:00:00.000Z" };
  try {
    assert.equal(await reclaimExpiredLease(lease), true);
    const command = commands[1];
    assert.equal(command[0], "EVAL");
    assert.match(command[1], /item\.lease_id ~= ARGV\[1\]/);
    assert.match(command[1], /owned\.digest == record\.digest/);
    assert.match(command[1], /owned\.identity == record\.identity/);
    assert.match(command[1], /DEL',KEYS\[3\]/);
    assert.equal(command[2], "4");
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

test("promotion renewal refreshes each nonterminal path only", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.UPSTASH_REDIS_REST_URL = "https://redis.test";
  process.env.UPSTASH_REDIS_REST_TOKEN = "redis-token";
  const records = [
    { identity: JSON.stringify(["org/model", "one", false, true]), canonical_path: "org_model/one__test__fewshot.json", digest: "a".repeat(64) },
    { identity: JSON.stringify(["org/model", "two", false, true]), canonical_path: "org_model/two__test__fewshot.json", digest: "b".repeat(64) },
  ];
  const commands = [];
  globalThis.fetch = async (_input, init) => {
    commands.push(JSON.parse(init.body));
    return Response.json({ result: "reserved" });
  };
  try {
    assert.equal(await reservePromotionReservation({ issue_number: 12, submission_id: "one", outcome: "accepted",
      records, token: "token", decision_nonce: "nonce", status: "reserved" }), "reserved");
    const command = commands[0];
    assert.equal(command[0], "EVAL");
    assert.equal(command[2], "3");
    assert.match(command[1], /redis\.call\('EXPIRE',KEYS\[i\],ARGV\[4\]\)/);
    assert.match(command[1], /item\.status ~= 'terminal'/);
    assert.equal(command[3], "euroeval:worker:promotion:12:one");
    assert.equal(command[4], await promotionIdentityKey(records[0].canonical_path));
    assert.equal(command[5], await promotionIdentityKey(records[1].canonical_path));
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

test("retries atomic multi-record reservation cleanup", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.UPSTASH_REDIS_REST_URL = "https://redis.test";
  process.env.UPSTASH_REDIS_REST_TOKEN = "redis-token";
  const records = [{ identity: JSON.stringify(["org/model", "one", false, true]), digest: "a".repeat(64) },
    { identity: JSON.stringify(["org/model", "two", false, true]), digest: "b".repeat(64) }];
  const commands = [];
  let attempts = 0;
  globalThis.fetch = async (_input, init) => {
    attempts += 1;
    commands.push(JSON.parse(init.body));
    if (attempts === 1) throw new Error("simulated interruption");
    return Response.json({ result: 1 });
  };
  try {
    assert.equal(await releaseResultReservations(records, "submission"), true);
    assert.equal(attempts, 2);
    assert.equal(commands[1][0], "EVAL");
    assert.match(commands[1][1], /for i=1,#KEYS-1/);
    assert.equal(commands[1][2], "3");
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});
