import test from "node:test";
import assert from "node:assert/strict";
import { assertAssignable, expectedScope, extractModelId, parsePromotionRecords, parseVolunteerMarker, PROMOTION_RESERVATION_TTL, renderVolunteerMarker, replaceVolunteerMarker, requireAssignee, selectedLanguages, validateRecord, volunteerAssigneesMatch } from "../../../api/worker/_lib.ts";
import { fitsGpu, resolveModel, selectedGpu } from "../../../api/worker/_lib/model.ts";
import { putLease, reclaimExpiredLease, releaseResultReservations, reserveResultIdentity } from "../../../api/worker/_lib/redis.ts";
import { bindPromotionDecision, promotionIdentityKey, reservePromotionReservation } from "../../../api/worker/_lib/promotion.ts";

test("resolves immutable Hub capability evidence without suffix heuristics", async () => {
  const originalFetch = globalThis.fetch;
  const revision = "b".repeat(40);
  globalThis.fetch = async (input) => {
    const url = String(input);
    if (url.includes("/api/models/org/base")) {
      return Response.json({ id: "org/base", sha: revision, private: false, gated: false,
        pipeline_tag: "fill-mask", siblings: [{ rfilename: "model.safetensors", size: 10 }] });
    }
    if (url.includes(`/org/base/raw/${revision}/config.json`)) {
      return Response.json({ model_type: "new_encoder", architectures: ["NovelBaseModel"] });
    }
    throw new Error(`unexpected URL ${url}`);
  };
  try {
    const model = await resolveModel("org/base");
    assert.equal(model.model_type, "encoder");
    assert.deepEqual(model.model_metadata, {
      pipeline_tag: "fill-mask", architectures: ["NovelBaseModel"],
      model_type: "encoder", is_encoder_decoder: null,
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("rejects missing or contradictory immutable capability metadata", async () => {
  const originalFetch = globalThis.fetch;
  const revision = "c".repeat(40);
  globalThis.fetch = async (input) => {
    const url = String(input);
    const modelId = url.includes("org/missing") ? "org/missing" : "org/contradictory";
    return url.includes("/api/models/")
      ? Response.json({ id: modelId, sha: revision, private: false, gated: false,
        ...(modelId === "org/contradictory" ? { pipeline_tag: "fill-mask" } : {}),
        siblings: [{ rfilename: "model.safetensors", size: 10 }] })
      : Response.json({ model_type: "seq2seq", architectures: ["InventedModel"], is_encoder_decoder: true });
  };
  try {
    await assert.rejects(resolveModel("org/missing"), /pipeline tag/);
    await assert.rejects(resolveModel("org/contradictory"), /contradictory/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("rejects Hub repositories that require custom code", async () => {
  const originalFetch = globalThis.fetch;
  const revision = "d".repeat(40);
  globalThis.fetch = async (input) => String(input).includes("/api/models/")
    ? Response.json({ id: "org/custom", sha: revision, private: false, gated: false,
      pipeline_tag: "fill-mask", siblings: [{ rfilename: "model.safetensors", size: 10 },
        { rfilename: "modeling_custom.py", size: 10 }] })
    : Response.json({ model_type: "custom", architectures: ["CustomModel"], auto_map: {} });
  try {
    await assert.rejects(resolveModel("org/custom"), /custom repository Python/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("parses the queue model and language checkboxes", () => {
  const body = "### Model ID\n\norg/model\n\n- [x] Greek\n- [ ] Albanian\n";
  assert.equal(extractModelId("ignored", body), "org/model");
  assert.deepEqual(selectedLanguages(body), ["el"]);
});

test("unassignable contributors receive a stable actionable error", async () => {
  const originalFetch = globalThis.fetch;
  const originalToken = process.env.GITHUB_TOKEN;
  process.env.GITHUB_TOKEN = "github-token";
  globalThis.fetch = async () => new Response(null, { status: 404 });
  try {
    await assert.rejects(assertAssignable("alice"), (error) =>
      error.status === 422 && error.code === "github_login_not_assignable" &&
      /cannot be assigned/.test(error.message));
  } finally {
    globalThis.fetch = originalFetch;
    if (originalToken === undefined) delete process.env.GITHUB_TOKEN;
    else process.env.GITHUB_TOKEN = originalToken;
  }
});

test("assignee ownership is case-insensitive and excludes manual identities", () => {
  const marker = { protocol_version: "volunteer-worker/v1", coordinator: "sentinel", submission: "active", leases: [{
    lease_id: "lease", language: "da", worker: "worker", contributor: "Alice",
    expires_at: new Date(Date.now() + 60_000).toISOString(),
  }] };
  assert.equal(volunteerAssigneesMatch([{ login: "alice" }], marker), true);
  assert.equal(volunteerAssigneesMatch([{ login: "alice" }, { login: "maintainer" }], marker), false);
  assert.equal(volunteerAssigneesMatch([{ login: "maintainer" }], null), false);
  assert.equal(volunteerAssigneesMatch([], null), true);
});

test("expired lease contributors remain attributable only during recovery", () => {
  const marker = { protocol_version: "volunteer-worker/v1", coordinator: "sentinel", submission: "active", leases: [{
    lease_id: "expired", language: "da", worker: "worker", contributor: "Alice",
    expires_at: "2020-01-01T00:00:00Z",
  }] };
  assert.equal(volunteerAssigneesMatch([{ login: "alice" }], marker), false);
  assert.equal(volunteerAssigneesMatch([{ login: "alice" }], marker, Date.now(), true), true);
  assert.equal(volunteerAssigneesMatch([{ login: "manual" }], marker, Date.now(), true), false);
});

test("assignment loss uses the stable lease error contract", async () => {
  await assert.rejects(requireAssignee({ assignees: [] }, "alice"), (error) =>
    error.status === 409 && error.code === "lease_assignment_lost" &&
    error.message === "The lease contributor is no longer assigned to this issue.");
});

test("strictly parses the shared ownership marker", () => {
  assert.equal(parseVolunteerMarker("<!-- euroeval-volunteer-worker:v1 not-json -->"), null);
  const marker = { protocol_version: "volunteer-worker/v1", coordinator: "coordinator", submission: "active", leases: [{ lease_id: "lease", language: "da", worker: "abc", contributor: "contributor", expires_at: new Date(Date.now() + 1000).toISOString() }] };
  const body = replaceVolunteerMarker("queue", marker);
  assert.deepEqual(parseVolunteerMarker(body), marker);
});

test("marker replacement preserves comments and selected checkboxes", () => {
  const marker = { protocol_version: "volunteer-worker/v1", coordinator: "coordinator", submission: "active", leases: [] };
  const body = "<!-- harmless comment -->\n- [x] Greek\n\n" + renderVolunteerMarker(marker);
  const replaced = replaceVolunteerMarker(body, null);
  assert.match(replaced, /harmless comment/);
  assert.match(replaced, /- \[x\] Greek/);
  assert.doesNotMatch(replaced, /euroeval-volunteer-worker:v1/);
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
  const scope = expectedScope("18.0.0.dev", "encoder", "da");
  assert.equal(scope.language, "da");
  assert.equal(scope.policy_version, "volunteer-scope/18.0.0.dev0");
  assert.ok(scope.identity_suffixes.length > 0);
});

test("trusted exact-language policy owns the lease language group", () => {
  const original = process.env.VOLUNTEER_SCOPE_POLICY_JSON;
  process.env.VOLUNTEER_SCOPE_POLICY_JSON = JSON.stringify({ policy_version: "test-policy", policies: [{
    euroeval_version: "1.0.0", model_type: "encoder", language: "da", language_group: "policy-da",
    identity_suffixes: [JSON.stringify(["dataset", false, true])],
    task_groups: ["sequence_classification"],
  }] });
  try {
    const scope = expectedScope("1.0.0", "encoder", "da");
    assert.equal(scope.language, "da");
    assert.equal(scope.language_group, "policy-da");
  } finally {
    if (original === undefined) delete process.env.VOLUNTEER_SCOPE_POLICY_JSON;
    else process.env.VOLUNTEER_SCOPE_POLICY_JSON = original;
  }
});

test("fits only the explicitly selected GPU", () => {
  const model = { id: "org/model", revision: "r", config: {}, weight_bytes: 300, repo_bytes: 500, model_type: "generative" };
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
  const model = { id: "org/model", revision: "r", config: {}, weight_bytes: 100, repo_bytes: 500, model_type: "generative" };
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
      records, token: "token", decision_reviewer: "alice",
      decision_created_at: "2026-09-06T12:00:00Z", status: "reserved" }), "reserved");
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

test("decision binding is token-fenced and set once", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.UPSTASH_REDIS_REST_URL = "https://redis.test";
  process.env.UPSTASH_REDIS_REST_TOKEN = "redis-token";
  const commands = [];
  globalThis.fetch = async (_input, init) => {
    commands.push(JSON.parse(init.body));
    return Response.json({ result: commands.length === 1 ? "bound" : "mismatch" });
  };
  const reservation = { issue_number: 12, submission_id: "one", outcome: "rejected", records: [],
    token: "token", decision_reviewer: "alice", decision_created_at: "2026-09-06T12:00:00Z", status: "reserved" };
  try {
    assert.equal(await bindPromotionDecision(reservation, "a".repeat(64)), "bound");
    assert.equal(await bindPromotionDecision(reservation, "b".repeat(64)), "mismatch");
    assert.match(commands[0][1], /item\.decision_digest/);
    assert.match(commands[0][1], /item\.decision_digest=ARGV\[2\]/);
    assert.equal(commands[0][3], "euroeval:worker:promotion:12:one");
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

test("terminal decision binding is idempotent without a write", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.UPSTASH_REDIS_REST_URL = "https://redis.test";
  process.env.UPSTASH_REDIS_REST_TOKEN = "redis-token";
  const commands = [];
  globalThis.fetch = async (_input, init) => {
    commands.push(JSON.parse(init.body));
    return Response.json({ result: "bound" });
  };
  const reservation = { issue_number: 12, submission_id: "one", outcome: "rejected", records: [],
    token: "token", decision_reviewer: "alice", decision_created_at: "2026-09-06T12:00:00Z",
    decision_digest: "a".repeat(64), status: "terminal" };
  try {
    assert.equal(await bindPromotionDecision(reservation, reservation.decision_digest), "bound");
    const script = commands[0][1];
    assert.ok(script.indexOf("return 'bound'") < script.indexOf("item.decision_digest=ARGV[2]"));
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

const putLeaseFixture = (overrides = {}) => ({
  issue_number: 12, language: "da", worker: "worker", contributor: "alice",
  model_id: "org/model", model_revision: "revision", euroeval_version: "1.0.0",
  image_digest: "sha256:image", worker_version: "worker-v1", gpu_memory_utilisation: 0.8,
  selected_gpu_index: 0, selected_gpu_uuid: "GPU-0",
  expires_at: new Date(Date.now() + 60_000).toISOString(), lease_id: "lease-id",
  model_type: "generative", expected_scope: {
    policy_version: "policy-v1", language_group: "da", identity_suffixes: [], count: 0,
    task_groups: ["text_to_text"], warnings: [],
  }, ...overrides,
});

const emulatePutLeaseEval = (command, state) => {
  assert.equal(command[0], "EVAL");
  assert.match(command[1], /local issue=redis\.call\('GET',KEYS\[1\]\)/);
  assert.match(command[1], /issue ~= ARGV\[1\]/);
  assert.equal(command[2], "2");
  assert.ok(Number(command[6]) >= 30 * 24 * 60 * 60 + 60);
  const [issueKey, leaseKey, payload, , leaseId] = command.slice(3);
  const issue = state.get(issueKey);
  const byId = state.get(leaseKey);
  if (issue !== undefined || byId !== undefined) {
    if (issue === undefined || byId === undefined || issue !== payload || byId !== payload) return 0;
    return JSON.parse(issue).lease_id === leaseId ? 1 : 0;
  }
  state.set(issueKey, payload);
  state.set(leaseKey, payload);
  return 1;
};

test("claim lease retry succeeds after committed EVAL response loss", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.UPSTASH_REDIS_REST_URL = "https://redis.test";
  process.env.UPSTASH_REDIS_REST_TOKEN = "redis-token";
  const state = new Map();
  const lease = putLeaseFixture();
  let attempts = 0;
  globalThis.fetch = async (_input, init) => {
    const command = JSON.parse(init.body);
    attempts += 1;
    const result = emulatePutLeaseEval(command, state);
    if (attempts === 1) throw new Error("response lost after commit");
    return Response.json({ result });
  };
  try {
    assert.equal(await putLease(lease), true);
    assert.equal(attempts, 2);
    assert.equal(state.size, 2);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

test("claim lease retry fails closed for split or mismatched state", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.UPSTASH_REDIS_REST_URL = "https://redis.test";
  process.env.UPSTASH_REDIS_REST_TOKEN = "redis-token";
  const lease = putLeaseFixture();
  const payload = JSON.stringify(lease);
  const issueKey = "euroeval:worker:lease:12:da";
  const byIdKey = "euroeval:worker:lease-id:lease-id";
  try {
    for (const state of [
      new Map([[issueKey, payload]]),
      new Map([[issueKey, payload], [byIdKey, JSON.stringify({ ...lease, contributor: "mallory" })]]),
    ]) {
      globalThis.fetch = async (_input, init) => Response.json({ result: emulatePutLeaseEval(JSON.parse(init.body), state) });
      assert.equal(await putLease(lease), false);
    }
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});
