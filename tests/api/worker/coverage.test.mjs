import test from "node:test";
import assert from "node:assert/strict";
import claim from "../../../api/worker/claim.ts";
import { sha256, parseVolunteerMarker } from "../../../api/worker/_lib.ts";
import { narrowCoverage } from "../../../api/worker/_lib/coverage.ts";
import { validateResultScope } from "../../../api/worker/finalise.ts";

const suffix = '["dataset",true,true]';
const scope = { allowed_identity_suffix_sets: [[suffix, '["missing",true,true]']] };

function result({ revision = "rev", version = "1.0.0", dataset = "dataset" } = {}) {
  const modelInfo = { id: "org/model", revision };
  if (revision === null) delete modelInfo.revision;
  return Response.json({
    schema_version: "0.3.0",
    evaluation_results: [],
    model_info: modelInfo,
    eval_library: { name: "euroeval", version,
      additional_details: { dataset, validation_split: true, few_shot: true } },
  });
}

test("canonical coverage resolves bucket files through the Hugging Face bucket route", async () => {
  let requested;
  await narrowCoverage({ allowed_identity_suffix_sets: [[suffix]] }, "org/model", "rev", "1.0.0", async (input, options) => {
    requested = { url: String(input), options };
    return result();
  });
  assert.equal(requested.url,
    "https://huggingface.co/buckets/EuroEval/results/resolve/org_model/dataset__val__fewshot.json");
  assert.equal(requested.options.headers.Range, "bytes=0-1048575");
});

test("canonical coverage accepts exact revision and same or newer EuroEval versions", async () => {
  for (const version of ["1.0.0", "1.1.0", "1.0.1", "1.0.0.dev2"]) {
    const coverage = await narrowCoverage(scope, "org/model", "rev", "1.0.0.dev0", async () => result({ version }));
    assert.deepEqual(coverage.allowed_identity_suffix_sets, [['["missing",true,true]']]);
  }
});

test("canonical coverage reports an all-covered scope without remaining work", async () => {
  const completeScope = { allowed_identity_suffix_sets: [[suffix]] };
  const coverage = await narrowCoverage(completeScope, "org/model", "rev", "1.0.0", async () => result());
  assert.deepEqual(coverage.allowed_identity_suffix_sets, []);
  assert.deepEqual(coverage.covered_identity_suffixes, [suffix]);

  const alternativeScope = { allowed_identity_suffix_sets: [[suffix], ['["uncovered",true,true]']] };
  const alternatives = await narrowCoverage(
    alternativeScope, "org/model", "rev", "1.0.0", async (input) =>
      String(input).includes("dataset__") ? result() : new Response(null, { status: 404 }),
  );
  assert.deepEqual(alternatives.allowed_identity_suffix_sets, []);
});

test("canonical coverage does not count older, wrong-revision, or missing-revision records", async () => {
  for (const options of [
    { revision: "older-revision" },
    { revision: null },
    { version: "0.9.9" },
  ]) {
    const coverage = await narrowCoverage(scope, "org/model", "rev", "1.0.0.dev0", async () => result(options));
    assert.deepEqual(coverage.allowed_identity_suffix_sets, [[suffix, '["missing",true,true]']]);
  }
});

test("canonical coverage treats only an actual 404 as absent and fails closed otherwise", async () => {
  const missing = await narrowCoverage(scope, "org/model", "rev", "1.0.0", async () =>
    new Response(null, { status: 404 }));
  assert.deepEqual(missing.allowed_identity_suffix_sets, scope.allowed_identity_suffix_sets);
  await assert.rejects(
    narrowCoverage(scope, "org/model", "rev", "1.0.0", async () => new Response(null, { status: 503 })),
    /coverage is unavailable/,
  );
  await assert.rejects(
    narrowCoverage(scope, "org/model", "rev", "1.0.0", async () => new Response("not json")),
    /malformed/,
  );
  await assert.rejects(
    narrowCoverage(scope, "org/model", "rev", "1.0.0", async () =>
      Response.json({ schema_version: "bad" })),
    /malformed/,
  );
  await assert.rejects(
    narrowCoverage(scope, "org/model", "rev", "1.0.0", async () =>
      new Response("x".repeat(1_048_577))),
    /exceeds the coverage limit/,
  );
});

test("canonical coverage bounds a stalled bucket request", async () => {
  const started = Date.now();
  await assert.rejects(
    narrowCoverage(scope, "org/model", "rev", "1.0.0", () => new Promise(() => {})),
    /timed out/,
  );
  assert.ok(Date.now() - started < 5_000);
});

test("finalisation accepts only the narrowed subset of the official scope", () => {
  const originalEnv = process.env.VOLUNTEER_SCOPE_POLICY_JSON;
  process.env.VOLUNTEER_SCOPE_POLICY_JSON = JSON.stringify({ policy_version: "policy",
    policies: [{ euroeval_version: "1.0.0", model_type: "encoder", language: "da",
      language_group: "Danish", allowed_identity_suffix_sets: [[suffix, '["missing",true,true]']],
      task_groups: ["classification"], warnings: [] }] });
  const lease = { euroeval_version: "1.0.0", model_type: "encoder", language: "da",
    expected_scope: { policy_version: "policy", language_group: "Danish",
      allowed_identity_suffix_sets: [['["missing",true,true]']],
      original_identity_suffix_sets: [[suffix, '["missing",true,true]']],
      covered_identity_suffixes: [suffix], task_groups: ["classification"], warnings: [] } };
  try {
    const entry = { identity: JSON.stringify(["org/model", "missing", true, true]) };
    assert.deepEqual(validateResultScope([entry], lease), ['["missing",true,true]']);
    assert.throws(() => validateResultScope([
      entry, { identity: JSON.stringify(["org/model", "dataset", true, true]) },
    ], lease), /expected scope/);
  } finally {
    if (originalEnv === undefined) delete process.env.VOLUNTEER_SCOPE_POLICY_JSON;
    else process.env.VOLUNTEER_SCOPE_POLICY_JSON = originalEnv;
  }
});

test("canonical coverage rejects mismatched canonical record identities", async () => {
  const coverage = await narrowCoverage(scope, "org/model", "rev", "1.0.0", async () =>
    result({ dataset: "different" }));
  assert.deepEqual(coverage.allowed_identity_suffix_sets, scope.allowed_identity_suffix_sets);
});

for (const [coverage, capabilities, expectedStatus] of [
  ["partial", undefined, 426],
  ["partial", ["unknown"], 426],
  ["partial", "canonical_coverage_v1", 426],
  ["partial", ["canonical_coverage_v1"], 200],
  ["none", undefined, 200],
  ["all", undefined, 200],
]) test(`claim coverage=${coverage} capabilities=${JSON.stringify(capabilities)}`, async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  const suffixes = ['["dataset",true,true]', '["missing",true,true]'];
  Object.assign(process.env, { VOLUNTEER_MARKER_SECRET: "marker-secret",
    VOLUNTEER_WORKER_IMAGE_DIGEST: "sha256:image", EUROEVAL_VERSION: "1.0.0",
    VOLUNTEER_WORKER_VERSION: "1.0.0", UPSTASH_REDIS_REST_URL: "https://redis.test",
    UPSTASH_REDIS_REST_TOKEN: "redis-token", GITHUB_TOKEN: "github-token",
    VOLUNTEER_SCOPE_POLICY_JSON: JSON.stringify({ policy_version: "test-policy", policies: [{
      euroeval_version: "1.0.0", model_type: "encoder", language: "da", language_group: "da",
      allowed_identity_suffix_sets: [suffixes],
      task_groups: ["sequence_classification"], warnings: [],
    }] }) });
  const language = "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)";
  const issue = { number: 12, title: "[MODEL EVALUATION REQUEST] org/model",
    body: `### Model ID\n\norg/model\n\n- [x] ${language}\n`, state: "open", assignees: [] };
  const credentialKey = `euroeval:worker:credential:${await sha256("credential")}`;
  const values = new Map([[credentialKey, JSON.stringify({ contributor: "alice" })]]);
  const reservations = [];
  const mutations = [];
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input); const method = init.method || "GET";
    if (url.includes("/buckets/EuroEval/results/resolve/")) {
      const dataset = url.includes("/dataset__") ? "dataset" : "missing";
      if (coverage === "none" || coverage === "partial" && dataset === "missing") {
        return new Response(null, { status: 404 });
      }
      return result({ dataset, revision: "a".repeat(40) });
    }
    if (url === "https://redis.test") {
      const command = JSON.parse(init.body);
      if (command[0] === "GET") return Response.json({ result: values.get(command[1]) || null });
      if (command[0] === "SMEMBERS") return Response.json({ result: [] });
      if (command[0] === "SET") { values.set(command[1], command[2]); return Response.json({ result: "OK" }); }
      if (command[0] === "EVAL" && command[1].includes("local issue=redis.call")) {
        reservations.push(command);
        values.set(command[3], command[5]); values.set(command[4], command[5]);
      }
      return Response.json({ result: 1 });
    }
    if (url.includes("/repos/EuroEval/EuroEval/assignees/")) return new Response(null, { status: 204 });
    if (url.endsWith("/issues/12/assignees") && method === "POST") {
      issue.assignees = [{ login: "alice" }]; return Response.json(issue);
    }
    if (url.includes("/repos/EuroEval/EuroEval/issues?")) return Response.json([issue]);
    if (url.includes("huggingface.co/api/models/org/model")) return Response.json({ id: "org/model",
      sha: "a".repeat(40), private: false, gated: false, pipeline_tag: "fill-mask",
      siblings: [{ rfilename: "model.safetensors", size: 100 }, { rfilename: "config.json", size: 10 }] });
    if (url.includes(`/org/model/raw/${"a".repeat(40)}/config.json`)) {
      return Response.json({ model_type: "bert", architectures: ["BertModel"] });
    }
    if (url.endsWith("/issues/12") && method === "GET") return Response.json(issue);
    if (url.endsWith("/issues/12") && method === "PATCH") {
      mutations.push(JSON.parse(init.body)); Object.assign(issue, mutations.at(-1)); return Response.json(issue);
    }
    throw new Error(`Unexpected request: ${method} ${url}`);
  };
  try {
    const response = await claim(new Request("https://euroeval.test/api/worker/claim", {
      method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", worker_version: "1.0.0", capabilities,
        hardware: { architecture: "amd64", free_disk_bytes: 1_000_000, gpu_memory_utilisation: 0.8,
          selected_gpu_index: 0, selected_gpu_uuid: "gpu", gpus: [{ index: 0, name: "GPU", uuid: "gpu",
            free_memory_bytes: 1_000_000, total_memory_bytes: 1_000_000 }] } }),
    }));
    const body = await response.json();
    assert.equal(response.status, expectedStatus, JSON.stringify(body));
    if (expectedStatus === 426) {
      assert.equal(body.code, "worker_update_required");
      assert.match(body.error, /update.*worker image/i);
    }
    if (expectedStatus === 426 || coverage === "all") {
      assert.deepEqual(reservations, []);
      assert.deepEqual(mutations, []);
      assert.equal(parseVolunteerMarker(issue.body), null);
      if (coverage === "all") assert.equal(body.status, "no_work");
    } else {
      assert.equal(reservations.length, 1);
      assert.deepEqual(body.expected_scope.allowed_identity_suffix_sets,
        [coverage === "partial" ? [suffixes[1]] : suffixes]);
      assert.equal(body.contamination_canary.status, "required");
    }
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});
