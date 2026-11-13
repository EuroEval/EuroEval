import test from "node:test";
import assert from "node:assert/strict";
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
