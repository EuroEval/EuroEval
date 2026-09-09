import test from "node:test";
import assert from "node:assert/strict";
import { expectedScope, extractModelId, fitsGpu, parseVolunteerMarker, replaceVolunteerMarker, selectedLanguages, validateRecord } from "./_lib.ts";

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

test("generated trusted scopes are exact-language and versioned", () => {
  const scope = expectedScope("18.0.0.dev", "bert", "da");
  assert.equal(scope.language, "da");
  assert.equal(scope.policy_version, "volunteer-scope/18.0.0.dev");
  assert.ok(scope.identity_suffixes.length > 0);
});

test("fits a model on one reported GPU and requires repository disk", () => {
  const model = { id: "org/model", revision: "r", config: {}, weight_bytes: 100, repo_bytes: 500, model_profile: "llama" };
  assert.equal(fitsGpu(model, { free_disk_bytes: 500, gpus: [{ name: "a", uuid: "1", free_memory_bytes: 50, total_memory_bytes: 50 }, { name: "b", uuid: "2", free_memory_bytes: 135, total_memory_bytes: 135 }] }), true);
  assert.equal(fitsGpu(model, { free_disk_bytes: 499, gpus: [{ name: "a", uuid: "1", free_memory_bytes: 135, total_memory_bytes: 135 }] }), false);
});
