import test from "node:test";
import assert from "node:assert/strict";
import {
  extractModelId,
  parseVolunteerMarker,
  replaceVolunteerMarker,
  selectedLanguages,
  validateRecord,
} from "./_lib.ts";

test("parses the queue model and language checkboxes", () => {
  const body = "### Model ID\n\norg/model\n\n- [x] Greek\n- [ ] Albanian\n";
  assert.equal(extractModelId("ignored", body), "org/model");
  assert.deepEqual(selectedLanguages(body), ["el"]);
});

test("malformed broker markers are ignored and never trusted", () => {
  assert.equal(parseVolunteerMarker("<!-- euroeval-volunteer-worker:v1 not-json -->"), null);
  const marker = { winner: "contributor", leases: [{ language: "da", worker: "abc", contributor: "contributor", expires_at: Date.now() + 1000 }] };
  const body = replaceVolunteerMarker("queue", marker);
  assert.deepEqual(parseVolunteerMarker(body), marker);
});

test("validates canonical model identity and score bounds", () => {
  const record = {
    schema_version: "0.2.1",
    model_info: { id: "org/model@deadbeef" },
    eval_library: { additional_details: { dataset: "task", language: "da", few_shot: false, validation_split: false, num_failed_instances: 0 } },
    evaluation_results: [{ score_details: { score: 85 } }],
  };
  assert.equal(validateRecord(record, { modelId: "org/model", revision: "deadbeef", language: "da" }).failed, 0);
  assert.throws(() => validateRecord({ ...record, evaluation_results: [{ score_details: { score: 101 } }] }, { modelId: "org/model", revision: "deadbeef", language: "da" }));
  assert.throws(() => validateRecord({ ...record, evaluation_results: [{ score_details: { score: 85, details: { num_failed_instances: "1.0" } } }] }, { modelId: "org/model", revision: "deadbeef", language: "da" }));
});
