import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import {
  decodeGgufArtifact,
  decodeGgufBindings,
  decodeLeaseExecution,
  findApprovedGgufArtifact,
  ggufBindings,
} from "../../../api/worker/_lib/gguf.ts";

const PINNED = {
  repo_id: "LiquidAI/LFM3-1.7B-GGUF",
  revision: "cd0c74c575b5c416f3011b5b4c298077f26ca8c2",
  filename: "LFM3-1.7B-Q8_0.gguf",
  sha256: "8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6",
  source_repo_id: "LiquidAI/LFM3-1.7B",
  source_revision: "1da06bec6a819da10b54c67e5c90972b46d81871",
};

const bindingsPath = fileURLToPath(new URL("../../../api/worker/gguf-bindings.json", import.meta.url));
const bindingsRaw = JSON.parse(await readFile(bindingsPath, "utf8"));

test("checked-in bindings decode to the pinned contract artifact", () => {
  const bindings = decodeGgufBindings(bindingsRaw);
  assert.equal(bindings.version, 1);
  assert.deepEqual(bindings.artifacts, [PINNED]);
});

test("the broker's runtime bindings deep-equal the checked-in gguf-bindings.json", () => {
  // Edge endpoints cannot read files, so gguf.ts embeds the binding document;
  // the checked-in JSON stays the reviewed source of truth via this sync check.
  assert.deepEqual(ggufBindings(), decodeGgufBindings(bindingsRaw));
});

test("approval lookup accepts only the exact approved artifact", () => {
  const bindings = decodeGgufBindings(bindingsRaw);
  const approved = findApprovedGgufArtifact(bindings, decodeGgufArtifact(PINNED));
  assert.deepEqual(approved, PINNED);
  for (const [key, value] of [
    ["repo_id", "LiquidAI/Other-GGUF"],
    ["revision", "b".repeat(40)],
    ["filename", "LFM3-1.7B-Q4_0.gguf"],
    ["sha256", "c".repeat(64)],
    ["source_repo_id", "LiquidAI/Other"],
    ["source_revision", "d".repeat(40)],
  ]) {
    const claim = { ...PINNED, [key]: value };
    assert.equal(findApprovedGgufArtifact(bindings, decodeGgufArtifact(claim)), null, key);
  }
  assert.equal(
    findApprovedGgufArtifact(bindings, decodeGgufArtifact({ ...PINNED, repo_id: "org/unapproved" })),
    null,
  );
});

test("decodeGgufArtifact rejects unsafe filenames", () => {
  for (const filename of [
    "../LFM3-1.7B-Q8_0.gguf",
    "nested/LFM3-1.7B-Q8_0.gguf",
    "back\\slash.gguf",
    "LFM3-1.7B-Q8_0.GGUF",
    "LFM3-1.7B-Q8_0.gguf.bak",
    "no-extension",
    "",
    ".",
    "..",
    ".gguf",
    "spaced name.gguf",
    "tab\tname.gguf",
  ]) {
    assert.throws(() => decodeGgufArtifact({ ...PINNED, filename }), /filename/, filename);
  }
});

test("decodeGgufArtifact rejects nonhex revisions/hashes and malformed repo ids", () => {
  for (const [key, value] of [
    ["revision", "c".repeat(39)],
    ["revision", "c".repeat(41)],
    ["revision", "g".repeat(40)],
    ["revision", "C".repeat(40)],
    ["sha256", "8".repeat(63)],
    ["sha256", "8".repeat(65)],
    ["sha256", "z".repeat(64)],
    ["sha256", "8F6B24182258205E9E01E21F1F24A1F208C1E8115578825D202CE39976AF20B6"],
    ["source_revision", "not-hex"],
    ["repo_id", "NoSlash"],
    ["repo_id", "too/many/slashes"],
    ["repo_id", ""],
    ["repo_id", "org/ model"],
    ["source_repo_id", "/leading"],
    ["filename", 123],
    ["revision", null],
  ]) {
    assert.throws(() => decodeGgufArtifact({ ...PINNED, [key]: value }), undefined, key);
  }
});

test("decodeGgufArtifact rejects non-objects, missing and extra keys", () => {
  assert.throws(() => decodeGgufArtifact(null));
  assert.throws(() => decodeGgufArtifact("artifact"));
  const missing = { ...PINNED };
  delete missing.sha256;
  assert.throws(() => decodeGgufArtifact(missing), /artifact/);
  assert.throws(() => decodeGgufArtifact({ ...PINNED, quantization: "Q8_0" }), /artifact/);
});

test("decodeLeaseExecution maps absence to CUDA and rejects unknown kinds", () => {
  assert.equal(decodeLeaseExecution(undefined), null);
  assert.equal(decodeLeaseExecution(null), null);
  const execution = decodeLeaseExecution({ kind: "external_server", artifact: PINNED });
  assert.deepEqual(execution, { kind: "external_server", artifact: PINNED });
  assert.throws(() => decodeLeaseExecution({ kind: "cuda_server", artifact: PINNED }), /kind/);
  assert.throws(() => decodeLeaseExecution({ artifact: PINNED }), /kind/);
  assert.throws(() => decodeLeaseExecution("external_server"));
  assert.throws(() => decodeLeaseExecution({ kind: "external_server" }));
  assert.throws(
    () => decodeLeaseExecution({ kind: "external_server", artifact: PINNED, extra: true }),
  );
  assert.throws(
    () => decodeLeaseExecution({ kind: "external_server", artifact: { ...PINNED, filename: "../evil.gguf" } }),
    /filename/,
  );
});

test("decodeGgufBindings rejects malformed envelopes and duplicate bindings", () => {
  assert.throws(() => decodeGgufBindings(null));
  assert.throws(() => decodeGgufBindings({ version: 2, artifacts: [PINNED] }), /version/);
  assert.throws(() => decodeGgufBindings({ version: 1, artifacts: [] }), /artifacts/);
  assert.throws(() => decodeGgufBindings({ version: 1, artifacts: [PINNED, PINNED] }), /duplicate/);
  assert.throws(() => decodeGgufBindings({ version: 1, artifacts: [PINNED, { ...PINNED, filename: "other.gguf" }] }), /duplicate/);
  assert.throws(
    () => decodeGgufBindings({ version: 1, artifacts: [{ ...PINNED, sha256: "z".repeat(64) }] }),
  );
  assert.throws(() => decodeGgufBindings({ version: 1, artifacts: [PINNED], extra: 1 }));
});

test("external instruction scope stays incomplete when only base identities are covered", async () => {
  const { expectedScope } = await import("../../../api/worker/_lib/model.ts");
  const { narrowCoverage } = await import("../../../api/worker/_lib/coverage.ts");
  const legacy = expectedScope("18.4.0.dev0", "generative", "en");
  const instruction = expectedScope("18.4.0.dev0", "generative", "en", true);
  const base = legacy.allowed_identity_suffix_sets.reduce((a, b) => a.length < b.length ? a : b);
  assert.equal(instruction.allowed_identity_suffix_sets.length, 1);
  assert.ok(instruction.allowed_identity_suffix_sets[0].length > base.length);
  const coveredFiles = new Map(base.map((suffix) => {
    const [dataset, split, shot] = JSON.parse(suffix);
    const file = `${dataset.replaceAll("/", "_")}__${split === true ? "val" : split === false ? "test" : "none"}__${shot === true ? "fewshot" : shot === false ? "zeroshot" : "none"}.json`;
    return [file, { dataset, validation_split: split, few_shot: shot }];
  }));
  const fetcher = async (input) => {
    const details = coveredFiles.get(decodeURIComponent(String(input).split("/").at(-1)));
    if (!details) return new Response(null, { status: 404 });
    return Response.json({ schema_version: "0.3.0", evaluation_results: [],
      model_info: { id: PINNED.repo_id, revision: PINNED.revision },
      eval_library: { name: "euroeval", version: "18.4.0.dev0", additional_details: details } });
  };
  const covered = await narrowCoverage(instruction, PINNED.repo_id, PINNED.revision, "18.4.0.dev0", fetcher);
  assert.equal(covered.allowed_identity_suffix_sets.length, 1);
  assert.equal(covered.allowed_identity_suffix_sets[0].length,
    instruction.allowed_identity_suffix_sets[0].length - base.length);
  const old = await narrowCoverage(legacy, PINNED.repo_id, PINNED.revision, "18.4.0.dev0", fetcher);
  assert.deepEqual(old.allowed_identity_suffix_sets, []);
});
