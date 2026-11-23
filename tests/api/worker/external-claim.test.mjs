import test from "node:test";
import assert from "node:assert/strict";
import claim from "../../../api/worker/claim.ts";
import { sha256 } from "../../../api/worker/_lib.ts";

const PINNED = {
  repo_id: "LiquidAI/LFM3-1.7B-GGUF",
  revision: "cd0c74c575b5c416f3011b5b4c298077f26ca8c2",
  filename: "LFM3-1.7B-Q8_0.gguf",
  sha256: "8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6",
  source_repo_id: "LiquidAI/LFM3-1.7B",
  source_revision: "1da06bec6a819da10b54c67e5c90972b46d81871",
};
const ISSUE = 12;
const LANGUAGE_GROUP = "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)";

const nativeHardware = {
  architecture: "arm64", os: "Darwin 25.1.0", ram_bytes: 34_359_738_368,
  free_disk_bytes: 500_000_000_000, gpus: [], gpu_memory_utilisation: 0.8,
};
const cudaHardware = {
  architecture: "amd64", free_disk_bytes: 1_000_000, gpu_memory_utilisation: 0.8,
  selected_gpu_index: 0, selected_gpu_uuid: "gpu",
  gpus: [{ index: 0, name: "GPU", uuid: "gpu", free_memory_bytes: 1_000_000, total_memory_bytes: 1_000_000 }],
};

function externalBody(overrides = {}) {
  return {
    protocol_version: "volunteer-worker/v1", worker_version: "1.0.0", euroeval_version: "1.0.0",
    capabilities: ["external_generation_v1", "canonical_coverage_v1"],
    hardware: { ...nativeHardware },
    execution: { kind: "external_server", artifact: { ...PINNED } },
    ...overrides,
  };
}

const ggufIssue = {
  number: ISSUE, title: `[MODEL EVALUATION REQUEST] ${PINNED.repo_id}`,
  body: `### Model ID\n\n${PINNED.repo_id}\n\n- [x] ${LANGUAGE_GROUP}\n`, state: "open", assignees: [],
};
const sourceIssue = {
  number: 13, title: `[MODEL EVALUATION REQUEST] ${PINNED.source_repo_id}`,
  body: `### Model ID\n\n${PINNED.source_repo_id}\n\n- [x] ${LANGUAGE_GROUP}\n`, state: "open", assignees: [],
};
const cudaIssue = {
  number: 14, title: "[MODEL EVALUATION REQUEST] org/model",
  body: `### Model ID\n\norg/model\n\n- [x] ${LANGUAGE_GROUP}\n`, state: "open", assignees: [],
};

const artifactInfo = {
  id: PINNED.repo_id, sha: PINNED.revision, private: false, gated: false,
  siblings: [
    { rfilename: PINNED.filename, size: 1_700_000_000,
      lfs: { sha256: PINNED.sha256, size: 1_700_000_000, pointerSize: 512 } },
    { rfilename: "README.md", size: 500 },
  ],
};
const sourceInfo = {
  id: PINNED.source_repo_id, sha: PINNED.source_revision, private: false, gated: false,
  pipeline_tag: "text-generation",
};
const sourceConfig = { model_type: "lfm3", architectures: ["Lfm3ForCausalLM"], is_encoder_decoder: false };

/**
 * Hermetic claim fixture: in-memory Redis, GitHub and Hugging Face. Every
 * Hugging Face model lookup is recorded so tests can prove that external
 * claims pin revisions and never resolve latest or read non-JSON files.
 */
async function runClaim({
  body, issues = [ggufIssue], enabled = "true", artifact = artifactInfo,
  source = sourceInfo, config = sourceConfig, envOverrides = {},
}) {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  Object.assign(process.env, {
    VOLUNTEER_MARKER_SECRET: "marker-secret",
    VOLUNTEER_WORKER_IMAGE_DIGEST: "sha256:image", EUROEVAL_VERSION: "1.0.0",
    VOLUNTEER_WORKER_VERSION: "1.0.0", UPSTASH_REDIS_REST_URL: "https://redis.test",
    UPSTASH_REDIS_REST_TOKEN: "redis-token", GITHUB_TOKEN: "github-token",
    VOLUNTEER_EXTERNAL_WORKERS_ENABLED: enabled,
    VOLUNTEER_SCOPE_POLICY_JSON: JSON.stringify({
      policy_version: "test-policy", policies: [
        { euroeval_version: "1.0.0", model_type: "generative", language: "da", language_group: LANGUAGE_GROUP,
          allowed_identity_suffix_sets: [[JSON.stringify(["sst5", true, false])]],
          task_groups: ["classification"], warnings: [] },
        { euroeval_version: "1.0.0", model_type: "encoder", language: "da", language_group: LANGUAGE_GROUP,
          allowed_identity_suffix_sets: [[JSON.stringify(["dkhate", true, false])]],
          task_groups: ["classification"], warnings: [] },
      ],
    }),
    ...envOverrides,
  });
  const credentialKey = `euroeval:worker:credential:${await sha256("credential")}`;
  const values = new Map([[credentialKey, JSON.stringify({ contributor: "alice" })]]);
  const calls = { hf: [], issueLists: 0, mutations: 0 };
  const mutableIssues = issues.map((issue) => ({ ...issue }));
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input); const method = init.method || "GET";
    if (url === "https://redis.test") {
      const command = JSON.parse(init.body);
      if (command[0] === "GET") return Response.json({ result: values.get(command[1]) || null });
      if (command[0] === "SMEMBERS") return Response.json({ result: [] });
      if (command[0] === "INCR" || command[0] === "EXPIRE") return Response.json({ result: 1 });
      if (command[0] === "SET") { values.set(command[1], command[2]); return Response.json({ result: "OK" }); }
      if (command[0] === "EVAL") {
        if (String(command[1]).includes("released=true")) { values.delete(command[3]); values.delete(command[4]); }
        else if (String(command[1]).includes("local issue=redis.call")) {
          values.set(command[3], command[5]); values.set(command[4], command[5]);
        }
        return Response.json({ result: 1 });
      }
      return Response.json({ result: 1 });
    }
    if (url.includes("huggingface.co")) {
      calls.hf.push(url);
      if (url.includes("/buckets/EuroEval/results/resolve/")) return new Response(null, { status: 404 });
      if (url.includes(`/api/models/${PINNED.repo_id}/revision/${PINNED.revision}`)) return Response.json(artifact);
      if (url.includes(`/api/models/${PINNED.source_repo_id}/revision/${PINNED.source_revision}`)) return Response.json(source);
      if (url.includes(`/${PINNED.source_repo_id}/raw/${PINNED.source_revision}/config.json`)) return Response.json(config);
      if (url.includes("/api/models/org/model")) return Response.json({ id: "org/model",
        sha: "a".repeat(40), private: false, gated: false, pipeline_tag: "fill-mask",
        siblings: [{ rfilename: "model.safetensors", size: 100 }, { rfilename: "config.json", size: 10 }] });
      if (url.includes("/org/model/raw/")) return Response.json({ model_type: "bert", architectures: ["BertModel"] });
      throw new Error(`Unexpected Hugging Face request: ${url}`);
    }
    if (url.includes("/repos/EuroEval/EuroEval/issues?")) { calls.issueLists++; return Response.json(mutableIssues); }
    const issue = mutableIssues.find((item) =>
      url.endsWith(`/issues/${item.number}`) || url.endsWith(`/issues/${item.number}/assignees`));
    if (issue && url.endsWith(`/issues/${issue.number}/assignees`)) {
      calls.mutations++;
      issue.assignees = [{ login: "alice" }];
      return new Response(null, { status: 204 });
    }
    if (issue && method === "GET") return Response.json(issue);
    if (issue && method === "PATCH") {
      calls.mutations++;
      issue.body = JSON.parse(init.body).body;
      return Response.json(issue);
    }
    throw new Error(`Unexpected request: ${method} ${url}`);
  };
  try {
    const response = await claim(new Request("https://euroeval.test/api/worker/claim", {
      method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
      body: JSON.stringify(body),
    }));
    return {
      response, json: await response.json(), values, calls, issues: mutableIssues,
      storedLease: (issue = ISSUE) => {
        const raw = values.get(`euroeval:worker:lease:${issue}:da`);
        return raw ? JSON.parse(raw) : null;
      },
    };
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
}

test("external claims are rejected while the feature gate is off", async () => {
  const { response, json, calls, values } = await runClaim({ body: externalBody(), enabled: "false" });
  assert.equal(response.status, 404, JSON.stringify(json));
  assert.match(json.error, /not enabled/i);
  assert.equal(calls.hf.length, 0);
  assert.equal(calls.issueLists, 0);
  assert.equal([...values.keys()].some((key) => key.startsWith("euroeval:worker:lease")), false);
});

test("a valid native claim leases the approved GGUF artifact with execution and null digest", async () => {
  const { response, json, calls, storedLease } = await runClaim({ body: externalBody() });
  assert.equal(response.status, 200, JSON.stringify(json));
  assert.equal(json.issue_number, ISSUE);
  assert.equal(json.language, "da");
  assert.equal(json.model_id, PINNED.repo_id);
  assert.equal(json.model_revision, PINNED.revision);
  assert.equal(json.image_digest, null);
  assert.deepEqual(json.execution, { kind: "external_server", artifact: PINNED });
  assert.equal("selected_gpu_index" in json, false);
  assert.equal("selected_gpu_uuid" in json, false);
  assert.equal(json.worker_version, "1.0.0");
  assert.equal(json.model_type, "generative");
  assert.equal(json.model_metadata.pipeline_tag, "text-generation");
  assert.deepEqual(json.model_metadata.architectures, ["Lfm3ForCausalLM"]);
  assert.ok(json.expected_scope.allowed_identity_suffix_sets.length > 0);
  assert.ok(json.contamination_canary);

  const lease = storedLease();
  assert.ok(lease, "lease must be persisted");
  assert.equal(lease.lease_id, json.lease_id);
  assert.deepEqual(lease.execution, { kind: "external_server", artifact: PINNED });
  assert.equal(lease.image_digest, null);
  assert.equal("selected_gpu_index" in lease, false);
  assert.equal("selected_gpu_uuid" in lease, false);
  assert.equal(lease.model_id, PINNED.repo_id);
  assert.equal(lease.model_revision, PINNED.revision);
  assert.equal(lease.gpu_memory_utilisation, 0.8);
  assert.equal(lease.record_revision_required, true);

  // The artifact is verified at its pinned revision and the source metadata is
  // read only as JSON at the pinned source revision.
  assert.ok(calls.hf.some((url) =>
    url.includes(`/api/models/${PINNED.repo_id}/revision/${PINNED.revision}?blobs=true`)));
  assert.ok(calls.hf.some((url) =>
    url.includes(`/api/models/${PINNED.source_repo_id}/revision/${PINNED.source_revision}`)));
  assert.ok(calls.hf.some((url) =>
    url.includes(`/${PINNED.source_repo_id}/raw/${PINNED.source_revision}/config.json`)));
  assert.equal(calls.hf.filter((url) =>
    url.includes("/api/models/") && !url.includes("/revision/")).length, 0);
  assert.equal(calls.hf.filter((url) => /\.(py|safetensors)$/.test(url)).length, 0);
  // Canonical coverage is keyed on the GGUF artifact repo, never the source repo.
  const coverage = calls.hf.filter((url) => url.includes("/buckets/EuroEval/results/resolve/"));
  assert.ok(coverage.some((url) => url.includes("LiquidAI_LFM3-1.7B-GGUF/")), coverage.join(" "));
  assert.equal(coverage.some((url) => url.includes("LiquidAI_LFM3-1.7B/")), false);
});

for (const [field, value] of [
  ["sha256", "c".repeat(64)],
  ["revision", "b".repeat(40)],
  ["filename", "LFM3-1.7B-Q4_0.gguf"],
  ["repo_id", "LiquidAI/LFM3-1.7B-GGUF-other"],
  ["source_revision", "e".repeat(40)],
]) {
  test(`an unapproved claimed artifact (${field}) is rejected before discovery`, async () => {
    const body = externalBody();
    body.execution.artifact[field] = value;
    const { response, json, calls, values } = await runClaim({ body });
    assert.equal(response.status, 422, JSON.stringify(json));
    assert.match(json.error, /not approved/i);
    assert.equal(calls.hf.length, 0);
    assert.equal(calls.issueLists, 0);
    assert.equal([...values.keys()].some((key) => key.startsWith("euroeval:worker:lease")), false);
  });
}

for (const overrides of [
  { worker_version: "0.9" },
  { euroeval_version: "9.9.9" },
  { euroeval_version: undefined },
]) {
  test(`a wrong native version is rejected (${JSON.stringify(overrides)})`, async () => {
    const { response, json, calls } = await runClaim({ body: externalBody(overrides) });
    assert.equal(response.status, 422, JSON.stringify(json));
    assert.equal(calls.hf.length, 0);
    assert.equal(calls.issueLists, 0);
  });
}

test("external claims require both capabilities and reject CUDA-only fields", async () => {
  const cases = [
    externalBody({ capabilities: ["canonical_coverage_v1"] }),
    externalBody({ capabilities: ["external_generation_v1"] }),
    externalBody({ capabilities: ["external_generation_v1", "canonical_coverage_v1", "managed_image_v1"] }),
    externalBody({ image_digest: "sha256:image" }),
    (() => { const body = externalBody(); body.hardware.gpus = [{ name: "GPU", uuid: "gpu",
      free_memory_bytes: 1, total_memory_bytes: 1 }]; return body; })(),
    (() => { const body = externalBody(); body.hardware.selected_gpu_uuid = "gpu"; return body; })(),
    (() => { const body = externalBody(); body.hardware.selected_gpu_index = 0; return body; })(),
    (() => { const body = externalBody(); delete body.hardware.os; return body; })(),
    (() => { const body = externalBody(); body.execution = { kind: "cuda_server", artifact: { ...PINNED } }; return body; })(),
  ];
  for (const body of cases) {
    const { response, calls, values } = await runClaim({ body });
    assert.equal(response.status, 400, JSON.stringify(body));
    assert.equal(calls.hf.length, 0);
    assert.equal(calls.issueLists, 0);
    assert.equal([...values.keys()].some((key) => key.startsWith("euroeval:worker:lease")), false);
  }
});

test("the upstream source model issue is never claimable by an external worker", async () => {
  const { response, json, calls, values } = await runClaim({ body: externalBody(), issues: [sourceIssue] });
  assert.equal(response.status, 200, JSON.stringify(json));
  assert.deepEqual(json, { protocol_version: "volunteer-worker/v1", status: "no_work" });
  // No artifact verification and no source-issue scanning at all.
  assert.equal(calls.hf.length, 0);
  assert.equal(calls.mutations, 0);
  assert.equal([...values.keys()].some((key) => key.startsWith("euroeval:worker:lease")), false);
});

for (const [name, artifact] of [
  ["LFS hash mismatch", { ...artifactInfo, siblings: [
    { rfilename: PINNED.filename, size: 1_700_000_000, lfs: { sha256: "d".repeat(64), size: 1_700_000_000 } },
    { rfilename: "README.md", size: 500 }] }],
  ["file missing at the pinned revision", { ...artifactInfo, siblings: [{ rfilename: "README.md", size: 500 }] }],
  ["gated artifact", { ...artifactInfo, gated: true }],
  ["revision drift", { ...artifactInfo, sha: "e".repeat(40) }],
]) {
  test(`an approved artifact that fails HF verification is not leased (${name})`, async () => {
    const { response, json, calls, values } = await runClaim({ body: externalBody(), artifact });
    assert.equal(response.status, 200, JSON.stringify(json));
    assert.deepEqual(json, { protocol_version: "volunteer-worker/v1", status: "no_work" });
    assert.equal(calls.mutations, 0);
    assert.equal([...values.keys()].some((key) => key.startsWith("euroeval:worker:lease")), false);
  });
}

test("a CUDA claim keeps its digest and selected GPU next to the external gate", async () => {
  const body = { protocol_version: "volunteer-worker/v1", worker_version: "1.0.0", hardware: { ...cudaHardware } };
  const { response, json, storedLease } = await runClaim({ body, issues: [cudaIssue] });
  assert.equal(response.status, 200, JSON.stringify(json));
  assert.equal(json.issue_number, 14);
  assert.equal(json.model_id, "org/model");
  assert.equal(json.image_digest, "sha256:image");
  assert.equal(json.selected_gpu_index, 0);
  assert.equal(json.selected_gpu_uuid, "gpu");
  assert.equal("execution" in json, false);
  const lease = storedLease(14);
  assert.ok(lease);
  assert.equal(lease.image_digest, "sha256:image");
  assert.equal(lease.selected_gpu_uuid, "gpu");
  assert.equal("execution" in lease, false);
});
