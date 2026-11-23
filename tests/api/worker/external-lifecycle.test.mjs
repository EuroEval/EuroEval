import test from "node:test";
import assert from "node:assert/strict";
import { fetch as resultFetch } from "../../../api/worker/result.ts";
import { fetch as finaliseFetch, EXTERNAL_IMAGE_DIGEST_PROVENANCE } from "../../../api/worker/finalise.ts";
import { signVolunteerMarker, sha256 } from "../../../api/worker/_lib.ts";

const PINNED = {
  repo_id: "LiquidAI/LFM3-1.7B-GGUF",
  revision: "cd0c74c575b5c416f3011b5b4c298077f26ca8c2",
  filename: "LFM3-1.7B-Q8_0.gguf",
  sha256: "8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6",
  source_repo_id: "LiquidAI/LFM3-1.7B",
  source_revision: "1da06bec6a819da10b54c67e5c90972b46d81871",
};
const ISSUE = 21;
const LANGUAGE = "da";
const LANGUAGE_GROUP = "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)";
const SCOPE_SUFFIX = JSON.stringify(["sst5", true, false]);
const EXECUTION = { kind: "external_server", artifact: { ...PINNED } };

function externalRecord(artifactOverride) {
  return {
    schema_version: "0.3.0",
    model_info: {
      id: PINNED.repo_id, name: PINNED.repo_id, revision: PINNED.revision,
      additional_details: artifactOverride === undefined
        ? { gguf_artifact: { ...PINNED } } : { gguf_artifact: artifactOverride },
    },
    eval_library: { name: "euroeval", version: "1.0.0", additional_details: {
      dataset: "sst5", task: "classification", raw_results: [], languages: ["da"],
      validation_split: true, few_shot: false } },
    evaluation_results: [{ evaluation_name: "accuracy", source_data: { dataset_name: "sst5" },
      metric_config: { lower_is_better: false }, score_details: { score: 0.5 } }],
  };
}

function cudaRecord() {
  return {
    schema_version: "0.3.0",
    model_info: { id: "org/model", revision: "a".repeat(40) },
    eval_library: { name: "euroeval", version: "1.0.0", additional_details: {
      dataset: "sst5", task: "classification", raw_results: [], languages: ["da"],
      validation_split: true, few_shot: false } },
    evaluation_results: [{ evaluation_name: "accuracy", source_data: { dataset_name: "sst5" },
      metric_config: { lower_is_better: false }, score_details: { score: 0.5 } }],
  };
}

function externalLease(overrides = {}) {
  return {
    lease_id: "ext-lease", issue_number: ISSUE, language: LANGUAGE, worker: "worker", contributor: "alice",
    model_id: PINNED.repo_id, model_revision: PINNED.revision, euroeval_version: "1.0.0",
    image_digest: null, worker_version: "1.0.0", gpu_memory_utilisation: 0.8,
    expires_at: new Date(Date.now() + 60_000).toISOString(), model_type: "generative",
    model_metadata: { pipeline_tag: "text-generation", architectures: ["Lfm3ForCausalLM"],
      model_type: "generative", is_encoder_decoder: false },
    ownership_mode: "public", record_revision_required: true,
    execution: { kind: "external_server", artifact: { ...PINNED } },
    expected_scope: { policy_version: "test-policy", language_group: LANGUAGE_GROUP,
      allowed_identity_suffix_sets: [[SCOPE_SUFFIX]], task_groups: ["classification"], warnings: [] },
    ...overrides,
  };
}

function cudaLease(overrides = {}) {
  return {
    lease_id: "cuda-lease", issue_number: ISSUE, language: LANGUAGE, worker: "worker", contributor: "alice",
    model_id: "org/model", model_revision: "a".repeat(40), euroeval_version: "1.0.0",
    image_digest: "sha256:image", worker_version: "1.0.0", gpu_memory_utilisation: 0.8,
    selected_gpu_index: 0, selected_gpu_uuid: "gpu",
    expires_at: new Date(Date.now() + 60_000).toISOString(), model_type: "generative",
    model_metadata: { pipeline_tag: "text-generation", architectures: ["LlamaForCausalLM"],
      model_type: "generative", is_encoder_decoder: false },
    ownership_mode: "public", record_revision_required: true,
    expected_scope: { policy_version: "test-policy", language_group: LANGUAGE_GROUP,
      allowed_identity_suffix_sets: [[SCOPE_SUFFIX]], task_groups: ["classification"], warnings: [] },
    ...overrides,
  };
}

/**
 * Hermetic in-memory broker fixture: Redis (string values plus sets), GitHub
 * issue state, and the Hugging Face staging upload path. No child processes,
 * so the file also runs under Node versions whose process teardown aborts on
 * spawned helpers.
 */
async function runWorker({ lease, issueNumber = ISSUE, preseed = [] }) {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  Object.assign(process.env, {
    VOLUNTEER_MARKER_SECRET: "marker-secret", GITHUB_TOKEN: "github-token",
    HF_STAGING_BUCKET: "EuroEval/staging", HF_TOKEN: "hf_test-token",
    UPSTASH_REDIS_REST_URL: "https://redis.test", UPSTASH_REDIS_REST_TOKEN: "redis-token",
  });
  const credential = "worker-credential";
  const values = new Map([
    [`euroeval:worker:credential:${await sha256(credential)}`, JSON.stringify({ contributor: "alice" })],
    [`euroeval:worker:lease-id:${lease.lease_id}`, JSON.stringify(lease)],
    ...preseed,
  ]);
  const sets = new Map();
  const marker = await signVolunteerMarker(issueNumber, {
    protocol_version: "volunteer-worker/v1", coordinator: "coordinator", submission: "active",
    leases: [{ lease_id: lease.lease_id, language: lease.language, worker: lease.worker,
      contributor: lease.contributor, expires_at: lease.expires_at, ownership_mode: "public" }],
    public_assignees: ["alice"],
  });
  const modelId = lease.model_id;
  const issue = { number: issueNumber, title: `[MODEL EVALUATION REQUEST] ${modelId}`,
    body: `### Model ID\n\n${modelId}\n\n- [x] ${LANGUAGE_GROUP}\n\n<!-- euroeval-volunteer-worker:v1 ${JSON.stringify(marker)} -->`,
    state: "open", labels: [], assignees: [{ login: "alice" }] };
  const redisCommand = (command) => {
    if (command[0] === "GET") return { result: values.get(command[1]) ?? null };
    if (command[0] === "SET") {
      const nx = command.includes("NX");
      if (nx && values.has(command[1])) return { result: null };
      values.set(command[1], command[2]); return { result: "OK" };
    }
    if (command[0] === "DEL") { let n = 0; for (const key of command.slice(1)) if (values.delete(key)) n += 1; return { result: n }; }
    if (command[0] === "EXPIRE" || command[0] === "PTTL") return { result: 1 };
    if (command[0] === "INCR") { const next = Number(values.get(command[1]) || 0) + 1; values.set(command[1], String(next)); return { result: next }; }
    if (command[0] === "SADD") { const set = sets.get(command[1]) || new Set(); let n = 0;
      for (const member of command.slice(2)) if (!set.has(member)) { set.add(member); n += 1; } sets.set(command[1], set); return { result: n }; }
    if (command[0] === "SMEMBERS") return { result: [...(sets.get(command[1]) || [])] };
    if (command[0] === "SISMEMBER") return { result: (sets.get(command[1]) || new Set()).has(command[2]) ? 1 : 0 };
    if (command[0] === "SCARD") return { result: (sets.get(command[1]) || new Set()).size };
    if (command[0] === "EVAL") {
      const script = String(command[1]); const numkeys = Number(command[2]);
      const keys = command.slice(3, 3 + numkeys); const argv = command.slice(3 + numkeys);
      if (script.includes("released=true")) {
        const issueRaw = values.get(keys[0]);
        if (issueRaw) { const item = JSON.parse(issueRaw); if (item.lease_id === argv[0]) values.delete(keys[0]); }
        const byidRaw = values.get(keys[1]);
        if (byidRaw) { const item = JSON.parse(byidRaw); if (item.lease_id === argv[0]) { item.released = true; values.set(keys[1], JSON.stringify(item)); } }
        return { result: 1 };
      }
      if (script.includes("then return redis.call('DEL',KEYS[1]")) {
        return { result: values.get(keys[0]) === argv[0] ? (values.delete(keys[0]), 1) : 0 };
      }
      if (script.includes("then return redis.call('EXPIRE'")) return { result: values.get(keys[0]) === argv[0] ? 1 : 0 };
      if (script.includes("return 'reserved'")) {
        const current = values.get(keys[0]);
        const track = () => { const set = sets.get(keys[1]) || new Set();
          if (!set.has(argv[4]) && set.size >= Number(argv[5])) return 0;
          set.add(argv[4]); sets.set(keys[1], set); return 1; };
        if (!current) {
          if (track() === 0) return { result: "full" };
          values.set(keys[0], argv[0]); return { result: "reserved" };
        }
        const item = JSON.parse(current);
        if (item.digest !== argv[1]) return { result: "busy" };
        if (item.status === "uploaded") return { result: track() === 0 ? "full" : "duplicate" };
        if (item.lease_id === argv[2]) return { result: track() === 0 ? "full" : "retry" };
        return { result: "busy" };
      }
      if (script.includes("if not current then return 1 end")) {
        const current = values.get(keys[0]);
        if (!current) return { result: 1 };
        const item = JSON.parse(current);
        if (item.digest !== argv[0] || item.lease_id !== argv[1]) return { result: 0 };
        values.delete(keys[0]); return { result: 1 };
      }
      if (script.includes("item.status == 'uploading'")) {
        const current = values.get(keys[0]);
        if (current) { const item = JSON.parse(current);
          if (item.digest === argv[0] && item.lease_id === argv[1] && item.status === "uploading") values.delete(keys[0]); }
        return { result: 1 };
      }
      if (script.includes("ARGV[3],'EX',ARGV[4]); return 1")) {
        const current = values.get(keys[0]);
        if (!current) return { result: 0 };
        const item = JSON.parse(current);
        if (item.digest !== argv[0] || item.lease_id !== argv[1]) return { result: 0 };
        values.set(keys[0], argv[2]); return { result: 1 };
      }
      return { result: 1 };
    }
    throw new Error(`Unexpected Redis command: ${command[0]}`);
  };
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input); const method = init.method || "GET";
    if (url === "https://redis.test") return Response.json(redisCommand(JSON.parse(init.body)));
    if (url.endsWith(`/issues/${issueNumber}`) && method === "GET") return Response.json(issue);
    if (url.endsWith(`/issues/${issueNumber}`) && method === "PATCH") { issue.body = JSON.parse(init.body).body; return Response.json(issue); }
    if (url.endsWith(`/issues/${issueNumber}/assignees`)) {
      const requested = JSON.parse(init.body).assignees.map((login) => login.toLowerCase());
      if (method === "POST") issue.assignees = [...new Map([...issue.assignees, ...requested.map((login) => ({ login }))].map((item) => [item.login.toLowerCase(), item])).values()];
      if (method === "DELETE") issue.assignees = issue.assignees.filter((item) => !requested.includes(item.login.toLowerCase()));
      return new Response(null, { status: 204 });
    }
    if (url.endsWith(`/issues/${issueNumber}/labels`) && method === "POST") {
      issue.labels = [...(issue.labels || []), ...JSON.parse(init.body).labels.map((name) => ({ name }))];
      return Response.json(issue.labels);
    }
    const labelDelete = url.match(new RegExp(`/issues/${issueNumber}/labels/([^/?]+)$`));
    if (labelDelete && method === "DELETE") {
      issue.labels = (issue.labels || []).filter((item) => item.name !== decodeURIComponent(labelDelete[1]));
      return Response.json(issue.labels);
    }
    if (url.includes("xet-write-token")) return Response.json({ accessToken: "xet-token", casUrl: "https://cas.test", exp: Math.floor(Date.now() / 1000) + 3600 });
    if (url.endsWith("/batch")) return Response.json({ failed: [] });
    if (url.includes("huggingface.co/api/buckets/")) return Response.json({ private: true });
    if (url.includes("cas.test") || url.includes("/v1/chunks/")) return method === "GET" ? new Response(null, { status: 404 }) : Response.json({ ok: true });
    if (url.includes("huggingface.co")) return Response.json({ failed: [] });
    throw new Error(`Unexpected request: ${method} ${url}`);
  };
  try {
    const post = (handler, path, payload) => handler(new Request(`https://euroeval.test/api/worker/${path}`, {
      method: "POST", headers: { authorization: `Bearer ${credential}`, "content-type": "application/json" },
      body: JSON.stringify(payload),
    }));
    const submit = async (body) => post(resultFetch, "result", body);
    const finalise = async () => post(finaliseFetch, "finalise", { protocol_version: "volunteer-worker/v1", lease_id: lease.lease_id });
    const resultBody = async (record, overrides = {}) => {
      const recordJson = JSON.stringify(record);
      return { protocol_version: "volunteer-worker/v1", lease_id: lease.lease_id, issue_number: issueNumber,
        language: lease.language, model_id: lease.model_id, model_revision: lease.model_revision,
        euroeval_version: lease.euroeval_version, image_digest: lease.image_digest,
        worker_version: lease.worker_version, record_json: recordJson, digest: await sha256(recordJson),
        ...(lease.execution ? { execution: { kind: "external_server", artifact: { ...PINNED } } } : {}),
        ...overrides };
    };
    const redisGet = async (key) => (await globalThis.fetch("https://redis.test", { method: "POST", body: JSON.stringify(["GET", key]) })).json().then((payload) => payload.result);
    const redisSmembers = async (key) => (await globalThis.fetch("https://redis.test", { method: "POST", body: JSON.stringify(["SMEMBERS", key]) })).json().then((payload) => payload.result);
    const restore = () => { globalThis.fetch = originalFetch; process.env = originalEnv; };
    return { submit, finalise, resultBody, redisGet, redisSmembers, values, sets, issue, lease, restore };
  } catch (error) {
    globalThis.fetch = originalFetch; process.env = originalEnv;
    throw error;
  }
}

async function withWorker(options, run) {
  const fixture = await runWorker(options);
  try { return await run(fixture); } finally { fixture.restore(); }
}

test("external result upload accepts the exact approved lease execution and stores the artifact", async () => {
  await withWorker({ lease: externalLease() }, async ({ submit, resultBody, redisGet, redisSmembers, lease }) => {
    const record = externalRecord();
    const response = await submit(await resultBody(record));
    assert.equal(response.status, 201, await response.clone().text());
    const body = await response.json();
    assert.equal(body.status, "uploaded");
    const identity = JSON.stringify([PINNED.repo_id, "sst5", true, false]);
    const stored = JSON.parse(await redisGet(`euroeval:worker:record-identity:${await sha256(identity)}`));
    assert.equal(stored.status, "uploaded");
    assert.deepEqual(stored.artifact, PINNED);
    const entries = (await redisSmembers(`euroeval:worker:results:${lease.lease_id}`)).map((item) => JSON.parse(item));
    assert.equal(entries.length, 1);
    assert.deepEqual(entries[0].artifact, PINNED);
  });
});

test("duplicate external results replay only after re-verifying the record artifact", async () => {
  const lease = externalLease();
  const record = externalRecord();
  const recordJson = JSON.stringify(record);
  const digest = await sha256(recordJson);
  const identity = JSON.stringify([PINNED.repo_id, "sst5", true, false]);
  const identityKey = `euroeval:worker:record-identity:${await sha256(identity)}`;
  await withWorker({ lease, preseed: [[identityKey, JSON.stringify({ digest, lease_id: lease.lease_id, status: "uploaded",
    issue_number: ISSUE, language: LANGUAGE, path: `volunteer/submissions/${lease.lease_id}/results/${digest}.json`, artifact: { ...PINNED } })]] },
    async ({ submit, resultBody }) => {
      const replay = await submit(await resultBody(record));
      assert.equal(replay.status, 200, await replay.clone().text());
      assert.equal((await replay.json()).status, "duplicate");
      // A tampered record descriptor is rejected even on the duplicate path.
      const tampered = await submit(await resultBody(externalRecord({ ...PINNED, sha256: "c".repeat(64) })));
      assert.equal(tampered.status, 422);
      assert.match((await tampered.json()).error, /artifact/i);
      const withoutArtifact = externalRecord();
      delete withoutArtifact.model_info.additional_details.gguf_artifact;
      const missing = await submit(await resultBody(withoutArtifact));
      assert.equal(missing.status, 422);
      assert.match((await missing.json()).error, /artifact/i);
    });
});

test("result rejects body, lease, artifact and record mismatches and mode downgrades", async () => {
  const cases = [];
  const external = externalLease();
  cases.push(["external lease without body execution", external, externalRecord(), { execution: undefined }, 422]);
  cases.push(["external lease with tampered body artifact", external, externalRecord(), { execution: { kind: "external_server", artifact: { ...PINNED, sha256: "c".repeat(64) } } }, 422]);
  cases.push(["external lease with unapproved body artifact", external, externalRecord(), { execution: { kind: "external_server", artifact: { ...PINNED, repo_id: "org/unapproved-gguf" } } }, 422]);
  cases.push(["external lease with unknown kind", external, externalRecord(), { execution: { kind: "cuda_server", artifact: { ...PINNED } } }, 400]);
  cases.push(["external lease with string image digest", external, externalRecord(), { image_digest: "sha256:image" }, 422]);
  cases.push(["external lease record missing artifact", external, (() => { const r = externalRecord(); delete r.model_info.additional_details.gguf_artifact; return r; })(), {}, 422]);
  cases.push(["external lease record with wrong artifact", external, externalRecord({ ...PINNED, filename: "LFM3-1.7B-Q4_0.gguf" }), {}, 422]);
  cases.push(["external record with arbitrary name", external, (() => { const r = externalRecord(); r.model_info.name = "wrong-model"; return r; })(), {}, 422]);
  const cuda = cudaLease();
  cases.push(["CUDA lease with null image digest", cuda, cudaRecord(), { image_digest: null }, 422]);
  cases.push(["CUDA lease with body execution", cuda, cudaRecord(), { execution: { kind: "external_server", artifact: { ...PINNED } } }, 422]);
  cases.push(["CUDA lease record carrying artifact", cuda, (() => { const r = cudaRecord(); r.model_info.additional_details = { gguf_artifact: { ...PINNED } }; return r; })(), {}, 422]);
  for (const [name, lease, record, overrides, expected] of cases) {
    await withWorker({ lease }, async ({ submit, resultBody }) => {
      const response = await submit(await resultBody(record, overrides));
      assert.equal(response.status, expected, `${name}: ${await response.clone().text()}`);
    });
  }
});

test("result accepts a CUDA lease unchanged (string digest, no execution)", async () => {
  await withWorker({ lease: cudaLease() }, async ({ submit, resultBody, redisGet }) => {
    const response = await submit(await resultBody(cudaRecord()));
    assert.equal(response.status, 201, await response.clone().text());
    const identity = JSON.stringify(["org/model", "sst5", true, false]);
    const stored = JSON.parse(await redisGet(`euroeval:worker:record-identity:${await sha256(identity)}`));
    assert.equal("artifact" in stored, false);
  });
});

test("result treats a malformed or unapproved stored lease execution as a deployment failure", async () => {
  for (const [name, execution, leaseOverrides] of [
    ["malformed", { kind: "external_server" }, {}],
    ["unapproved", { kind: "external_server", artifact: { ...PINNED, sha256: "c".repeat(64) } }, {}],
    ["identity mismatch", { kind: "external_server", artifact: { ...PINNED } }, { model_id: "LiquidAI/Other-GGUF" }],
  ]) {
    await withWorker({ lease: externalLease({ execution, ...leaseOverrides }) }, async ({ submit, resultBody }) => {
      const response = await submit(await resultBody(externalRecord()));
      assert.equal(response.status, 503, name);
      assert.match((await response.json()).error, /configuration|execution|identity/i);
    });
  }
});

test("finalise persists execution with null image and explicit external provenance", async () => {
  const lease = externalLease();
  const record = externalRecord();
  const recordJson = JSON.stringify(record);
  const digest = await sha256(recordJson);
  const identity = JSON.stringify([PINNED.repo_id, "sst5", true, false]);
  await withWorker({ lease }, async ({ submit, finalise, resultBody, redisGet, redisSmembers }) => {
    assert.equal((await submit(await resultBody(record))).status, 201);
    const response = await finalise();
    assert.equal(response.status, 200, await response.clone().text());
    const body = await response.json();
    assert.equal(body.status, "ready");
    assert.equal(body.coverage.records, 1);
    const receipt = JSON.parse(await redisGet(`euroeval:worker:finalisation:${lease.lease_id}`));
    const manifest = JSON.parse(receipt.manifest);
    assert.deepEqual(manifest.execution, EXECUTION);
    assert.equal(manifest.image_digest, null);
    assert.equal(manifest.image_digest_provenance, EXTERNAL_IMAGE_DIGEST_PROVENANCE);
    assert.equal(manifest.image_digest_provenance, "native-external-server-not-runtime-attested");
    assert.equal("selected_gpu_index" in manifest, false);
    assert.equal("selected_gpu_uuid" in manifest, false);
    assert.equal(manifest.model.id, PINNED.repo_id);
    assert.equal(manifest.model.revision, PINNED.revision);
    assert.deepEqual(manifest.results[0].artifact, PINNED);
  });
});

test("finalise rejects entries that do not carry the lease artifact, including resumed replays", async () => {
  const lease = externalLease();
  const record = externalRecord();
  const recordJson = JSON.stringify(record);
  const digest = await sha256(recordJson);
  const identity = JSON.stringify([PINNED.repo_id, "sst5", true, false]);
  const entry = { digest, identity, path: `volunteer/submissions/${lease.lease_id}/results/${digest}.json`, failed: 0 };
  // Fresh finalisation with an entry missing the artifact.
  await withWorker({ lease }, async ({ finalise, sets }) => {
    sets.set(`euroeval:worker:results:${lease.lease_id}`, new Set([JSON.stringify(entry)]));
    const response = await finalise();
    assert.equal(response.status, 422, await response.clone().text());
    assert.match((await response.json()).error, /artifact/i);
  });
  // Resumed manifest_uploaded receipt with a tampered entry artifact.
  const statusKey = `euroeval:worker:finalisation:${lease.lease_id}`;
  await withWorker({ lease, preseed: [[statusKey, JSON.stringify({ status: "manifest_uploaded", submission_id: lease.lease_id,
    lease, entries: [{ ...entry, artifact: { ...PINNED, sha256: "c".repeat(64) } }],
    manifest_path: `volunteer/manifests/${lease.lease_id}.json` })]] }, async ({ finalise }) => {
    const response = await finalise();
    assert.equal(response.status, 422, await response.clone().text());
    assert.match((await response.json()).error, /artifact/i);
  });
  // already_finalised replay with a missing artifact is also rejected.
  await withWorker({ lease, preseed: [[statusKey, JSON.stringify({ status: "ready", submission_id: lease.lease_id,
    lease, entries: [entry], manifest_path: `volunteer/manifests/${lease.lease_id}.json` })]] }, async ({ finalise }) => {
    const response = await finalise();
    assert.equal(response.status, 422, await response.clone().text());
  });
  // A correct resumed receipt completes.
  await withWorker({ lease, preseed: [[statusKey, JSON.stringify({ status: "manifest_uploaded", submission_id: lease.lease_id,
    lease, entries: [{ ...entry, artifact: { ...PINNED } }], manifest_path: `volunteer/manifests/${lease.lease_id}.json` })]] },
    async ({ finalise }) => {
      const response = await finalise();
      assert.equal(response.status, 200, await response.clone().text());
      assert.equal((await response.json()).status, "ready");
    });
});

test("CUDA finalise keeps the string digest, CUDA provenance and selected GPU", async () => {
  const lease = cudaLease();
  const record = cudaRecord();
  const recordJson = JSON.stringify(record);
  const digest = await sha256(recordJson);
  const identity = JSON.stringify(["org/model", "sst5", true, false]);
  await withWorker({ lease }, async ({ submit, finalise, resultBody, redisGet }) => {
    assert.equal((await submit(await resultBody(record))).status, 201);
    const response = await finalise();
    assert.equal(response.status, 200, await response.clone().text());
    const receipt = JSON.parse(await redisGet(`euroeval:worker:finalisation:${lease.lease_id}`));
    const manifest = JSON.parse(receipt.manifest);
    assert.equal("execution" in manifest, false);
    assert.equal(manifest.image_digest, "sha256:image");
    assert.equal(manifest.image_digest_provenance, "configured-required-not-runtime-attested");
    assert.equal(manifest.selected_gpu_index, 0);
    assert.equal(manifest.selected_gpu_uuid, "gpu");
    assert.equal("artifact" in manifest.results[0], false);
  });
});
