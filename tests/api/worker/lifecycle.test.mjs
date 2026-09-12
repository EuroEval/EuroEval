import test from "node:test";
import assert from "node:assert/strict";
import claim from "../../../api/worker/claim.ts";
import finalise from "../../../api/worker/finalise.ts";
import heartbeat from "../../../api/worker/heartbeat.ts";
import release from "../../../api/worker/release.ts";
import { parseVolunteerMarker, signVolunteerMarker, sha256, verifyVolunteerMarker } from "../../../api/worker/_lib.ts";

const issueNumber = 12;
const lease = {
  issue_number: issueNumber, language: "da", worker: "worker", contributor: "alice",
  model_id: "org/model", model_revision: "a".repeat(40), euroeval_version: "1.0.0",
  image_digest: "sha256:image", worker_version: "1.0.0", gpu_memory_utilisation: 0.8,
  expires_at: new Date(Date.now() + 60_000).toISOString(), lease_id: "lease",
  model_type: "encoder", model_metadata: { pipeline_tag: "fill-mask", architectures: ["BertModel"],
    model_type: "encoder", is_encoder_decoder: null },
  expected_scope: { policy_version: "test-policy", language_group: "Danish", identity_suffixes: ["x"], count: 1,
    task_groups: ["sequence_classification"], warnings: [] },
};

function markerBody(marker) {
  return `request\n\n<!-- harmless comment -->\n- [x] Greek\n\n<!-- euroeval-volunteer-worker:v1 ${JSON.stringify(marker)} -->\n`;
}

function mockBroker(issue, values) {
  return async (input, init = {}) => {
    const url = String(input);
    const method = init.method || "GET";
    if (url === "https://redis.test") {
      const command = JSON.parse(init.body);
      let result = "OK";
      if (command[0] === "GET") result = values.get(command[1]) || null;
      if (command[0] === "SET") values.set(command[1], command[2]);
      if (command[0] === "INCR") result = 1;
      if (command[0] === "EVAL") result = 1;
      return Response.json({ result });
    }
    if (url.endsWith(`/issues/${issueNumber}`) && method === "GET") return Response.json(issue);
    if (url.endsWith(`/issues/${issueNumber}`) && method === "PATCH") {
      issue.body = JSON.parse(init.body).body;
      return Response.json(issue);
    }
    if (url.endsWith(`/issues/${issueNumber}/assignees`) && method === "DELETE") {
      const requested = JSON.parse(init.body).assignees.map((login) => login.toLowerCase());
      issue.assignees = (issue.assignees || []).filter((item) => !requested.includes(item.login.toLowerCase()));
      return Response.json(issue);
    }
    throw new Error(`Unexpected request: ${method} ${url}`);
  };
}

test("claim rejects an issue model edited during metadata resolution", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.VOLUNTEER_MARKER_SECRET = "marker-secret";
  process.env.VOLUNTEER_WORKER_IMAGE_DIGEST = "sha256:image";
  process.env.EUROEVAL_VERSION = "1.0.0";
  process.env.VOLUNTEER_WORKER_VERSION = "1.0.0";
  process.env.UPSTASH_REDIS_REST_URL = "https://redis.test";
  process.env.UPSTASH_REDIS_REST_TOKEN = "redis-token";
  process.env.GITHUB_TOKEN = "github-token";
  process.env.VOLUNTEER_SCOPE_POLICY_JSON = JSON.stringify({ policy_version: "test-policy", policies: [{
    euroeval_version: "1.0.0", model_type: "encoder", language: "da", language_group: "da",
    identity_suffixes: [JSON.stringify(["race", false, true])], count: 1,
    task_groups: ["sequence_classification"], warnings: [],
  }] });
  const language = "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)";
  const issue = { number: issueNumber, title: "[MODEL EVALUATION REQUEST] org/model",
    body: `### Model ID\n\norg/model\n\n- [x] ${language}\n`, state: "open", assignees: [] };
  const credentialKey = `euroeval:worker:credential:${await sha256("credential")}`;
  const redisCommands = [];
  let githubMutations = 0;
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input); const requestMethod = init.method || "GET";
    if (url === "https://redis.test") {
      const command = JSON.parse(init.body); redisCommands.push(command);
      if (command[0] === "GET") return Response.json({ result: command[1] === credentialKey ? JSON.stringify({ contributor: "alice" }) : null });
      if (command[0] === "INCR") return Response.json({ result: 1 });
      return Response.json({ result: command[0] === "SET" ? "OK" : 1 });
    }
    if (url.includes("/repos/EuroEval/EuroEval/assignees/")) return new Response(null, { status: 204 });
    if (url.includes("/repos/EuroEval/EuroEval/issues?")) return Response.json([issue]);
    if (url.includes("huggingface.co/api/models/org/model")) {
      issue.title = "[MODEL EVALUATION REQUEST] org/edited";
      issue.body = issue.body.replace("org/model", "org/edited");
      return Response.json({ id: "org/model", sha: "a".repeat(40), private: false, gated: false,
        pipeline_tag: "fill-mask", siblings: [
        { rfilename: "model.safetensors", size: 100 }, { rfilename: "config.json", size: 10 },
      { rfilename: "tokenizer.json", size: 10 },
      ] });
    }
    if (url.includes(`/org/model/raw/${"a".repeat(40)}/config.json`)) {
      return Response.json({ model_type: "bert", architectures: ["BertModel"] });
    }
    if (url.endsWith(`/issues/${issueNumber}`) && requestMethod === "GET") return Response.json(issue);
    if (url.includes(`/issues/${issueNumber}`)) { githubMutations++; return Response.json(issue); }
    throw new Error(`Unexpected request: ${requestMethod} ${url}`);
  };
  try {
    const response = await claim(new Request("https://euroeval.test/api/worker/claim", {
      method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", worker_version: "1.0.0",
        hardware: { architecture: "amd64", free_disk_bytes: 1_000_000,
          gpu_memory_utilisation: 0.8, selected_gpu_index: 0, selected_gpu_uuid: "gpu",
          gpus: [{ index: 0, name: "GPU", uuid: "gpu", free_memory_bytes: 1_000_000,
            total_memory_bytes: 1_000_000 }] } }),
    }));
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), { protocol_version: "volunteer-worker/v1", status: "no_work" });
    assert.equal(githubMutations, 0);
    assert.equal(redisCommands.some((command) => command.some((item) =>
      typeof item === "string" && item.startsWith("euroeval:worker:lease"))), false);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

test("finalise rejects an issue edited before its locked transition", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.VOLUNTEER_MARKER_SECRET = "marker-secret";
  process.env.UPSTASH_REDIS_REST_URL = "https://redis.test";
  process.env.UPSTASH_REDIS_REST_TOKEN = "redis-token";
  process.env.GITHUB_TOKEN = "github-token";
  const activeLease = { ...lease, expires_at: new Date(Date.now() + 60_000).toISOString(),
    expected_scope: { policy_version: "test-policy", language_group: "Danish",
      identity_suffixes: [JSON.stringify(["dataset", false, true])], count: 1,
      task_groups: ["multiple_choice_classification"], warnings: [] } };
  const signed = await signVolunteerMarker(issueNumber, {
    protocol_version: "volunteer-worker/v1", coordinator: "coordinator", submission: "active",
    leases: [{ lease_id: activeLease.lease_id, language: activeLease.language,
      worker: activeLease.worker, contributor: activeLease.contributor,
      expires_at: activeLease.expires_at }],
  });
  const language = "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)";
  const issue = { number: issueNumber, title: "[MODEL EVALUATION REQUEST] org/model",
    body: `### Model ID\n\norg/model\n\n- [x] ${language}\n\n${markerBody(signed)}`,
    state: "open", assignees: [{ login: "alice" }], labels: [] };
  const receipt = { status: "manifest_uploaded", submission_id: activeLease.lease_id,
    lease: activeLease, entries: [{ digest: "digest",
      identity: JSON.stringify(["org/model", "dataset", false, true]), path: "result.json" }],
    manifest_path: "volunteer/manifests/lease.json" };
  const values = new Map();
  values.set(`euroeval:worker:credential:${await sha256("credential")}`, JSON.stringify({ contributor: "alice" }));
  values.set(`euroeval:worker:lease-id:${activeLease.lease_id}`, JSON.stringify(activeLease));
  values.set(`euroeval:worker:lease:${issueNumber}:da`, JSON.stringify(activeLease));
  values.set(`euroeval:worker:finalisation:${activeLease.lease_id}`, JSON.stringify(receipt));
  const broker = mockBroker(issue, values); let editedBody = "";
  globalThis.fetch = async (input, init = {}) => {
    if (String(input).endsWith(`/issues/${issueNumber}`) && (init.method || "GET") === "GET") {
      issue.body = issue.body.replace("org/model", "org/edited"); editedBody = issue.body;
    }
    return broker(input, init);
  };
  try {
    const response = await finalise(new Request("https://euroeval.test/api/worker/finalise", {
      method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", lease_id: activeLease.lease_id }),
    }));
    assert.equal(response.status, 409);
    assert.match((await response.json()).error, /different model/);
    assert.equal(issue.body, editedBody);
    assert.equal(JSON.parse(values.get(`euroeval:worker:lease-id:${activeLease.lease_id}`)).lease_id,
      activeLease.lease_id);
    assert.equal(JSON.parse(values.get(`euroeval:worker:finalisation:${activeLease.lease_id}`)).status,
      "manifest_uploaded");
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

test("heartbeat renews a full TTL and preserves a valid marker signature", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.VOLUNTEER_MARKER_SECRET = "marker-secret";
  process.env.VOLUNTEER_LEASE_SECONDS = "60";
  process.env.UPSTASH_REDIS_REST_URL = "https://redis.test";
  process.env.UPSTASH_REDIS_REST_TOKEN = "redis-token";
  process.env.GITHUB_TOKEN = "github-token";
  const signed = await signVolunteerMarker(issueNumber, {
    protocol_version: "volunteer-worker/v1", coordinator: "coordinator", submission: "active",
    leases: [{ lease_id: lease.lease_id, language: lease.language, worker: lease.worker,
      contributor: lease.contributor, expires_at: lease.expires_at }],
  });
  const issue = { number: issueNumber, body: markerBody(signed), state: "open", assignees: [{ login: "alice" }] };
  const values = new Map();
  values.set(`euroeval:worker:credential:${await sha256("credential")}`, JSON.stringify({ contributor: "alice" }));
  values.set(`euroeval:worker:lease-id:${lease.lease_id}`, JSON.stringify(lease));
  values.set(`euroeval:worker:lease:${issueNumber}:da`, JSON.stringify(lease));
  globalThis.fetch = mockBroker(issue, values);
  try {
    const response = await heartbeat(new Request("https://euroeval.test/api/worker/heartbeat", {
      method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", lease_id: lease.lease_id }),
    }));
    assert.equal(response.status, 200, await response.text());
    const renewed = parseVolunteerMarker(issue.body);
    assert.ok(renewed);
    assert.equal(await verifyVolunteerMarker(issueNumber, renewed), true);
    assert.ok(Date.parse(renewed.leases[0].expires_at) >= Date.now() + 59_000);
    assert.match(issue.body, /harmless comment/);
    assert.match(issue.body, /- \[x\] Greek/);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

test("release re-signs while preserving history and other leases", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.VOLUNTEER_MARKER_SECRET = "marker-secret";
  process.env.UPSTASH_REDIS_REST_URL = "https://redis.test";
  process.env.UPSTASH_REDIS_REST_TOKEN = "redis-token";
  process.env.GITHUB_TOKEN = "github-token";
  const other = { lease_id: "other", language: "de", worker: "worker-2", contributor: "bob",
    expires_at: new Date(Date.now() + 120_000).toISOString() };
  const history = { submission_id: "submission", language: "fr", manifest_path: "manifest",
    submitted_at: "2026-09-06T10:00:00Z", verified_contributor: "carol", result_count: 2, status: "rejected" };
  const signed = await signVolunteerMarker(issueNumber, {
    protocol_version: "volunteer-worker/v1", coordinator: "coordinator", submission: "submitted",
    leases: [{ lease_id: lease.lease_id, language: lease.language, worker: lease.worker,
      contributor: lease.contributor, expires_at: lease.expires_at }, other],
    submissions: [history], completed_languages: [],
  });
  const issue = { number: issueNumber, body: markerBody(signed), state: "open", assignees: [{ login: "alice" }] };
  const values = new Map();
  values.set(`euroeval:worker:credential:${await sha256("credential")}`, JSON.stringify({ contributor: "alice" }));
  values.set(`euroeval:worker:lease-id:${lease.lease_id}`, JSON.stringify(lease));
  values.set(`euroeval:worker:lease:${issueNumber}:da`, JSON.stringify(lease));
  globalThis.fetch = mockBroker(issue, values);
  try {
    const response = await release(new Request("https://euroeval.test/api/worker/release", {
      method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", lease_id: lease.lease_id }),
    }));
    assert.equal(response.status, 200, await response.text());
    const released = parseVolunteerMarker(issue.body);
    assert.ok(released);
    assert.equal(await verifyVolunteerMarker(issueNumber, released), true);
    assert.deepEqual(released.submissions, [history]);
    assert.deepEqual(released.leases.map((item) => item.lease_id), ["other"]);
    assert.match(issue.body, /harmless comment/);
    assert.match(issue.body, /- \[x\] Greek/);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});
