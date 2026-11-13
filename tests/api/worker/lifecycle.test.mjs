import test from "node:test";
import assert from "node:assert/strict";
import claim from "../../../api/worker/claim.ts";
import { fetch as finalise, validateResultScope } from "../../../api/worker/finalise.ts";
import heartbeat from "../../../api/worker/heartbeat.ts";
import release from "../../../api/worker/release.ts";
import { parseVolunteerMarker, signVolunteerMarker, sha256, verifyVolunteerMarker, reconcilePublicAssignments } from "../../../api/worker/_lib.ts";

const issueNumber = 12;

test("ignored GitHub assignee requests do not break signed public lease ownership", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.GITHUB_TOKEN = "token";
  process.env.VOLUNTEER_MARKER_SECRET = "marker-secret";
  const marker = await signVolunteerMarker(issueNumber, {
    protocol_version: "volunteer-worker/v1", coordinator: "coordinator", submission: "active",
    leases: [{ lease_id: "public", language: "da", worker: "worker", contributor: "ineligible-user",
      expires_at: new Date(Date.now() + 60_000).toISOString(), ownership_mode: "public" }],
  });
  const issue = { number: issueNumber, title: "request", body: markerBody(marker), state: "open", assignees: [] };
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input);
    if (url.endsWith(`/issues/${issueNumber}/assignees`) && init.method === "POST") return Response.json(issue);
    if (url.endsWith(`/issues/${issueNumber}`)) return Response.json(issue);
    throw new Error(`Unexpected request: ${init.method || "GET"} ${url}`);
  };
  try {
    const reconciled = await reconcilePublicAssignments(issue, marker, async () => {});
    assert.deepEqual(reconciled.marker.leases.map(({ contributor }) => contributor), ["ineligible-user"]);
    assert.deepEqual(reconciled.marker.public_assignees, undefined);
    assert.deepEqual(reconciled.issue.assignees, []);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

test("reconciles verified public contributors as broker-tracked assignees", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.GITHUB_TOKEN = "token";
  process.env.VOLUNTEER_MARKER_SECRET = "marker-secret";
  const marker = await signVolunteerMarker(issueNumber, {
    protocol_version: "volunteer-worker/v1", coordinator: "coordinator", submission: "active",
    leases: [{ lease_id: "public", language: "da", worker: "worker", contributor: "alice",
      expires_at: new Date(Date.now() + 60_000).toISOString(), ownership_mode: "public" }],
  });
  const issue = { number: issueNumber, title: "request", body: markerBody(marker), state: "open", assignees: [] };
  assert.equal(await verifyVolunteerMarker(issueNumber, parseVolunteerMarker(issue.body)), true);
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input);
    if (url.endsWith(`/issues/${issueNumber}/assignees`) && init.method === "POST") {
      issue.assignees = [{ login: "alice" }];
      return Response.json(issue);
    }
    if (url.endsWith(`/issues/${issueNumber}`) && init.method === "PATCH") {
      issue.body = JSON.parse(init.body).body;
      return Response.json(issue);
    }
    if (url.endsWith(`/issues/${issueNumber}`)) return Response.json(issue);
    throw new Error(`Unexpected request: ${init.method || "GET"} ${url}`);
  };
  try {
    const reconciled = await reconcilePublicAssignments(issue, marker, async () => {});
    assert.deepEqual(reconciled.issue.assignees, [{ login: "alice" }]);
    assert.deepEqual(reconciled.marker.public_assignees, ["alice"]);
    assert.equal(await verifyVolunteerMarker(issueNumber, reconciled.marker), true);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

test("claim rejects an unsupported configured EuroEval version before queue discovery", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  Object.assign(process.env, {
    VOLUNTEER_MARKER_SECRET: "marker-secret", VOLUNTEER_WORKER_IMAGE_DIGEST: "sha256:image",
    EUROEVAL_VERSION: "18.3.0", VOLUNTEER_WORKER_VERSION: "1.0.0",
    UPSTASH_REDIS_REST_URL: "https://redis.test", UPSTASH_REDIS_REST_TOKEN: "redis-token",
    GITHUB_TOKEN: "github-token", VOLUNTEER_SCOPE_POLICY_JSON: JSON.stringify({
      policy_version: "test-policy", policies: [{ euroeval_version: "18.4.0", model_type: "encoder",
        language: "da", language_group: "Danish", allowed_identity_suffix_sets: [["test"]],
        task_groups: ["classification"], warnings: [] }],
    }),
  });
  const credentialKey = `euroeval:worker:credential:${await sha256("credential")}`;
  let queueDiscoveryCalls = 0;
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input);
    if (url === "https://redis.test") {
      const command = JSON.parse(init.body);
      return Response.json({ result: command[1] === credentialKey
        ? JSON.stringify({ contributor: "alice" }) : command[0] === "SET" ? "OK" : 1 });
    }
    if (url.includes("/issues?")) queueDiscoveryCalls++;
    throw new Error(`Unexpected request: ${init.method || "GET"} ${url}`);
  };
  try {
    const response = await claim(new Request("https://euroeval.test/api/worker/claim", {
      method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", worker_version: "1.0.0",
        hardware: { architecture: "amd64", free_disk_bytes: 1, gpu_memory_utilisation: 0.8, selected_gpu_index: 0,
          selected_gpu_uuid: "gpu", gpus: [{ index: 0, name: "GPU", uuid: "gpu",
            free_memory_bytes: 1, total_memory_bytes: 1 }] } }),
    }));
    assert.equal(response.status, 503);
    assert.match((await response.json()).error, /configuration/i);
    assert.equal(queueDiscoveryCalls, 0);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});
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
      if (command[0] === "SMEMBERS") result = values.get(command[1]) || [];
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
    if (url.endsWith(`/issues/${issueNumber}/labels`) && method === "POST") {
      const labels = JSON.parse(init.body).labels.map((name) => ({ name }));
      issue.labels = [...(issue.labels || []), ...labels];
      return Response.json(issue.labels);
    }
    if (url.includes(`/issues/${issueNumber}/comments`) && method === "GET") {
      return Response.json(issue.comments || []);
    }
    if (url.endsWith(`/issues/${issueNumber}/comments`) && method === "POST") {
      issue.comments = [...(issue.comments || []), { body: JSON.parse(init.body).body }];
      return Response.json(issue.comments.at(-1));
    }
    throw new Error(`Unexpected request: ${method} ${url}`);
  };
}

for (const upstreamSlow of [false, true]) test(
  `claim scans 148 candidates without GPU allocation${upstreamSlow ? " and bounds slow Hub retries" : ""}`,
  async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  Object.assign(process.env, { VOLUNTEER_MARKER_SECRET: "marker-secret",
    VOLUNTEER_WORKER_IMAGE_DIGEST: "sha256:image", EUROEVAL_VERSION: "1.0.0",
    VOLUNTEER_WORKER_VERSION: "1.0.0", UPSTASH_REDIS_REST_URL: "https://redis.test",
    UPSTASH_REDIS_REST_TOKEN: "redis-token", GITHUB_TOKEN: "github-token",
    VOLUNTEER_SCOPE_POLICY_JSON: JSON.stringify({ policy_version: "test-policy", policies: [
      { euroeval_version: "1.0.0", model_type: "encoder", language: "da", language_group: "Danish",
        allowed_identity_suffix_sets: [["test"]], task_groups: ["classification"], warnings: [] },
    ] }), });
  const credentialKey = `euroeval:worker:credential:${await sha256("credential")}`;
  const issues = Array.from({ length: 148 }, (_, index) => ({
    number: index + 1, title: `[MODEL EVALUATION REQUEST] org/model-${index}`,
    body: `### Model ID\\n\\norg/model-${index}\\n\\n- [x] Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)`,
    state: "open", assignees: [],
  }));
  let inFlight = 0;
  let maxInFlight = 0;
  let metadataCalls = 0;
  let configCalls = 0;
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input);
    if (url === "https://redis.test") {
      const command = JSON.parse(init.body);
      if (command[0] === "GET") return Response.json({ result: command[1] === credentialKey
        ? JSON.stringify({ contributor: "alice" }) : null });
      return Response.json({ result: command[0] === "SET" ? "OK" : 1 });
    }
    if (url.includes("/repos/EuroEval/EuroEval/issues?")) {
      const page = Number(new URL(url).searchParams.get("page"));
      return Response.json(issues.slice((page - 1) * 100, page * 100));
    }
    if (url.includes("huggingface.co/api/models/")) {
      metadataCalls++;
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((resolve) => setTimeout(resolve, 2));
      inFlight--;
      const id = url.split("/api/models/")[1].split("?")[0];
      if (upstreamSlow && id === "org/model-0") {
        return new Response(null, { status: 429, headers: { "retry-after": "30" } });
      }
      return Response.json({ id, private: false, gated: false, siblings: [
        { rfilename: "model.safetensors", size: 100 }, { rfilename: "config.json", size: 10 },
      ] });
    }
    if (url.includes("/raw/") && url.endsWith("/config.json")) configCalls++;
    throw new Error(`Unexpected request: ${init.method || "GET"} ${url}`);
  };
  try {
    const started = Date.now();
    const response = await claim(new Request("https://euroeval.test/api/worker/claim", {
      method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", worker_version: "1.0.0",
        hardware: { architecture: "amd64", free_disk_bytes: 0, gpu_memory_utilisation: 0.8,
          selected_gpu_index: 0, selected_gpu_uuid: "gpu", gpus: [{ index: 0, name: "GPU", uuid: "gpu",
            free_memory_bytes: 1_000_000, total_memory_bytes: 1_000_000 }] } }),
    }));
    assert.equal(response.status, upstreamSlow ? 503 : 200);
    const result = await response.json();
    if (upstreamSlow) {
      assert.match(result.error, /retry claim/i);
      assert.ok(Date.now() - started < 5_000, "slow Hub retry exceeded the request budget");
    } else {
      assert.deepEqual(result, { protocol_version: "volunteer-worker/v1", status: "no_work" });
    }
    assert.ok(metadataCalls >= 148);
    assert.ok(maxInFlight <= 6);
    assert.equal(configCalls, 0);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

for (const transientConfigError of [false, true]) test(
  `claim ${transientConfigError ? "retries transient config failure" : "rejects model edited during metadata resolution"}`,
  async () => {
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
    allowed_identity_suffix_sets: [[JSON.stringify(["race", false, true])]],
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
      if (!transientConfigError) {
        issue.title = "[MODEL EVALUATION REQUEST] org/edited";
        issue.body = issue.body.replace("org/model", "org/edited");
      }
      return Response.json({ id: "org/model", sha: "a".repeat(40), private: false, gated: false,
        pipeline_tag: "fill-mask", siblings: [
        { rfilename: "model.safetensors", size: 100 }, { rfilename: "config.json", size: 10 },
      { rfilename: "tokenizer.json", size: 10 },
      ] });
    }
    if (url.includes(`/org/model/raw/${"a".repeat(40)}/config.json`)) {
      return transientConfigError ? new Response(null, { status: 429 }) :
        Response.json({ model_type: "bert", architectures: ["BertModel"] });
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
    assert.equal(response.status, transientConfigError ? 503 : 200);
    const result = await response.json();
    if (transientConfigError) assert.match(result.error, /retry claim/i);
    else assert.deepEqual(result, { protocol_version: "volunteer-worker/v1", status: "no_work" });
    assert.equal(githubMutations, 0);
    assert.equal(redisCommands.some((command) => command.some((item) =>
      typeof item === "string" && item.startsWith("euroeval:worker:lease"))), false);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

for (const stalledOperation of ["marker PATCH", "lease response"]) test(
  `claim bounds a stalled ${stalledOperation} and cleans its Redis lease`, async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  Object.assign(process.env, { VOLUNTEER_MARKER_SECRET: "marker-secret",
    VOLUNTEER_WORKER_IMAGE_DIGEST: "sha256:image", EUROEVAL_VERSION: "1.0.0",
    VOLUNTEER_WORKER_VERSION: "1.0.0", UPSTASH_REDIS_REST_URL: "https://redis.test",
    UPSTASH_REDIS_REST_TOKEN: "redis-token", GITHUB_TOKEN: "github-token",
    VOLUNTEER_SCOPE_POLICY_JSON: JSON.stringify({ policy_version: "test-policy", policies: [{
      euroeval_version: "1.0.0", model_type: "encoder", language: "da", language_group: "da",
      allowed_identity_suffix_sets: [[JSON.stringify(["recovered", false, true])]],
      task_groups: ["sequence_classification"], warnings: [],
    }] }) });
  const language = "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)";
  const issue = { number: issueNumber, title: "[MODEL EVALUATION REQUEST] org/model",
    body: `### Model ID\n\norg/model\n\n- [x] ${language}\n`, state: "open", assignees: [] };
  const credentialKey = `euroeval:worker:credential:${await sha256("credential")}`;
  const values = new Map([[credentialKey, JSON.stringify({ contributor: "alice" })]]);
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input); const method = init.method || "GET";
    if (url.includes("/buckets/EuroEval/results/resolve/")) return new Response(null, { status: 404 });
    if (url === "https://redis.test") {
      const command = JSON.parse(init.body);
      if (command[0] === "GET") return Response.json({ result: values.get(command[1]) || null });
      if (command[0] === "INCR") return Response.json({ result: 1 });
      if (command[0] === "EVAL" && command[1].includes("local issue=redis.call")) {
        values.set(command[3], command[5]); values.set(command[4], command[5]);
        if (stalledOperation === "lease response") {
          await new Promise((resolve, reject) => {
            const timer = setTimeout(resolve, 30_000);
            init.signal?.addEventListener("abort", () => { clearTimeout(timer); reject(init.signal.reason); }, { once: true });
          });
        }
        return Response.json({ result: 1 });
      }
      if (command[0] === "EVAL" && command[1].includes("released=true")) {
        values.delete(command[3]); values.delete(command[4]);
        return Response.json({ result: 1 });
      }
      return Response.json({ result: command[0] === "SET" ? "OK" : 1 });
    }
    if (url.includes("/repos/EuroEval/EuroEval/issues?")) return Response.json([issue]);
    if (url.includes("huggingface.co/api/models/org/model")) return Response.json({ id: "org/model",
      sha: "a".repeat(40), private: false, gated: false, pipeline_tag: "fill-mask",
      siblings: [{ rfilename: "model.safetensors", size: 100 }, { rfilename: "config.json", size: 10 }] });
    if (url.includes(`/org/model/raw/${"a".repeat(40)}/config.json`)) {
      return Response.json({ model_type: "bert", architectures: ["BertModel"] });
    }
    if (url.endsWith(`/issues/${issueNumber}`) && method === "GET") return Response.json(issue);
    if (url.endsWith(`/issues/${issueNumber}`) && method === "PATCH") {
      if (stalledOperation === "marker PATCH") {
        await new Promise((resolve, reject) => {
          const timer = setTimeout(resolve, 30_000);
          init.signal?.addEventListener("abort", () => { clearTimeout(timer); reject(init.signal.reason); }, { once: true });
        });
      }
      issue.body = JSON.parse(init.body).body;
      return Response.json(issue);
    }
    throw new Error(`Unexpected request: ${method} ${url}`);
  };
  try {
    const started = Date.now();
    const response = await claim(new Request("https://euroeval.test/api/worker/claim", {
      method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", worker_version: "1.0.0",
        hardware: { architecture: "amd64", free_disk_bytes: 1_000_000, gpu_memory_utilisation: 0.8,
          selected_gpu_index: 0, selected_gpu_uuid: "gpu", gpus: [{ index: 0, name: "GPU", uuid: "gpu",
            free_memory_bytes: 1_000_000, total_memory_bytes: 1_000_000 }] } }),
    }));
    assert.equal(response.status, 503, JSON.stringify({ body: await response.clone().json(), keys: [...values.keys()] }));
    assert.match((await response.json()).error, /retry claim/i);
    assert.ok(Date.now() - started < 8_000, "stalled claim exceeded its cleanup budget");
    assert.equal([...values.keys()].some((key) => key.startsWith("euroeval:worker:lease:")), false);
    assert.equal(parseVolunteerMarker(issue.body), null);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

for (const lateCandidate of [false, true]) test(
  `claim recovers expired ownership${lateCandidate ? " while scanning page two" : " alongside an active public lease"}`,
  async () => {
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
    allowed_identity_suffix_sets: [[JSON.stringify(["recovered", false, true])]],
    task_groups: ["sequence_classification"], warnings: [],
  }] });
  const language = "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)";
  const expired = { ...lease, expires_at: "2020-01-01T00:00:00.000Z" };
  const publicLease = { lease_id: "public-lease", language: "fr", worker: "public-worker",
    contributor: "alice", expires_at: new Date(Date.now() + 120_000).toISOString(), ownership_mode: "public" };
  const signed = await signVolunteerMarker(issueNumber, {
    protocol_version: "volunteer-worker/v1", coordinator: "coordinator", submission: "active",
    leases: [{ lease_id: expired.lease_id, language: expired.language, worker: expired.worker,
      contributor: expired.contributor, expires_at: expired.expires_at }, publicLease],
    submissions: [{ submission_id: "public-submission", language: "fr", manifest_path: "manifest",
      submitted_at: "2026-09-06T10:00:00Z", verified_contributor: "alice", result_count: 1,
      status: "accepted", ownership_mode: "public" }],
  });
  const issue = { number: issueNumber, title: "[MODEL EVALUATION REQUEST] org/model",
    body: `### Model ID\n\norg/model\n\n- [x] ${language}\n\n${markerBody(signed)}`,
    state: "open", assignees: [{ login: "alice" }], labels: [] };
  const competing = await signVolunteerMarker(issueNumber + 1, {
    protocol_version: "volunteer-worker/v1", coordinator: "coordinator", submission: "active",
    leases: [{ lease_id: "competing-lease", language: "da", worker: "another-worker",
      contributor: "another-user", expires_at: new Date(Date.now() + 120_000).toISOString(),
      ownership_mode: "public" }],
  });
  const otherIssue = { number: issueNumber + 1, title: issue.title,
    body: `### Model ID\n\norg/model\n\n- [x] ${language}\n\n<!-- euroeval-volunteer-worker:v1 ${JSON.stringify(competing)} -->`,
    state: "open", assignees: [], labels: [] };
  const laterIssues = Array.from({ length: 146 }, (_, index) => ({
    number: 1_000 + index, title: `[MODEL EVALUATION REQUEST] org/model-${index}`,
    body: `### Model ID\n\norg/model-${index}\n\n- [x] ${language}`,
    state: "open", assignees: [], labels: [],
  }));
  let competingClaim = true;
  let metadataCalls = 0;
  const credentialKey = `euroeval:worker:credential:${await sha256("credential")}`;
  const leaseKey = `euroeval:worker:lease:${issueNumber}:da`;
  const values = new Map([[credentialKey, JSON.stringify({ contributor: "alice" })],
    [leaseKey, JSON.stringify(expired)]]);
  const mutations = [];
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input); const method = init.method || "GET";
    if (url.includes("/buckets/EuroEval/results/resolve/")) return new Response(null, { status: 404 });
    if (url === "https://redis.test") {
      const command = JSON.parse(init.body);
      if (command[0] === "GET") return Response.json({ result: values.get(command[1]) || null });
      if (command[0] === "SMEMBERS") return Response.json({ result: [] });
      if (command[0] === "EVAL" && String(command[1]).includes("DEL")) {
        values.delete(leaseKey); values.delete(`euroeval:worker:lease-id:${expired.lease_id}`);
      }
      return Response.json({ result: command[0] === "SET" ? "OK" : 1 });
    }
    if (url.includes("/repos/EuroEval/EuroEval/assignees/")) return new Response(null, { status: 204 });
    if (url.includes("/repos/EuroEval/EuroEval/issues?")) {
      const page = Number(new URL(url).searchParams.get("page"));
      const listed = competingClaim ? [issue, otherIssue, ...laterIssues] :
        lateCandidate ? [...laterIssues, issue] : [issue, ...laterIssues];
      return Response.json(listed.slice((page - 1) * 100, page * 100));
    }
    if (url.includes("huggingface.co/api/models/org/model")) {
      metadataCalls++;
      return Response.json({
      id: "org/model", sha: "a".repeat(40), private: false, gated: false,
      pipeline_tag: "fill-mask", siblings: [{ rfilename: "model.safetensors", size: 100 }],
      });
    }
    if (url.includes(`/org/model/raw/${"a".repeat(40)}/config.json`)) {
      return Response.json({ model_type: "bert", architectures: ["BertModel"] });
    }
    if (url.endsWith(`/issues/${issueNumber}`) && method === "GET") return Response.json(issue);
    if (url.endsWith(`/issues/${issueNumber + 1}`) && method === "GET") return Response.json(otherIssue);
    if (url.endsWith(`/issues/${issueNumber}`) && method === "PATCH") {
      mutations.push("patch"); issue.body = JSON.parse(init.body).body; return Response.json(issue);
    }
    if (url.endsWith(`/issues/${issueNumber}/assignees`) && method === "DELETE") {
      mutations.push("unassign"); issue.assignees = []; return Response.json(issue);
    }
    if (url.endsWith(`/issues/${issueNumber}/assignees`) && method === "POST") {
      mutations.push("assign"); issue.assignees = [{ login: "alice" }]; return Response.json(issue);
    }
    throw new Error(`Unexpected request: ${method} ${url}`);
  };
  try {
    const claimRequest = () => new Request("https://euroeval.test/api/worker/claim", {
      method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", worker_version: "1.0.0",
        hardware: { architecture: "amd64", free_disk_bytes: 1_000_000,
          gpu_memory_utilisation: 0.8, selected_gpu_index: 0, selected_gpu_uuid: "gpu",
          gpus: [{ index: 0, name: "GPU", uuid: "gpu", free_memory_bytes: 1_000_000,
            total_memory_bytes: 1_000_000 }] } }),
    });
    const blocked = await claim(claimRequest());
    const blockedBody = await blocked.json();
    assert.equal(blockedBody.status, "no_work", JSON.stringify(blockedBody));
    assert.deepEqual(mutations, ["unassign", "patch", "assign", "patch"]);
    assert.ok(parseVolunteerMarker(issue.body).leases.every((item) => item.lease_id !== expired.lease_id));
    competingClaim = false;
    metadataCalls = 0;
    const response = await claim(claimRequest());
    assert.equal(response.status, 200, await response.text());
    if (lateCandidate) assert.ok(metadataCalls >= 147, `did not reach page-two issue (${metadataCalls} models)`);
    assert.deepEqual(mutations, ["unassign", "patch", "assign", "patch", "patch"]);
    assert.deepEqual(issue.assignees, [{ login: "alice" }]);
    assert.equal(parseVolunteerMarker(issue.body).leases.length, 2);
    assert.ok(parseVolunteerMarker(issue.body).leases.some((item) => item.lease_id === publicLease.lease_id));
    assert.ok(parseVolunteerMarker(issue.body).leases.every((item) => item.lease_id !== expired.lease_id));
    assert.equal(parseVolunteerMarker(issue.body).submissions[0].ownership_mode, "public");
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

for (const contention of ["global", "issue-only", "issue-later", "issue-later-arm64", "issue-later-aarch64", "issue-later-arm-fallback"]) test(
  `claim retries on ${contention} mutex contention${contention.startsWith("issue-later") ? " and claims a later candidate" : ""}`,
  async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  Object.assign(process.env, { VOLUNTEER_MARKER_SECRET: "marker-secret",
    VOLUNTEER_WORKER_IMAGE_DIGEST: "sha256:image", EUROEVAL_VERSION: "1.0.0",
    VOLUNTEER_WORKER_VERSION: "1.0.0", UPSTASH_REDIS_REST_URL: "https://redis.test",
    UPSTASH_REDIS_REST_TOKEN: "redis-token", GITHUB_TOKEN: "github-token",
    VOLUNTEER_SCOPE_POLICY_JSON: JSON.stringify({ policy_version: "test-policy", policies: [{
      euroeval_version: "1.0.0", model_type: "encoder", language: "da", language_group: "da",
      allowed_identity_suffix_sets: [[JSON.stringify(["race", false, true])]],
      task_groups: ["sequence_classification"], warnings: [],
    }] }) });
  const language = "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)";
  const issueNumbers = contention === "issue-only" ? [12] : [12, 13];
  const architecture = contention.includes("arm") ? "arm64" :
    contention.endsWith("aarch64") ? "aarch64" : "amd64";
  const expectedImageDigest = architecture === "amd64" || contention.endsWith("fallback")
    ? "sha256:image" : "sha256:arm-image";
  if (architecture !== "amd64" && !contention.endsWith("fallback")) {
    process.env.VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64 = expectedImageDigest;
  }
  const issues = issueNumbers.map((number) => ({ number, title: "[MODEL EVALUATION REQUEST] org/model",
    body: `### Model ID\n\norg/model\n\n- [x] ${language}\n`, state: "open", assignees: [] }));
  const credentialKey = `euroeval:worker:credential:${await sha256("credential")}`;
  const values = new Map([[credentialKey, JSON.stringify({ contributor: "alice" })]]);
  const mutations = [];
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input); const method = init.method || "GET";
    if (url.includes("/buckets/EuroEval/results/resolve/")) return new Response(null, { status: 404 });
    if (url === "https://redis.test") {
      const command = JSON.parse(init.body);
      if (command[0] === "GET") return Response.json({ result: values.get(command[1]) || null });
      if (command[0] === "SMEMBERS") return Response.json({ result: [] });
      if (command[0] === "INCR" || command[0] === "EXPIRE") return Response.json({ result: 1 });
      if (command[0] === "SET" && command[1] === "euroeval:worker:mutex:9007199254740991" && contention === "global") {
        return Response.json({ result: null });
      }
      if (command[0] === "SET" && command[1] === "euroeval:worker:mutex:12" && contention.startsWith("issue")) {
        return Response.json({ result: null });
      }
      if (command[0] === "SET") { values.set(command[1], command[2]); return Response.json({ result: "OK" }); }
      if (command[0] === "EVAL") {
        if (String(command[1]).includes("released=true")) values.delete(command[3]);
        else if (String(command[1]).includes("redis.call('SET'")) values.set(command[3], command[5]);
        return Response.json({ result: 1 });
      }
    }
    if (url.includes("/repos/EuroEval/EuroEval/assignees/")) return new Response(null, { status: 204 });
    if (url.includes("/repos/EuroEval/EuroEval/issues?")) return Response.json(issues);
    if (url.includes("huggingface.co/api/models/org/model")) return Response.json({ id: "org/model",
      sha: "a".repeat(40), private: false, gated: false, pipeline_tag: "fill-mask",
      siblings: [{ rfilename: "model.safetensors", size: 100 }, { rfilename: "config.json", size: 10 }] });
    if (url.includes(`/org/model/raw/${"a".repeat(40)}/config.json`)) {
      return Response.json({ model_type: "bert", architectures: ["BertModel"] });
    }
    const issue = issues.find((item) => url.endsWith(`/issues/${item.number}`));
    if (issue && method === "GET") return Response.json(issue);
    if (issue && method === "PATCH") {
      mutations.push(issue.number); issue.body = JSON.parse(init.body).body; return Response.json(issue);
    }
    if (url === "https://redis.test") {
      throw new Error(`Unexpected Redis command: ${JSON.parse(init.body)[0]}`);
    }
    throw new Error(`Unexpected request: ${method} ${url}`);
  };
  try {
    const makeRequest = () => new Request("https://euroeval.test/api/worker/claim", {
      method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", worker_version: "1.0.0",
        hardware: { architecture, free_disk_bytes: 1_000_000,
          gpu_memory_utilisation: 0.8, selected_gpu_index: 0, selected_gpu_uuid: "gpu",
          gpus: [{ index: 0, name: "GPU", uuid: "gpu", free_memory_bytes: 1_000_000,
            total_memory_bytes: 1_000_000 }] } }),
    });
    const response = await claim(makeRequest());
    if (!contention.startsWith("issue-later")) {
      assert.equal(response.status, 503, await response.clone().text());
      assert.deepEqual(await response.json(), { protocol_version: "volunteer-worker/v1",
        error: "Claim is busy; retry claim.", code: "claim_busy" });
      assert.deepEqual(mutations, []);
    } else {
      assert.equal(response.status, 200, await response.clone().text());
      const result = await response.json();
      assert.equal(result.issue_number, 13);
      assert.equal(result.image_digest, expectedImageDigest);
      assert.deepEqual(mutations, [13]);
    }
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

async function finaliseSuccessfulScope(language, suffixes, retryDigest, prevalidated = false) {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.VOLUNTEER_MARKER_SECRET = "marker-secret";
  process.env.UPSTASH_REDIS_REST_URL = "https://redis.test";
  process.env.UPSTASH_REDIS_REST_TOKEN = "redis-token";
  process.env.GITHUB_TOKEN = "github-token";
  process.env.HF_STAGING_BUCKET = "EuroEval/staging";
  process.env.HF_TOKEN = "hf_test-token";
  const languageGroup = "Romance languages (Catalan, French, Italian, Portuguese, Romanian, Spanish)";
  const activeLease = { ...lease, language, model_type: "generative", expected_scope: {
    policy_version: "test-policy", language_group: languageGroup,
    allowed_identity_suffix_sets: [suffixes], task_groups: ["multiple_choice_classification"], warnings: [],
  } };
  const signed = await signVolunteerMarker(issueNumber, {
    protocol_version: "volunteer-worker/v1", coordinator: "coordinator", submission: "active",
    leases: [{ lease_id: activeLease.lease_id, language: activeLease.language, worker: activeLease.worker,
      contributor: activeLease.contributor, expires_at: activeLease.expires_at }],
  });
  const issue = { number: issueNumber, title: "[MODEL EVALUATION REQUEST] org/model",
    body: `### Model ID\n\norg/model\n\n- [x] ${languageGroup}\n\n${markerBody(signed)}`,
    state: "open", assignees: [{ login: "alice" }], labels: [] };
  const entries = suffixes.map((item, index) => ({ digest: `digest-${index}`,
    identity: JSON.stringify(["org/model", ...JSON.parse(item)]), path: `result-${index}.json` }));
  if (retryDigest !== undefined) {
    entries.push({ ...entries[0], failed: 0, digest: retryDigest });
  }
  const receipt = { status: retryDigest === undefined || prevalidated ? "manifest_uploaded" : "validating",
    submission_id: activeLease.lease_id, lease: activeLease, entries,
    manifest_path: "volunteer/manifests/lease.json" };
  const values = new Map();
  values.set(`euroeval:worker:credential:${await sha256("credential")}`, JSON.stringify({ contributor: "alice" }));
  values.set(`euroeval:worker:lease-id:${activeLease.lease_id}`, JSON.stringify(activeLease));
  values.set(`euroeval:worker:lease:${issueNumber}:${activeLease.language}`, JSON.stringify(activeLease));
  values.set(`euroeval:worker:finalisation:${activeLease.lease_id}`, JSON.stringify(receipt));
  if (retryDigest !== undefined && !prevalidated) {
    values.set(`euroeval:worker:results:${activeLease.lease_id}`,
      entries.map((entry) => JSON.stringify(entry)));
  }
  const broker = mockBroker(issue, values);
  globalThis.fetch = async (input, init = {}) => {
    if (String(input).includes("huggingface.co/api/buckets/")) {
      return Response.json({ private: true });
    }
    if (String(input).includes("huggingface.co") || String(input).includes("/v1/")) {
      return Response.json({});
    }
    return broker(input, init);
  };
  try {
    const response = await finalise(new Request("https://euroeval.test/api/worker/finalise", {
      method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", lease_id: activeLease.lease_id }),
    }));
    return retryDigest !== undefined && !prevalidated ? { response, values } : response;
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
}

test("finalise deduplicates legacy retries without losing zero failure counts", async () => {
  const scope = [JSON.stringify(["multiloko-fr", null, true])];
  const matching = await finaliseSuccessfulScope("fr", scope, "digest-0");
  assert.equal(matching.response.status, 502);
  const stored = JSON.parse(matching.values.get(
    "euroeval:worker:finalisation:lease",
  ));
  assert.equal(stored.entries.length, 1);
  assert.equal(stored.entries[0].failed, 0);

  const response = await finaliseSuccessfulScope("fr", scope, "digest-0", true);
  assert.equal(response.status, 200, await response.text());

  const conflicting = await finaliseSuccessfulScope(
    "fr", [JSON.stringify(["multiloko-fr", null, true])], "different-digest",
  );
  assert.equal(conflicting.response.status, 409);
  assert.match((await conflicting.response.json()).error, /Conflicting results/);
});

test("finalise accepts a valid base alternative with a split-less dataset", async () => {
  const response = await finaliseSuccessfulScope("fr", [JSON.stringify(["multiloko-fr", null, true])]);
  assert.equal(response.status, 200);
  assert.equal((await response.json()).status, "ready");
});

test("finalise accepts a valid instruction-reasoning dual-mode alternative", async () => {
  const suffixes = [
    JSON.stringify(["ifeval-fr", null, null]),
    JSON.stringify(["multiloko-fr", null, false]),
    JSON.stringify(["multiloko-fr", null, true]),
  ];
  const response = await finaliseSuccessfulScope("fr", suffixes);
  assert.equal(response.status, 200);
  assert.equal((await response.json()).status, "ready");
});

test("scope validation excludes a required canary from an exact alternative", () => {
  const suffixes = [
    JSON.stringify(["ifeval-fr", null, null]),
    JSON.stringify(["multiloko-fr", null, false]),
    JSON.stringify(["multiloko-fr", null, true]),
  ];
  const activeLease = { ...lease, model_type: "generative",
    contamination_canary: { status: "required", protocol_version: "canary/v1",
      corpus_revision: "revision", corpus_sha256: "a".repeat(64) },
    expected_scope: { policy_version: "test-policy", language_group: "fr",
      allowed_identity_suffix_sets: [suffixes], task_groups: ["multiple_choice_classification"], warnings: [],
    } };
  const entries = suffixes.map((item, index) => ({ digest: `digest-${index}`,
    identity: JSON.stringify(["org/model", ...JSON.parse(item)]), path: `result-${index}.json` }));
  entries.push({ digest: "canary-digest",
    identity: JSON.stringify(["org/model", "contamination-canary-fr", null, null]),
    path: "canary-result.json" });

  assert.deepEqual(validateResultScope(entries, activeLease), suffixes);
  assert.throws(() => validateResultScope(entries.slice(0, -1), activeLease), /expected scope/);
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

test("finalise rejects unsupported HTTP methods", async () => {
  const response = await finalise(new Request("https://euroeval.test/api/worker/finalise", {
    method: "GET",
  }));

  assert.equal(response.status, 405);
  assert.deepEqual(await response.json(), { error: "Method not allowed" });
});

test("heartbeat reports issue mutex contention as retryable without hiding ownership conflicts", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.VOLUNTEER_MARKER_SECRET = "marker-secret";
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
  const broker = mockBroker(issue, values);
  let mutexBusy = true;
  globalThis.fetch = async (input, init = {}) => {
    if (String(input) === "https://redis.test") {
      const command = JSON.parse(init.body);
      if (mutexBusy && command[0] === "SET" && command[1] === `euroeval:worker:mutex:${issueNumber}`) {
        return Response.json({ result: null });
      }
    }
    return broker(input, init);
  };
  const request = () => new Request("https://euroeval.test/api/worker/heartbeat", {
    method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: "volunteer-worker/v1", lease_id: lease.lease_id }),
  });
  try {
    const busy = await heartbeat(request());
    assert.equal(busy.status, 503);
    assert.equal((await busy.json()).code, "issue_busy");
    mutexBusy = false;
    issue.assignees = [{ login: "bob" }];
    const ownershipConflict = await heartbeat(request());
    assert.equal(ownershipConflict.status, 409);
    assert.notEqual((await ownershipConflict.json()).code, "issue_busy");
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

test("release retries a released tombstone without touching GitHub", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.UPSTASH_REDIS_REST_URL = "https://redis.test";
  process.env.UPSTASH_REDIS_REST_TOKEN = "redis-token";
  const values = new Map();
  values.set(`euroeval:worker:credential:${await sha256("credential")}`, JSON.stringify({ contributor: "alice" }));
  values.set(`euroeval:worker:lease-id:${lease.lease_id}`, JSON.stringify({ ...lease, released: true }));
  let githubCalls = 0;
  globalThis.fetch = async (input, init = {}) => {
    const parsedUrl = new URL(String(input));
    if (parsedUrl.origin === "https://redis.test") {
      const command = JSON.parse(init.body);
      if (command[0] === "GET") return Response.json({ result: values.get(command[1]) || null });
      return Response.json({ result: 1 });
    }
    githubCalls += 1;
    throw new Error("GitHub must not be contacted for a released tombstone");
  };
  try {
    const response = await release(new Request("https://euroeval.test/api/worker/release", {
      method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", lease_id: lease.lease_id }),
    }));
    assert.equal(response.status, 200, await response.text());
    assert.equal(githubCalls, 0);
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
  const issue = { number: issueNumber, body: markerBody(signed), state: "open", assignees: [{ login: "alice" }, { login: "bob" }] };
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
    assert.deepEqual(issue.assignees, [{ login: "bob" }]);
    assert.match(issue.body, /harmless comment/);
    assert.match(issue.body, /- \[x\] Greek/);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

test("release unassigns legacy ownership despite matching public lease and submission", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.VOLUNTEER_MARKER_SECRET = "marker-secret";
  process.env.UPSTASH_REDIS_REST_URL = "https://redis.test";
  process.env.UPSTASH_REDIS_REST_TOKEN = "redis-token";
  process.env.GITHUB_TOKEN = "github-token";
  const publicLease = { lease_id: "public-lease", language: "fr", worker: "public-worker",
    contributor: "alice", expires_at: new Date(Date.now() + 120_000).toISOString(), ownership_mode: "public" };
  const publicSubmission = { submission_id: "public-submission", language: "fr", manifest_path: "manifest",
    submitted_at: "2026-09-06T10:00:00Z", verified_contributor: "alice", result_count: 1,
    status: "accepted", ownership_mode: "public" };
  const signed = await signVolunteerMarker(issueNumber, {
    protocol_version: "volunteer-worker/v1", coordinator: "coordinator", submission: "submitted",
    leases: [{ lease_id: lease.lease_id, language: lease.language, worker: lease.worker,
      contributor: lease.contributor, expires_at: lease.expires_at }, publicLease],
    submissions: [publicSubmission], completed_languages: [],
  });
  const issue = { number: issueNumber, body: markerBody(signed), state: "open", assignees: [{ login: "alice" }] };
  const values = new Map();
  values.set(`euroeval:worker:credential:${await sha256("credential")}`, JSON.stringify({ contributor: "alice" }));
  values.set(`euroeval:worker:lease-id:${lease.lease_id}`, JSON.stringify(lease));
  values.set(`euroeval:worker:lease:${issueNumber}:da`, JSON.stringify(lease));
  const mutations = [];
  const broker = mockBroker(issue, values);
  globalThis.fetch = async (input, init = {}) => {
    if (new URL(String(input)).origin === "https://api.github.com" && (init.method || "GET") !== "GET") {
      mutations.push((init.method || "GET") + (String(input).includes("/assignees") ? ":assignee" : ":issue"));
    }
    return broker(input, init);
  };
  try {
    const response = await release(new Request("https://euroeval.test/api/worker/release", {
      method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", lease_id: lease.lease_id }),
    }));
    assert.equal(response.status, 200, await response.text());
    assert.deepEqual(issue.assignees, []);
    const released = parseVolunteerMarker(issue.body);
    assert.deepEqual(released.leases, [publicLease]);
    assert.deepEqual(released.submissions, [publicSubmission]);
    assert.ok(mutations.includes("DELETE:assignee"));
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

test("release resumes after marker removal with retained ownership", async () => {
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
    leases: [other], submissions: [history], completed_languages: [],
  });
  const issue = { number: issueNumber, body: markerBody(signed), state: "open", assignees: [{ login: "bob" }] };
  const values = new Map();
  values.set(`euroeval:worker:credential:${await sha256("credential")}`, JSON.stringify({ contributor: "alice" }));
  values.set(`euroeval:worker:lease-id:${lease.lease_id}`, JSON.stringify(lease));
  const broker = mockBroker(issue, values);
  const redisCommands = [];
  const githubMutations = [];
  globalThis.fetch = async (input, init = {}) => {
    const method = init.method || "GET";
    const parsedUrl = new URL(String(input));
    if (parsedUrl.origin === "https://redis.test") redisCommands.push(JSON.parse(init.body));
    if (parsedUrl.origin === "https://api.github.com" && method !== "GET") githubMutations.push(method);
    return broker(input, init);
  };
  try {
    const response = await release(new Request("https://euroeval.test/api/worker/release", {
      method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", lease_id: lease.lease_id }),
    }));
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      protocol_version: "volunteer-worker/v1", status: "released", lease_id: lease.lease_id,
    });
    assert.deepEqual(githubMutations, []);
    assert.ok(redisCommands.some((command) => command[0] === "EVAL" &&
      String(command[1]).includes("released=true")));
    assert.deepEqual(parseVolunteerMarker(issue.body).leases.map((item) => item.lease_id), ["other"]);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

test("claim persists record-revision compatibility from client capabilities", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  Object.assign(process.env, { VOLUNTEER_MARKER_SECRET: "marker-secret",
    VOLUNTEER_WORKER_IMAGE_DIGEST: "sha256:image", EUROEVAL_VERSION: "1.0.0",
    VOLUNTEER_WORKER_VERSION: "1.0.0", UPSTASH_REDIS_REST_URL: "https://redis.test",
    UPSTASH_REDIS_REST_TOKEN: "redis-token", GITHUB_TOKEN: "github-token",
    VOLUNTEER_SCOPE_POLICY_JSON: JSON.stringify({ policy_version: "test-policy", policies: [{
      euroeval_version: "1.0.0", model_type: "encoder", language: "da", language_group: "da",
      allowed_identity_suffix_sets: [['["dataset",true,true]']],
      task_groups: ["sequence_classification"], warnings: [],
    }] }) });
  const language = "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)";
  const issue = { number: issueNumber, title: "[MODEL EVALUATION REQUEST] org/model",
    body: `### Model ID\n\norg/model\n\n- [x] ${language}\n`, state: "open", assignees: [] };
  const credentialKey = `euroeval:worker:credential:${await sha256("credential")}`;
  const values = new Map([[credentialKey, JSON.stringify({ contributor: "alice" })]]);
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input); const method = init.method || "GET";
    if (url.includes("/buckets/EuroEval/results/resolve/")) return new Response(null, { status: 404 });
    if (url === "https://redis.test") {
      const command = JSON.parse(init.body);
      if (command[0] === "GET") return Response.json({ result: values.get(command[1]) || null });
      if (command[0] === "SMEMBERS") return Response.json({ result: [] });
      if (command[0] === "SET") { values.set(command[1], command[2]); return Response.json({ result: "OK" }); }
      if (command[0] === "EVAL" && command[1].includes("local issue=redis.call")) {
        values.set(command[3], command[5]); values.set(command[4], command[5]);
      }
      return Response.json({ result: 1 });
    }
    if (url.includes("/repos/EuroEval/EuroEval/assignees/")) return new Response(null, { status: 204 });
    if (url.includes("/repos/EuroEval/EuroEval/issues?")) return Response.json([issue]);
    if (url.includes("huggingface.co/api/models/org/model")) return Response.json({ id: "org/model",
      sha: "a".repeat(40), private: false, gated: false, pipeline_tag: "fill-mask",
      siblings: [{ rfilename: "model.safetensors", size: 100 }, { rfilename: "config.json", size: 10 }] });
    if (url.includes(`/org/model/raw/${"a".repeat(40)}/config.json`)) {
      return Response.json({ model_type: "bert", architectures: ["BertModel"] });
    }
    if (url.endsWith(`/issues/${issueNumber}`) && method === "GET") return Response.json(issue);
    if (url.endsWith(`/issues/${issueNumber}`) && method === "PATCH") {
      issue.body = JSON.parse(init.body).body; return Response.json(issue);
    }
    throw new Error(`Unexpected request: ${method} ${url}`);
  };
  try {
    const makeRequest = (capabilities) => new Request("https://euroeval.test/api/worker/claim", {
      method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", worker_version: "1.0.0",
        ...(capabilities === undefined ? {} : { capabilities }), hardware: { architecture: "amd64",
          free_disk_bytes: 1_000_000, gpu_memory_utilisation: 0.8, selected_gpu_index: 0,
          selected_gpu_uuid: "gpu", gpus: [{ index: 0, name: "GPU", uuid: "gpu",
            free_memory_bytes: 1_000_000, total_memory_bytes: 1_000_000 }] } }),
    });
    const legacyResponse = await claim(makeRequest(undefined));
    assert.equal(legacyResponse.status, 200, await legacyResponse.clone().text());
    const legacy = await legacyResponse.json();
    assert.equal(legacy.issue_number, issueNumber);
    const legacyLease = JSON.parse(values.get(`euroeval:worker:lease:${issueNumber}:da`));
    assert.equal(legacyLease.lease_id, legacy.lease_id);
    assert.equal(legacyLease.record_revision_required, false);

    values.delete(`euroeval:worker:lease:${issueNumber}:da`);
    values.delete(`euroeval:worker:lease-id:${legacy.lease_id}`);
    issue.body = `### Model ID\n\norg/model\n\n- [x] ${language}\n`;
    const modernResponse = await claim(makeRequest(["canonical_coverage_v1"]));
    assert.equal(modernResponse.status, 200, await modernResponse.clone().text());
    const modern = await modernResponse.json();
    assert.equal(modern.issue_number, issueNumber);
    const modernLease = JSON.parse(values.get(`euroeval:worker:lease:${issueNumber}:da`));
    assert.equal(modernLease.lease_id, modern.lease_id);
    assert.equal(modernLease.record_revision_required, true);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});
