import test from "node:test";
import assert from "node:assert/strict";
import heartbeat from "./heartbeat.ts";
import release from "./release.ts";
import { parseVolunteerMarker, signVolunteerMarker, sha256, verifyVolunteerMarker } from "./_lib.ts";

const issueNumber = 12;
const lease = {
  issue_number: issueNumber, language: "da", worker: "worker", contributor: "alice",
  model_id: "org/model", model_revision: "revision", euroeval_version: "1.0.0",
  image_digest: "sha256:image", worker_version: "1.0.0", gpu_memory_utilisation: 0.8,
  expires_at: new Date(Date.now() + 60_000).toISOString(), lease_id: "lease",
  model_profile: "bert", expected_scope: { language_group: "Danish", identity_suffixes: ["x"] },
};

function markerBody(marker) {
  return `request\n\n<!-- euroeval-volunteer-worker:v1 ${JSON.stringify(marker)} -->\n`;
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
    throw new Error(`Unexpected request: ${method} ${url}`);
  };
}

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
  const issue = { number: issueNumber, body: markerBody(signed), state: "open", assignees: [] };
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
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

test("release re-signs while preserving history and other leases", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.VOLUNTEER_MARKER_SECRET = "marker-secret";
  process.env.WORKER_COORDINATOR_LOGIN = "coordinator";
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
  const issue = { number: issueNumber, body: markerBody(signed), state: "open", assignees: [] };
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
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});
