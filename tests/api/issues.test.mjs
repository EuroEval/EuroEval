import test from "node:test";
import assert from "node:assert/strict";
import issues from "../../api/issues.ts";
import { signVolunteerMarker, parseVolunteerMarker, issueLeaseKey } from "../../api/worker/_lib.ts";
import { toQueueEntry } from "../../src/frontend/services/github.ts";
import { redisFixture } from "./redis-rest-fixture.mjs";

for (const scenario of ["expired", "renewed", "replacement", "manual-and-healthy", "bad-signature", "redis-down", "delete-failure", "lost-delete-response", "delete-failure-with-healthy"]) {
  test(`queue refresh reconciles ownership safely: ${scenario}`, async (t) => {
    const originalEnv = { ...process.env };
    const originalFetch = globalThis.fetch;
    t.after(() => { process.env = originalEnv; globalThis.fetch = originalFetch; });
    Object.assign(process.env, {
      GITHUB_TOKEN: "fake-token", VOLUNTEER_MARKER_SECRET: "fake-secret",
      UPSTASH_REDIS_REST_URL: "https://redis.test", UPSTASH_REDIS_REST_TOKEN: "fake-redis-token",
    });
    const redis = await redisFixture(t);
    const expired = { lease_id: "old", language: "da", worker: "worker", contributor: "alice", ownership_mode: "public", expires_at: new Date(Date.now() - 60_000).toISOString() };
    const healthy = { ...expired, lease_id: "other", language: "en", contributor: "bob", expires_at: new Date(Date.now() + 600_000).toISOString() };
    const hasHealthyOwner = ["manual-and-healthy", "delete-failure-with-healthy"].includes(scenario);
    const marker = await signVolunteerMarker(12, {
      protocol_version: "volunteer-worker/v1", coordinator: "coordinator", submission: "active",
      leases: hasHealthyOwner ? [expired, healthy] : [expired],
      public_assignees: hasHealthyOwner ? ["alice", "bob"] : ["alice"],
      completed_languages: ["fr"],
    });
    if (scenario === "bad-signature") marker.signature = "invalid";
    const issue = {
      number: 12, title: "[MODEL EVALUATION REQUEST] org/model", html_url: "https://github.com/EuroEval/EuroEval/issues/12",
      state: "open", assignee: { login: "alice" },
      assignees: hasHealthyOwner ? [{ login: "alice" }, { login: "manual" }, { login: "bob" }] : [{ login: "alice" }],
      labels: [{ name: "not-gguf" }],
      body: `Original request\n<!-- euroeval-volunteer-worker:v1 ${JSON.stringify(marker)} -->`,
    };
    if (["renewed", "replacement"].includes(scenario)) {
      redis.execute(["SET", issueLeaseKey(12, "da"), JSON.stringify({ ...expired, issue_number: 12,
        lease_id: scenario === "renewed" ? "old" : "replacement", contributor: scenario === "renewed" ? "alice" : "bob",
        expires_at: new Date(Date.now() + 600_000).toISOString() })]);
    }
    let failDelete = ["delete-failure", "lost-delete-response", "delete-failure-with-healthy"].includes(scenario);
    globalThis.fetch = async (input, init = {}) => {
      const url = String(input);
      if (url === "https://redis.test") {
        if (scenario === "redis-down") throw new Error("Redis unavailable");
        return redis.respond(init);
      }
      if (url.includes("/issues?") && !init.method) return Response.json([issue]);
      if (url.endsWith("/issues/12/assignees") && init.method === "DELETE") {
        if (failDelete && ["delete-failure", "delete-failure-with-healthy"].includes(scenario)) return Response.json({ message: "unavailable" }, { status: 403 });
        const removed = JSON.parse(init.body).assignees;
        issue.assignees = issue.assignees.filter(({ login }) => !removed.includes(login));
        issue.assignee = issue.assignees[0] ?? null;
        if (failDelete) throw new Error("response lost after commit");
        return Response.json(issue);
      }
      if (url.endsWith("/issues/12")) {
        if (init.method === "PATCH") issue.body = JSON.parse(init.body).body;
        return Response.json(issue);
      }
      throw new Error(`Unexpected external request: ${init.method ?? "GET"} ${url}`);
    };
    const getQueue = async () => {
      const response = await issues(new Request("https://euroeval.test/api/issues?state=open"));
      assert.equal(response.status, 200);
      return (await response.json())[0];
    };
    let entry = await getQueue();
    if (["delete-failure", "lost-delete-response", "delete-failure-with-healthy"].includes(scenario)) {
      failDelete = false;
      entry = await getQueue();
    }
    if (["renewed", "replacement", "bad-signature", "redis-down"].includes(scenario)) {
      assert.deepEqual(entry.assignees, [{ login: "alice" }]);
      assert.equal(toQueueEntry(entry).status, "Evaluating");
      assert.equal(parseVolunteerMarker(issue.body).leases[0].lease_id, "old");
    } else if (hasHealthyOwner) {
      assert.deepEqual(entry.assignees.map(({ login }) => login), ["manual", "bob"]);
      assert.equal(toQueueEntry(entry).status, "Evaluating");
    } else {
      assert.deepEqual(entry.assignees, []);
      assert.equal(entry.assignee, null);
      assert.equal(toQueueEntry(entry).status, "Waiting");
    }
    assert.equal(issue.state, "open");
    assert.ok(issue.body.startsWith("Original request\n"));
    assert.deepEqual(parseVolunteerMarker(issue.body).completed_languages, ["fr"]);
  });
}
