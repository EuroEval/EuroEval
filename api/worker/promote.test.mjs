import test from "node:test";
import assert from "node:assert/strict";
import { claimableLanguages } from "./_lib.ts";
import promote, { largestAcceptedShare, promotionPlan } from "./promote.ts";

const submission = (id, language, contributor, count, status = "submitted") => ({
  submission_id: id,
  language,
  manifest_path: `volunteer/manifests/${id}.json`,
  submitted_at: "2026-09-06T10:00:00Z",
  verified_contributor: contributor,
  result_count: count,
  status,
});
const marker = (submissions, leases = []) => ({
  protocol_version: "volunteer-worker/v1",
  coordinator: "coordinator",
  submission: "submitted",
  leases,
  submissions,
});

test("acceptance awards no credit before every language completes", () => {
  const plan = promotionPlan(
    marker([submission("one", "da", "alice", 4), submission("two", "de", "bob", 9)]),
    ["da", "de"],
    "one",
    "accepted",
  );
  assert.equal(plan.complete, false);
  assert.equal(plan.winner, null);
  assert.equal(plan.marker.submission, "submitted");
  assert.equal(plan.releaseCoordinator, false);
});

test("all-language completion chooses largest server-derived share", () => {
  const plan = promotionPlan(
    marker([
      submission("da", "da", "alice", 8, "accepted"),
      submission("de", "de", "bob", 4),
      submission("fr", "fr", "bob", 5, "accepted"),
    ]),
    ["da", "de", "fr"],
    "de",
    "accepted",
  );
  assert.equal(plan.complete, true);
  assert.equal(plan.winner, "bob");
  assert.equal(plan.marker.submission, "accepted");
  assert.equal(plan.releaseCoordinator, true);
  assert.equal(plan.removeReviewLabel, true);
});

test("largest accepted share ties by lower-case login", () => {
  assert.equal(
    largestAcceptedShare([
      submission("one", "da", "Zed", 5, "accepted"),
      submission("two", "de", "alice", 5, "accepted"),
    ]),
    "alice",
  );
});

test("rejection preserves audit and makes language claimable", () => {
  const plan = promotionPlan(
    marker([submission("one", "da", "alice", 4)]),
    ["da"],
    "one",
    "rejected",
  );
  assert.equal(plan.marker.submissions[0].status, "rejected");
  assert.equal(plan.marker.submission, "rejected");
  assert.equal(plan.removeReviewLabel, true);
  assert.equal(plan.releaseCoordinator, true);
  assert.deepEqual(claimableLanguages(["da"], plan.marker), ["da"]);
});

test("submitted and accepted languages cannot be reclaimed", () => {
  const value = marker([
    submission("one", "da", "alice", 4),
    submission("two", "de", "bob", 3, "accepted"),
    submission("three", "fr", "cam", 2, "rejected"),
  ]);
  assert.deepEqual(claimableLanguages(["da", "de", "fr"], value), ["fr"]);
});

test("terminal retry completes the same lifecycle plan", () => {
  const value = marker([submission("one", "da", "alice", 4, "accepted")]);
  const plan = promotionPlan(value, ["da"], "one", "accepted");
  assert.equal(plan.complete, true);
  assert.equal(plan.winner, "alice");
  assert.throws(() => promotionPlan(value, ["da"], "one", "rejected"));
});

test("handler finishes GitHub labels, credit, ownership, and notification", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  let issue = {
    number: 12,
    title: "[MODEL EVALUATION REQUEST] org/model",
    body: `- [x] Greek\n\n<!-- euroeval-volunteer-worker:v1 ${JSON.stringify(marker([submission("one", "el", "alice", 4)]))} -->`,
    state: "open",
    assignees: [{ login: "coordinator" }],
    labels: [{ name: "community-review-ready" }],
  };
  const comments = [];
  process.env.VOLUNTEER_PROMOTION_SECRET = "promotion-secret";
  process.env.WORKER_COORDINATOR_LOGIN = "coordinator";
  process.env.GITHUB_TOKEN = "github-token";
  process.env.UPSTASH_REDIS_REST_URL = "https://redis.test";
  process.env.UPSTASH_REDIS_REST_TOKEN = "redis-token";
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input);
    const method = init.method || "GET";
    if (url === "https://redis.test") {
      const command = JSON.parse(init.body);
      return Response.json({ result: command[0] === "SET" ? "OK" : 1 });
    }
    if (url.endsWith("/issues/12") && method === "GET") return Response.json(issue);
    if (url.endsWith("/issues/12") && method === "PATCH") {
      issue = { ...issue, body: JSON.parse(init.body).body };
      return Response.json(issue);
    }
    if (url.endsWith("/issues/12/labels") && method === "POST") {
      issue.labels.push({ name: JSON.parse(init.body).labels[0] });
      return Response.json(issue.labels);
    }
    if (url.includes("/issues/12/labels/") && method === "DELETE") {
      issue.labels = issue.labels.filter((item) => item.name !== "community-review-ready");
      return Response.json(issue.labels);
    }
    if (url.endsWith("/issues/12/assignees") && method === "DELETE") {
      issue.assignees = [];
      return Response.json(issue);
    }
    if (url.includes("/issues/12/comments") && method === "GET") return Response.json(comments);
    if (url.endsWith("/issues/12/comments") && method === "POST") {
      comments.push({ body: JSON.parse(init.body).body });
      return Response.json(comments.at(-1), { status: 201 });
    }
    throw new Error(`Unexpected request: ${method} ${url}`);
  };
  try {
    const request = new Request("https://euroeval.com/api/worker/promote", {
      method: "POST",
      headers: { "content-type": "application/json", "x-promotion-secret": "promotion-secret" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", issue_number: 12, submission_id: "one", outcome: "accepted" }),
    });
    const response = await promote(request);
    const responseBody = await response.text();
    assert.equal(response.status, 200, responseBody);
    assert.match(issue.body, /euroeval-volunteer-credit:v1/);
    assert.deepEqual(issue.labels, [{ name: "results-ready" }]);
    assert.deepEqual(issue.assignees, []);
    assert.match(comments[0].body, /submission \*\*one\*\*/);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});
