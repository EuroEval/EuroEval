import test from "node:test";
import assert from "node:assert/strict";
import { claimableLanguages, signVolunteerMarker } from "./_lib.ts";
import promote, { largestAcceptedShare, promotionPlan } from "./promote.ts";
import reservePromotion from "./promotion-reserve.ts";

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

test("repeated rejection is safe after cleanup interruption", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  Object.assign(process.env, {
    VOLUNTEER_MARKER_SECRET: "marker-secret", VOLUNTEER_PROMOTION_SECRET: "promotion-secret",
    WORKER_COORDINATOR_LOGIN: "coordinator", GITHUB_TOKEN: "github-token",
    UPSTASH_REDIS_REST_URL: "https://redis.test", UPSTASH_REDIS_REST_TOKEN: "redis-token",
  });
  const records = [{ identity: JSON.stringify(["org/model", "dataset", false, true]), canonical_path: "org_model/dataset__test__fewshot.json", digest: "a".repeat(64) }];
  const decisionDigest = "d".repeat(64);
  const issueMarker = await signVolunteerMarker(12, marker([submission("one", "el", "alice", 4)]));
  let issue = { number: 12, title: "[MODEL EVALUATION REQUEST] org/model",
    body: `- [x] Greek\n\n<!-- euroeval-volunteer-worker:v1 ${JSON.stringify(issueMarker)} -->`, state: "open",
    assignees: [{ login: "coordinator" }], labels: [{ name: "community-review-ready" }] };
  const comments = [];
  const values = new Map([["euroeval:worker:promotion:12:one", JSON.stringify({
    issue_number: 12, submission_id: "one", outcome: "rejected", records, token: "reservation-token", decision_digest: decisionDigest, status: "reserved",
  })]]);
  let cleanupCalls = 0;
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input); const method = init.method || "GET";
    if (url === "https://redis.test") {
      const command = JSON.parse(init.body); let result = command[0] === "EVAL" ? 1 : "OK";
      if (command[0] === "GET") result = values.get(command[1]) || null;
      if (command[0] === "SET") { values.set(command[1], command[2]); result = "OK"; }
      if (command[0] === "EVAL" && command[1].includes("for i=1,#KEYS-1")) {
        cleanupCalls += 1;
        if (cleanupCalls === 1) throw new Error("simulated interruption");
        const keyCount = Number(command[2]);
        for (let i = 0; i < keyCount; i += 1) values.delete(command[3 + i]);
        result = 1;
      }
      return Response.json({ result });
    }
    if (url.endsWith("/issues/12") && method === "GET") return Response.json(issue);
    if (url.endsWith("/issues/12") && method === "PATCH") { issue = { ...issue, body: JSON.parse(init.body).body }; return Response.json(issue); }
    if (url.includes("/issues/12/labels/") && method === "DELETE") { issue.labels = []; return Response.json(issue.labels); }
    if (url.endsWith("/issues/12/assignees") && method === "DELETE") { issue.assignees = []; return Response.json(issue); }
    if (url.includes("/issues/12/comments") && method === "GET") return Response.json(comments);
    if (url.endsWith("/issues/12/comments") && method === "POST") { comments.push({ body: JSON.parse(init.body).body }); return Response.json(comments.at(-1)); }
    throw new Error(`Unexpected request: ${method} ${url}`);
  };
  const request = () => new Request("https://euroeval.com/api/worker/promote", { method: "POST",
    headers: { "content-type": "application/json", "x-promotion-secret": "promotion-secret" },
    body: JSON.stringify({ protocol_version: "volunteer-worker/v1", issue_number: 12, submission_id: "one",
      outcome: "rejected", reservation_token: "reservation-token", decision_digest: decisionDigest, records }) });
  try {
    const first = await promote(request());
    assert.equal(first.status, 200, await first.text());
    const second = await promote(request());
    assert.equal(second.status, 200, await second.text());
    assert.equal(cleanupCalls, 3); // first call retries the interrupted atomic EVAL
    assert.match(issue.body, /"submission":"rejected"/);
    assert.equal(comments.length, 1);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

test("promotion reservation returns stable review metadata on resume", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  Object.assign(process.env, {
    VOLUNTEER_MARKER_SECRET: "marker-secret", VOLUNTEER_PROMOTION_SECRET: "promotion-secret",
    GITHUB_TOKEN: "github-token", UPSTASH_REDIS_REST_URL: "https://redis.test", UPSTASH_REDIS_REST_TOKEN: "redis-token",
  });
  const records = [{ identity: JSON.stringify(["org/model", "dataset", false, true]), canonical_path: "org_model/dataset__test__fewshot.json", digest: "a".repeat(64) }];
  const decisionDigest = "d".repeat(64);
  const issueMarker = await signVolunteerMarker(12, marker([submission("one", "el", "alice", 4)]));
  const issue = { number: 12, title: "[MODEL EVALUATION REQUEST] org/model",
    body: `- [x] Greek\n\n<!-- euroeval-volunteer-worker:v1 ${JSON.stringify(issueMarker)} -->`, state: "open",
    assignees: [{ login: "coordinator" }], labels: [] };
  const values = new Map();
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input); const method = init.method || "GET";
    if (url === "https://redis.test") {
      const command = JSON.parse(init.body); let result = "OK";
      if (command[0] === "GET") result = values.get(command[1]) || null;
      if (command[0] === "SET") {
        if (command.includes("NX") && values.has(command[1])) result = null;
        else { values.set(command[1], command[2]); result = "OK"; }
      }
      if (command[0] === "EVAL") { values.delete(command[3]); result = 0; }
      return Response.json({ result });
    }
    if (url.endsWith("/issues/12") && method === "GET") return Response.json(issue);
    throw new Error(`Unexpected request: ${method} ${url}`);
  };
  const request = (token) => new Request("https://euroeval.com/api/worker/promotion-lock", {
    method: "POST", headers: { "content-type": "application/json", "x-promotion-secret": "promotion-secret" },
    body: JSON.stringify({ protocol_version: "volunteer-worker/v1", issue_number: 12, submission_id: "one",
      outcome: "rejected", reviewer: "alice", records, ...(token ? { reservation_token: token } : {}) }),
  });
  try {
    const first = await reservePromotion(request()); const firstBody = await first.json();
    assert.equal(first.status, 201, JSON.stringify(firstBody));
    assert.equal(firstBody.decision_reviewer, "alice");
    assert.match(firstBody.decision_created_at, /^\d{4}-\d\d-\d\dT/);
    assert.equal(firstBody.decision_nonce, undefined);
    const second = await reservePromotion(request(firstBody.token)); const secondBody = await second.json();
    assert.equal(second.status, 200);
    assert.equal(secondBody.token, firstBody.token);
    assert.equal(secondBody.decision_reviewer, firstBody.decision_reviewer);
    assert.equal(secondBody.decision_created_at, firstBody.decision_created_at);
    const incompatible = await reservePromotion(new Request("https://euroeval.com/api/worker/promotion-lock", {
      method: "POST", headers: { "content-type": "application/json", "x-promotion-secret": "promotion-secret" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", issue_number: 12, submission_id: "one",
        outcome: "rejected", reviewer: "alice", reservation_token: firstBody.token,
        decision_nonce: "wrong", records }),
    }));
    assert.equal(incompatible.status, 400);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});

test("handler finishes GitHub labels, credit, ownership, and notification", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  process.env.VOLUNTEER_MARKER_SECRET = "marker-secret";
  const issueMarker = await signVolunteerMarker(12, marker([submission("one", "el", "alice", 4)]));
  let issue = {
    number: 12,
    title: "[MODEL EVALUATION REQUEST] org/model",
    body: `- [x] Greek\n\n<!-- euroeval-volunteer-worker:v1 ${JSON.stringify(issueMarker)} -->`,
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
  const redisValues = new Map();
  const records = [{ identity: "[\"org/model\",\"dataset\",false,true]", canonical_path: "org_model/dataset__test__fewshot.json", digest: "a".repeat(64) }];
  const decisionDigest = "d".repeat(64);
  redisValues.set("euroeval:worker:promotion:12:one", JSON.stringify({    issue_number: 12, submission_id: "one", outcome: "accepted", records, token: "reservation-token", decision_digest: decisionDigest,
    decision_reviewer: "alice", decision_created_at: "2026-09-06T12:00:00Z", status: "reserved" }));
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input);
    const method = init.method || "GET";
    if (url === "https://redis.test") {
      const command = JSON.parse(init.body);
      let result = 1;
      if (command[0] === "GET") result = redisValues.get(command[1]) || null;
      if (command[0] === "SET") { redisValues.set(command[1], command[2]); result = "OK"; }
      if (command[0] === "SMEMBERS") result = [];
      return Response.json({ result });
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
    const missingDigest = new Request("https://euroeval.com/api/worker/promote", {
      method: "POST",
      headers: { "content-type": "application/json", "x-promotion-secret": "promotion-secret" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", issue_number: 12, submission_id: "one", outcome: "accepted", reservation_token: "reservation-token", records }),
    });
    assert.equal((await promote(missingDigest)).status, 400);
    const request = new Request("https://euroeval.com/api/worker/promote", {
      method: "POST",
      headers: { "content-type": "application/json", "x-promotion-secret": "promotion-secret" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", issue_number: 12, submission_id: "one", outcome: "accepted", reservation_token: "reservation-token", decision_digest: decisionDigest, records }),
    });
    const response = await promote(request);
    const responseBody = await response.json();
    assert.equal(response.status, 200, JSON.stringify(responseBody));
    assert.equal(responseBody.decision_reviewer, "alice");
    assert.equal(responseBody.decision_created_at, "2026-09-06T12:00:00Z");
    assert.match(issue.body, /euroeval-volunteer-credit:v1/);
    assert.match(issue.body, /"decision_reviewer":"alice"/);
    assert.match(issue.body, /"decision_created_at":"2026-09-06T12:00:00Z"/);
    assert.deepEqual(issue.labels, [{ name: "results-ready" }]);
    assert.deepEqual(issue.assignees, []);
    assert.match(comments[0].body, /submission \*\*one\*\*/);
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});
