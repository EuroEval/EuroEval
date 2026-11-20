import test from "node:test";
import assert from "node:assert/strict";
import {
  listOpenEvalIssues,
  toQueueEntry,
} from "../../src/frontend/services/github.ts";

const issue = {
  number: 42,
  title: "[MODEL EVALUATION REQUEST] org/model",
  html_url: "https://github.com/EuroEval/EuroEval/issues/42",
  body: "## Languages\nDanish",
  assignee: { login: "Alice", avatar_url: "https://avatar/alice" },
  assignees: [{ login: "Alice", avatar_url: "https://avatar/alice" }],
  created_at: "2026-01-01T00:00:00Z",
};

test("unassigned requests wait regardless of body credits", () => {
  const entry = toQueueEntry({
    ...issue,
    assignee: null,
    assignees: [],
    body: `${issue.body}\n\nCompleted evaluation by Alice`,
  });
  assert.equal(entry.status, "Waiting");
  assert.deepEqual(entry.evaluators, []);
});

test("current GitHub assignees indicate evaluation", () => {
  const entry = toQueueEntry({
    ...issue,
    assignee: { login: "Bob", avatar_url: "https://avatar/bob" },
    assignees: [{ login: "Bob", avatar_url: "https://avatar/bob" }],
  });
  assert.equal(entry.status, "Evaluating");
  assert.deepEqual(entry.evaluators.map(({ login }) => login), ["Bob"]);
});

test("all current GitHub issue assignees remain in the assignment display", () => {
  const entry = toQueueEntry({
    ...issue,
    assignees: [
      { login: "Alice", avatar_url: "https://avatar/alice" },
      { login: "Bob", avatar_url: "https://avatar/bob" },
    ],
  });
  assert.deepEqual(entry.evaluators.map(({ login }) => login), ["Alice", "Bob"]);
});

test("community review label means pending review", () => {
  const entry = toQueueEntry({ ...issue, labels: ["community-review-ready"] });
  assert.equal(entry.status, "Pending review");
});

test("results ready takes precedence over pending review and evaluation", () => {
  const entry = toQueueEntry({
    ...issue,
    labels: ["community-review-ready", "results-ready"],
  });
  assert.equal(entry.status, "Awaiting publish");
});

test("gated and failed labels preserve their statuses", () => {
  assert.equal(toQueueEntry({ ...issue, labels: ["gated"] }).status, "Gated model");
  assert.equal(
    toQueueEntry({ ...issue, labels: ["evaluation-failed"] }).status,
    "Error",
  );
});

test("queue sorting uses labels and makes no hall-of-fame request", async () => {
  const originalFetch = globalThis.fetch;
  const urls = [];
  const queuedIssues = [
    { ...issue, number: 1, labels: [], created_at: "2026-01-04T00:00:00Z" },
    { ...issue, number: 2, labels: ["community-review-ready"] },
    { ...issue, number: 3, labels: ["results-ready"] },
  ];
  globalThis.fetch = async (url) => {
    urls.push(url);
    return { ok: true, json: async () => queuedIssues };
  };
  try {
    const entries = await listOpenEvalIssues();
    assert.deepEqual(entries.map(({ number }) => number), [3, 2, 1]);
    assert.equal(urls.length, 1);
    assert.match(urls[0], /^\/api\/issues/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("fresh queue refresh bypasses the issues cache", async () => {
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.fetch = async (url, init) => {
    requests.push({ url, init });
    return { ok: true, json: async () => [] };
  };
  try {
    await listOpenEvalIssues({ fresh: true });
    assert.equal(requests.length, 1);
    assert.match(requests[0].url, /^\/api\/issues/);
    assert.match(requests[0].url, /[?&]_=/);
    assert.deepEqual(requests[0].init, { cache: "no-store" });
  } finally {
    globalThis.fetch = originalFetch;
  }
});
