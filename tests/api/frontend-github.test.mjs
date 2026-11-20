import test from "node:test";
import assert from "node:assert/strict";
import {
  listOpenEvalIssues,
  mergeQueueEvaluators,
  toQueueEntry,
} from "../../src/frontend/services/github.ts";

const issue = {
  number: 42,
  title: "[MODEL EVALUATION REQUEST] org/model",
  html_url: "https://github.com/EuroEval/EuroEval/issues/42",
  body: "## Languages\nDanish",
  assignee: { login: "Alice", avatar_url: "https://avatar/alice" },
  assignees: [
    { login: "Alice", avatar_url: "https://avatar/alice" },
    { login: "ALICE", avatar_url: "https://avatar/alice" },
  ],
  created_at: "2026-01-01T00:00:00Z",
};

test("assignment display uses only current GitHub issue assignees", () => {
  const projected = new Map([[42, [
    { login: "Alice", avatarUrl: "https://avatar/projected" },
  ]]]);
  const withoutCurrentAssignees = toQueueEntry({
    ...issue,
    assignee: null,
    assignees: [],
  }, projected);
  assert.deepEqual(withoutCurrentAssignees.evaluators, []);
  assert.equal(withoutCurrentAssignees.publicVolunteer, false);

  const currentAndStale = toQueueEntry({
    ...issue,
    assignee: { login: "Bob", avatar_url: "https://avatar/bob" },
    assignees: [{ login: "Bob", avatar_url: "https://avatar/bob" }],
  }, projected);
  assert.deepEqual(currentAndStale.evaluators.map(({ login }) => login), ["Bob"]);
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

test("community review submissions are pending review with or without assignees", () => {
  for (const assignee of [null, issue.assignee]) {
    const entry = toQueueEntry({
      ...issue,
      assignee,
      assignees: assignee ? [assignee] : [],
      labels: ["community-review-ready"],
    }, new Map(), new Set([42]));
    assert.equal(entry.status, "Pending review");
  }
});

test("community review label without trusted completion fails closed", () => {
  const entry = toQueueEntry({
    ...issue,
    labels: [{ name: "community-review-ready" }],
  }, new Map(), new Set());
  assert.equal(entry.status, "Evaluating");
});

test("verified completion controls pending review for a full model", () => {
  const complete = toQueueEntry({
    ...issue,
    labels: ["community-review-ready"],
  }, new Map(), new Set([42]));
  const incomplete = toQueueEntry({
    ...issue,
    labels: ["community-review-ready"],
  }, new Map(), new Set());
  assert.equal(complete.status, "Pending review");
  assert.equal(incomplete.status, "Evaluating");
});

test("partial language completion remains evaluating without an assignee", () => {
  const entry = toQueueEntry({
    ...issue,
    assignee: null,
    assignees: [],
    labels: ["community-review-ready"],
  }, new Map(), new Set());
  assert.equal(entry.status, "Evaluating");
});

test("results ready takes precedence over pending review and evaluation", () => {
  const entry = toQueueEntry({
    ...issue,
    labels: ["community-review-ready", "results-ready"],
  });
  assert.equal(entry.status, "Awaiting publish");
});

test("evaluation statuses and evaluator attribution are preserved", () => {
  const evaluating = toQueueEntry(issue);
  assert.equal(evaluating.status, "Evaluating");
  assert.deepEqual(evaluating.evaluators.map(({ login }) => login), ["Alice"]);

  const awaitingPublish = toQueueEntry({
    ...issue,
    labels: ["results-ready"],
  });
  assert.equal(awaitingPublish.status, "Awaiting publish");
});

test("queue sorts pending reviews after results ready and before evaluating", async () => {
  const originalFetch = globalThis.fetch;
  const queuedIssues = [
    { ...issue, number: 1, labels: [], created_at: "2026-01-04T00:00:00Z" },
    {
      ...issue,
      number: 2,
      labels: ["community-review-ready"],
      created_at: "2026-01-03T00:00:00Z",
    },
    {
      ...issue,
      number: 3,
      labels: ["results-ready"],
      created_at: "2026-01-02T00:00:00Z",
    },
  ];
  globalThis.fetch = async (url) => ({
    ok: true,
    json: async () => url.startsWith("/api/issues")
      ? queuedIssues
      : [{ number: 2, evaluators: [], allLanguagesComplete: true }],
  });
  try {
    const entries = await listOpenEvalIssues();
    assert.deepEqual(entries.map(({ number }) => number), [3, 2, 1]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("fresh queue refresh bypasses caches for completion projection", async () => {
  const originalFetch = globalThis.fetch;
  const urls = [];
  globalThis.fetch = async (url, init) => {
    urls.push({ url, init });
    return { ok: true, json: async () => [] };
  };
  try {
    await listOpenEvalIssues({ fresh: true });
    const projection = urls.find(({ url }) => url.startsWith("/api/hall-of-fame"));
    assert.match(projection.url, /[?&]_=/);
    assert.deepEqual(projection.init, { cache: "no-store" });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("queue evaluator merging retains manually assigned accounts", () => {
  assert.deepEqual(mergeQueueEvaluators(
    [{ login: "Maintainer", avatarUrl: null }],
    [{ login: "maintainer", avatarUrl: "https://avatar/maintainer" }],
  ), [{ login: "Maintainer", avatarUrl: "https://avatar/maintainer" }]);
});
