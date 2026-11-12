import test from "node:test";
import assert from "node:assert/strict";
import { mergeQueueEvaluators, toQueueEntry } from "../../src/frontend/services/github.ts";

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

test("queue entries merge assigned and trusted public evaluators case-insensitively", () => {
  const trusted = new Map([[42, [
    { login: "alice", avatarUrl: "https://avatar/trusted" },
    { login: "saattrupdan", avatarUrl: "https://github.com/saattrupdan.png" },
  ]]]);
  const entry = toQueueEntry(issue, trusted);
  assert.equal(entry.publicVolunteer, true);
  assert.deepEqual(entry.evaluators.map(({ login }) => login), ["Alice", "saattrupdan"]);
  assert.equal(entry.evaluators[0].avatarUrl, "https://avatar/alice");
});

test("queue evaluator merging retains manually assigned accounts", () => {
  assert.deepEqual(mergeQueueEvaluators(
    [{ login: "Maintainer", avatarUrl: null }],
    [{ login: "maintainer", avatarUrl: "https://avatar/maintainer" }],
  ), [{ login: "Maintainer", avatarUrl: "https://avatar/maintainer" }]);
});
