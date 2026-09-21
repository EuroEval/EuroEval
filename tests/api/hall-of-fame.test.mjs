import test from "node:test";
import assert from "node:assert/strict";
import { creditLogins } from "../../api/hall-of-fame.ts";

const issue = (body, assignees = []) => ({ title: "[MODEL EVALUATION REQUEST] org/model", body, assignee: assignees[0] || null, assignees });

test("Hall-of-Fame credits current assignees regardless of markers", () => {
  const body = '<!-- euroeval-volunteer-worker:v1 broken -->';
  assert.deepEqual(creditLogins(issue(body, [
    { login: "Alice", avatar_url: "" }, { login: "alice", avatar_url: "" },
    { login: "saattrupdan", avatar_url: "" }, { login: "Bob", avatar_url: "" },
  ])), ["Alice", "Bob"]);
});

test("Hall-of-Fame uses assignees when a legacy assignee field is present", () => {
  assert.deepEqual(creditLogins(issue(null, [{ login: "alice", avatar_url: "" }])), ["alice"]);
});
