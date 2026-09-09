import test from "node:test";
import assert from "node:assert/strict";
import { creditLogins } from "./hall-of-fame.ts";

const issue = (body, assignees = []) => ({ title: "[MODEL EVALUATION REQUEST] org/model", body, assignee: assignees[0] || null, assignees });

test("Hall-of-Fame gives one credit to the broker marker winner", () => {
  const body = '<!-- euroeval-volunteer-worker:v1 {"winner":"alice","leases":[{"language":"da","worker":"w","contributor":"alice","expires_at":9999999999999}]} -->';
  assert.deepEqual(creditLogins(issue(body, [{ login: "coordinator", avatar_url: "" }])), ["alice"]);
});

test("Hall-of-Fame safely falls back for malformed markers and excludes owner", () => {
  assert.deepEqual(creditLogins(issue("<!-- euroeval-volunteer-worker:v1 broken -->", [{ login: "alice", avatar_url: "" }, { login: "saattrupdan", avatar_url: "" }])), ["alice"]);
});
