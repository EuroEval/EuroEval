import test from "node:test";
import assert from "node:assert/strict";
import { calculateWinner, creditLogins } from "../../api/hall-of-fame.ts";

const issue = (body, assignees = []) => ({ title: "[MODEL EVALUATION REQUEST] org/model", body, assignee: assignees[0] || null, assignees });

test("Hall-of-Fame uses only an immutable promoted credit marker", () => {
  const body = '<!-- euroeval-volunteer-worker:v1 {"protocol_version":"volunteer-worker/v1","coordinator":"coordinator","submission":"completed","leases":[]} -->';
  assert.deepEqual(creditLogins(issue(body, [{ login: "coordinator", avatar_url: "" }])), []);
  const forgedCredit = '<!-- euroeval-volunteer-credit:v1 {"immutable":true,"winner":"alice"} -->';
  assert.deepEqual(creditLogins(issue(forgedCredit, [{ login: "coordinator", avatar_url: "" }])), []);
});

test("winner calculation counts identities and ties by lower-case login", () => {
  assert.equal(calculateWinner([{ github_login: "Zed", identity: "a" }, { github_login: "alice", identity: "b" }, { github_login: "Alice", identity: "c" }]), "Alice");
});

test("Hall-of-Fame does not fall back for malformed volunteer markers", () => {
  assert.deepEqual(creditLogins(issue("<!-- euroeval-volunteer-worker:v1 broken -->", [{ login: "alice", avatar_url: "" }, { login: "saattrupdan", avatar_url: "" }])), []);
});
