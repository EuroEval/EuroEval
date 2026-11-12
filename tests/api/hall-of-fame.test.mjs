import test from "node:test";
import assert from "node:assert/strict";
import { creditLogins, publicQueueEvaluators, verifiedPublicLogins } from "../../api/hall-of-fame.ts";
import { signVolunteerMarker } from "../../api/worker/_lib/protocol.ts";
import { renderVolunteerMarker } from "../../api/worker/_lib/markers.ts";

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

test("queue projection exposes signed active and submitted public identities", async () => {
  const previousSecret = process.env.VOLUNTEER_MARKER_SECRET;
  process.env.VOLUNTEER_MARKER_SECRET = "test-secret";
  try {
    const marker = await signVolunteerMarker(42, {
      protocol_version: "volunteer-worker/v1",
      coordinator: "sentinel",
      submission: "submitted",
      leases: [
        { lease_id: "one", language: "da", worker: "private", contributor: "Alice", expires_at: new Date(Date.now() + 60_000).toISOString(), ownership_mode: "public" },
        { lease_id: "two", language: "sv", worker: "private", contributor: "saattrupdan", expires_at: new Date(Date.now() + 60_000).toISOString(), ownership_mode: "public" },
      ],
      submissions: [{ submission_id: "submission", language: "fi", manifest_path: "manifest.json", submitted_at: "2025-01-01T00:00:00Z", verified_contributor: "alice", result_count: 1, status: "submitted", ownership_mode: "public" }],
    });
    const result = await publicQueueEvaluators({ ...issue(renderVolunteerMarker(marker)), number: 42 });
    assert.deepEqual(result.map(({ login }) => login), ["Alice", "saattrupdan"]);
    assert.deepEqual(await publicQueueEvaluators({ ...issue(renderVolunteerMarker(marker)), number: 43 }), []);
    assert.deepEqual(await publicQueueEvaluators({ ...issue("<!-- euroeval-volunteer-worker:v1 broken -->"), number: 42 }), []);
    assert.deepEqual(await verifiedPublicLogins({ ...issue(renderVolunteerMarker(marker)), number: 42 }), []);
  } finally {
    if (previousSecret === undefined) delete process.env.VOLUNTEER_MARKER_SECRET;
    else process.env.VOLUNTEER_MARKER_SECRET = previousSecret;
  }
});

test("Hall-of-Fame credits only accepted public submissions from signed markers", async () => {
  const previousSecret = process.env.VOLUNTEER_MARKER_SECRET;
  process.env.VOLUNTEER_MARKER_SECRET = "test-secret";
  const submission = (submission_id, status, ownership_mode = "public", verified_contributor = "Alice") => ({
    submission_id,
    language: "da",
    manifest_path: `manifests/${submission_id}.json`,
    submitted_at: "2025-01-01T00:00:00Z",
    verified_contributor,
    result_count: 1,
    status,
    ownership_mode,
  });
  try {
    const marker = await signVolunteerMarker(42, {
      protocol_version: "volunteer-worker/v1",
      coordinator: "sentinel",
      submission: "accepted",
      leases: [],
      submissions: [
        submission("accepted", "accepted"),
        submission("rejected", "rejected", "public", "Bob"),
        submission("legacy", "accepted", "assignee", "Carol"),
      ],
    });
    const body = renderVolunteerMarker(marker);
    assert.deepEqual(await verifiedPublicLogins({ ...issue(body), number: 42 }), ["Alice"]);
    assert.deepEqual(await verifiedPublicLogins({ ...issue(body), number: 43 }), []);
    assert.deepEqual(await verifiedPublicLogins({ ...issue(body.replace('"status":"accepted"', '"status":"rejected"')), number: 42 }), []);
  } finally {
    if (previousSecret === undefined) delete process.env.VOLUNTEER_MARKER_SECRET;
    else process.env.VOLUNTEER_MARKER_SECRET = previousSecret;
  }
});
