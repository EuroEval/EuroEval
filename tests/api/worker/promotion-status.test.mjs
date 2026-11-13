import test from "node:test";
import assert from "node:assert/strict";
import promotionStatus from "../../../api/worker/promotion-status.ts";
import { withRedisRestBridge } from "./redis-bridge.mjs";

const records = [
  { identity: JSON.stringify(["org/model", "dataset", false, true]), canonical_path: "org_model/dataset__test__fewshot.json", digest: "a".repeat(64) },
];
const digest = "d".repeat(64);
const reservationKey = "euroeval:worker:promotion:12:submission";
const request = (overrides = {}, secret = "promotion-secret") => new Request("https://euroeval.com/api/worker/promotion-status", {
  method: "POST",
  headers: { "content-type": "application/json", "x-promotion-secret": secret },
  body: JSON.stringify({ protocol_version: "volunteer-worker/v1", issue_number: 12, submission_id: "submission",
    outcome: "accepted", records, decision_digest: digest, ...overrides }),
});

test("promotion status is an authenticated read-only reservation lookup", async () => {
  await withRedisRestBridge(async () => {
    process.env.VOLUNTEER_PROMOTION_SECRET = "promotion-secret";
    const missing = await promotionStatus(request());
    assert.equal(missing.status, 200);
    assert.deepEqual(await missing.json(), { protocol_version: "volunteer-worker/v1", status: "absent" });

    const reservation = { issue_number: 12, submission_id: "submission", outcome: "accepted", records,
      token: "private-token", decision_digest: digest, status: "reserved" };
    await fetch("https://redis.test", { method: "POST", body: JSON.stringify(["SET", reservationKey, JSON.stringify(reservation)]) });
    const reserved = await promotionStatus(request());
    assert.equal(reserved.status, 200);
    assert.deepEqual(await reserved.json(), { protocol_version: "volunteer-worker/v1", status: "reserved", decision_digest: digest });

    await fetch("https://redis.test", { method: "POST", body: JSON.stringify(["SET", reservationKey, JSON.stringify({ ...reservation, status: "terminal" })]) });
    const terminal = await promotionStatus(request());
    assert.equal(terminal.status, 200);
    assert.deepEqual(await terminal.json(), { protocol_version: "volunteer-worker/v1", status: "terminal", decision_digest: digest });

    for (const mismatched of [
      request({ outcome: "rejected" }),
      request({ decision_digest: "e".repeat(64) }),
      request({ records: [{ ...records[0], digest: "b".repeat(64) }] }),
    ]) {
      assert.equal((await promotionStatus(mismatched)).status, 409);
    }
    assert.equal((await promotionStatus(request({}, "wrong-secret"))).status, 401);
    assert.equal((await promotionStatus(new Request("https://euroeval.com/api/worker/promotion-status", { method: "GET" }))).status, 405);

    const stored = await fetch("https://redis.test", { method: "POST", body: JSON.stringify(["GET", reservationKey]) }).then((response) => response.json());
    assert.equal(JSON.parse(stored.result).status, "terminal");
  });
});
