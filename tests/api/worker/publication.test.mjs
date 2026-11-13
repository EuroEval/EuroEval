import test from "node:test";
import assert from "node:assert/strict";
import publicationLock from "../../../api/worker/publication-lock.ts";
import promote from "../../../api/worker/promote.ts";
import reservePromotion from "../../../api/worker/promotion-reserve.ts";
import { signVolunteerMarker } from "../../../api/worker/_lib.ts";
import { withRedisRestBridge } from "./redis-bridge.mjs";

const request = (action, owner, secret = "maintainer-secret", bucket = "EuroEval/results") => new Request("https://euroeval.com/api/worker/publication-lock", {
  method: "POST", headers: { authorization: `Bearer ${secret}`, "content-type": "application/json" },
  body: JSON.stringify({ action, bucket, owner }),
});

test("bound promotion records survive Redis Lua ordering across the full lifecycle", async () => {
  await withRedisRestBridge(async () => {
    const originalFetch = globalThis.fetch;
    const originalEnv = { ...process.env };
    Object.assign(process.env, {
      CANONICAL_PUBLICATION_SECRET: "maintainer-secret",
      VOLUNTEER_MARKER_SECRET: "marker-secret",
      VOLUNTEER_PROMOTION_SECRET: "promotion-secret",
      GITHUB_TOKEN: "github-token",
    });
    const identity = JSON.stringify(["org/model", "dataset", false, true]);
    const records = [{ identity, canonical_path: "org_model/dataset__test__fewshot.json", digest: "a".repeat(64) }];
    const decisionDigest = "d".repeat(64);
    const createdAt = "2026-09-06T12:00:00Z";
    const marker = await signVolunteerMarker(12, {
      protocol_version: "volunteer-worker/v1", coordinator: "coordinator", submission: "submitted", leases: [],
      submissions: ["one", "expired"].map((submission_id) => ({ submission_id, language: "el",
        manifest_path: `volunteer/manifests/${submission_id}.json`, submitted_at: createdAt,
        verified_contributor: "alice", result_count: 1, status: "submitted" })),
    });
    let issue = { number: 12, title: "[MODEL EVALUATION REQUEST] org/model",
      body: `- [x] Greek\n\n<!-- euroeval-volunteer-worker:v1 ${JSON.stringify(marker)} -->`,
      state: "open", assignees: [{ login: "coordinator" }], labels: [{ name: "community-review-ready" }] };
    const comments = [];
    globalThis.fetch = async (input, init = {}) => {
      const url = String(input);
      const method = init.method || "GET";
      if (url === "https://redis.test") return originalFetch(input, init);
      if (url.endsWith("/issues/12") && method === "GET") return Response.json(issue);
      if (url.endsWith("/issues/12") && method === "PATCH") {
        issue = { ...issue, body: JSON.parse(init.body).body }; return Response.json(issue);
      }
      if (url.includes("/issues/12/comments") && method === "GET") return Response.json(comments);
      if (url.endsWith("/issues/12/comments") && method === "POST") {
        comments.push({ body: JSON.parse(init.body).body }); return Response.json(comments.at(-1));
      }
      if (url.endsWith("/issues/12/assignees") && method === "DELETE") { issue.assignees = []; return Response.json(issue); }
      if (url.includes("/issues/12/labels/") && method === "DELETE") { issue.labels = []; return Response.json(issue.labels); }
      if (url.endsWith("/issues/12/labels") && method === "POST") return Response.json(issue.labels);
      throw new Error(`Unexpected request: ${method} ${url}`);
    };
    const lockOwner = "o".repeat(32);
    const reserve = (changes = {}) => new Request("https://euroeval.com/api/worker/promotion-lock", {
      method: "POST", headers: { "content-type": "application/json", "x-promotion-secret": "promotion-secret" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", issue_number: 12, submission_id: "one",
        outcome: "accepted", records, publication_bucket: "EuroEval/results", publication_owner: lockOwner,
        reviewer: "alice", decision_reviewer: "alice", decision_created_at: createdAt, ...changes }),
    });
    try {
      assert.equal((await publicationLock(request("acquire", lockOwner))).status, 200);
      const created = await reservePromotion(reserve());
      const createdBody = await created.json();
      assert.equal(created.status, 201, JSON.stringify(createdBody));
      const bind = await reservePromotion(reserve({ reservation_token: createdBody.token, decision_digest: decisionDigest }));
      assert.equal(bind.status, 200);
      const stored = JSON.parse(await fetch("https://redis.test", {
        method: "POST", body: JSON.stringify(["GET", "euroeval:worker:promotion:12:one"]),
      }).then((response) => response.json()).then((result) => result.result));
      assert.notDeepEqual(Object.keys(stored.records[0]), Object.keys(records[0]));
      assert.deepEqual({ ...stored.records[0] }, { ...records[0] });

      const renewed = await reservePromotion(reserve({ reservation_token: createdBody.token, decision_digest: decisionDigest }));
      assert.equal(renewed.status, 200);
      const recovered = await reservePromotion(reserve({ reviewer: undefined, decision_digest: decisionDigest }));
      const recoveredBody = await recovered.json();
      assert.equal(recovered.status, 200, JSON.stringify(recoveredBody));
      assert.equal(recoveredBody.token, createdBody.token);

      const expiredInitial = await reservePromotion(reserve({ submission_id: "expired", decision_digest: decisionDigest }));
      const expiredInitialBody = await expiredInitial.json();
      assert.equal(expiredInitial.status, 201, JSON.stringify(expiredInitialBody));
      await fetch("https://redis.test", { method: "POST", body: JSON.stringify(["DEL", "euroeval:worker:promotion:12:expired"]) });
      const expiredRecovery = await reservePromotion(reserve({ submission_id: "expired", decision_digest: decisionDigest }));
      const expiredRecoveryBody = await expiredRecovery.json();
      assert.equal(expiredRecovery.status, 201, JSON.stringify(expiredRecoveryBody));
      assert.notEqual(expiredRecoveryBody.token, expiredInitialBody.token);

      const promoteRequest = (changes = {}) => new Request("https://euroeval.com/api/worker/promote", {
        method: "POST", headers: { "content-type": "application/json", "x-promotion-secret": "promotion-secret" },
        body: JSON.stringify({ protocol_version: "volunteer-worker/v1", issue_number: 12, submission_id: "one",
          outcome: "accepted", reservation_token: createdBody.token, decision_digest: decisionDigest, records,
          publication_bucket: "EuroEval/results", publication_owner: lockOwner, ...changes }),
      });
      const promoted = await promote(promoteRequest());
      assert.equal(promoted.status, 200, await promoted.text());
      assert.equal((await promote(promoteRequest({ decision_digest: "e".repeat(64) }))).status, 409);
      assert.equal((await promote(promoteRequest({ records: [{ ...records[0], digest: "b".repeat(64) }] }))).status, 409);
      const otherIdentity = JSON.stringify(["other/model", "dataset", false, true]);
      const otherRecords = [{ identity: otherIdentity, canonical_path: "other_model/dataset__test__fewshot.json",
        digest: records[0].digest }];
      assert.equal((await promote(promoteRequest({ records: otherRecords }))).status, 409);
      assert.equal((await reservePromotion(reserve({ outcome: "rejected", decision_digest: decisionDigest }))).status, 409);
      assert.equal((await reservePromotion(reserve({ decision_digest: decisionDigest,
        records: [{ ...records[0], digest: "b".repeat(64) }] }))).status, 409);
      assert.equal((await reservePromotion(reserve({ decision_digest: decisionDigest, records: otherRecords }))).status, 409);
    } finally {
      globalThis.fetch = originalFetch;
      process.env = originalEnv;
    }
  });
});

test("publication lock runs its Lua against Redis and survives contention and lost replies", async () => {
  await withRedisRestBridge(async ({ loseNextRedisResponse }) => {
    process.env.CANONICAL_PUBLICATION_SECRET = "maintainer-secret";
    const firstOwner = "a".repeat(32);
    const secondOwner = "b".repeat(32);
    assert.equal((await publicationLock(request("acquire", firstOwner))).status, 200);
    assert.equal((await publicationLock(request("acquire", firstOwner))).status, 200);
    for (let attempt = 0; attempt < 101; attempt += 1) {
      assert.equal((await publicationLock(request("assert", firstOwner))).status, 200);
    }
    assert.equal((await publicationLock(request("acquire", secondOwner))).status, 409);
    assert.equal((await publicationLock(request("assert", secondOwner))).status, 409);
    assert.equal((await publicationLock(request("release", secondOwner))).status, 409);
    assert.equal((await publicationLock(request("status", firstOwner))).status, 200);

    const redisKey = "euroeval:worker:canonical-publication:EuroEval/results";
    const ttl = await fetch("https://redis.test", { method: "POST", body: JSON.stringify(["TTL", redisKey]) }).then((response) => response.json());
    assert.equal(ttl.result, -1);

    loseNextRedisResponse((command) => command[0] === "EVAL");
    assert.equal((await publicationLock(request("release", firstOwner))).status, 200);
    assert.equal((await publicationLock(request("release", firstOwner))).status, 200);
    assert.equal((await publicationLock(request("acquire", secondOwner))).status, 200);
    const thirdOwner = "c".repeat(32);
    for (let attempt = 0; attempt < 57; attempt += 1) {
      assert.equal((await publicationLock(request("acquire", thirdOwner))).status, 409);
    }
    assert.equal((await publicationLock(request("acquire", thirdOwner))).status, 429);
    assert.equal((await publicationLock(request("release", secondOwner))).status, 200);
    assert.equal((await publicationLock(request("acquire", firstOwner, "wrong"))).status, 401);
    assert.equal((await publicationLock(request("acquire", firstOwner, "maintainer-secret", "other/results"))).status, 400);
  });
});
