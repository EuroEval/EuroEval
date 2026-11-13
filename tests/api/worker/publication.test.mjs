import test from "node:test";
import assert from "node:assert/strict";
import publicationLock from "../../../api/worker/publication-lock.ts";
import { withRedisRestBridge } from "./redis-bridge.mjs";

const request = (action, owner, secret = "maintainer-secret", bucket = "EuroEval/results") => new Request("https://euroeval.com/api/worker/publication-lock", {
  method: "POST", headers: { authorization: `Bearer ${secret}`, "content-type": "application/json" },
  body: JSON.stringify({ action, bucket, owner }),
});

test("publication lock runs its Lua against Redis and survives contention and lost replies", async () => {
  await withRedisRestBridge(async ({ loseNextRedisResponse }) => {
    process.env.CANONICAL_PUBLICATION_SECRET = "maintainer-secret";
    const firstOwner = "a".repeat(32);
    const secondOwner = "b".repeat(32);
    loseNextRedisResponse((command) => command[0] === "EVAL" && command[1].includes("local value=redis.call('GET',KEYS[1])"));
    assert.equal((await publicationLock(request("acquire", firstOwner))).status, 200);
    assert.equal((await publicationLock(request("acquire", firstOwner))).status, 200);
    assert.equal((await publicationLock(request("assert", firstOwner))).status, 200);
    assert.equal((await publicationLock(request("acquire", secondOwner))).status, 409);
    assert.equal((await publicationLock(request("assert", secondOwner))).status, 409);
    assert.equal((await publicationLock(request("release", secondOwner))).status, 409);
    assert.equal((await publicationLock(request("status", firstOwner))).status, 200);

    const redisKey = "euroeval:worker:canonical-publication:EuroEval/results";
    const ttl = await fetch("https://redis.test", { method: "POST", body: JSON.stringify(["TTL", redisKey]) }).then((response) => response.json());
    assert.equal(ttl.result, -1);

    assert.equal((await publicationLock(request("release", firstOwner))).status, 200);
    assert.equal((await publicationLock(request("release", firstOwner))).status, 200);
    assert.equal((await publicationLock(request("acquire", secondOwner))).status, 200);
    assert.equal((await publicationLock(request("acquire", firstOwner, "wrong"))).status, 401);
    assert.equal((await publicationLock(request("acquire", firstOwner, "maintainer-secret", "other/results"))).status, 400);
  });
});
