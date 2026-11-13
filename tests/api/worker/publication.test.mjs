import test from "node:test";
import assert from "node:assert/strict";
import publicationLock from "../../../api/worker/publication-lock.ts";

test("publication lock serializes owners and supports safe lost-response retries", async () => {
  const oldFetch = globalThis.fetch;
  const oldEnv = { ...process.env };
  Object.assign(process.env, { CANONICAL_PUBLICATION_SECRET: "maintainer-secret", UPSTASH_REDIS_REST_URL: "https://redis.test", UPSTASH_REDIS_REST_TOKEN: "redis-token" });
  const values = new Map();
  globalThis.fetch = async (_url, init) => {
    const command = JSON.parse(init.body);
    let result;
    if (command[0] === "INCR") result = "1";
    else if (command[0] === "EXPIRE") result = 1;
    else if (command[0] === "EVAL") {
      const [script, count, key, owner] = command.slice(1);
      assert.equal(count, "1");
      assert.doesNotMatch(script, /EXPIRE|['"]EX['"]/);
      if (script.includes("SET',KEYS[1],ARGV[1]")) {
        if (!values.has(key)) { values.set(key, owner); result = "owned"; }
        else result = values.get(key) === owner ? "owned" : "busy";
      } else if (script.includes("DEL")) {
        if (values.get(key) === owner) { values.delete(key); result = "released"; }
        else result = values.has(key) ? "busy" : "released";
      } else result = values.get(key) === owner ? "owned" : "busy";
    } else if (command[0] === "GET") result = values.get(command[1]) || null;
    return Response.json({ result });
  };
  const request = (action, owner, secret = "maintainer-secret", bucket = "EuroEval/results") => new Request("https://euroeval.com/api/worker/publication-lock", {
    method: "POST", headers: { authorization: `Bearer ${secret}`, "content-type": "application/json" },
    body: JSON.stringify({ action, bucket, owner }),
  });
  try {
    const firstOwner = "a".repeat(32); const secondOwner = "b".repeat(32);
    assert.equal((await publicationLock(request("acquire", firstOwner))).status, 200);
    assert.equal((await publicationLock(request("acquire", firstOwner))).status, 200);
    assert.equal((await publicationLock(request("acquire", secondOwner))).status, 409);
    assert.equal((await publicationLock(request("release", secondOwner))).status, 409);
    assert.equal((await publicationLock(request("status", firstOwner))).status, 200);
    assert.equal((await publicationLock(request("release", firstOwner))).status, 200);
    assert.equal((await publicationLock(request("release", firstOwner))).status, 200);
    assert.equal((await publicationLock(request("acquire", secondOwner))).status, 200);
    assert.equal((await publicationLock(request("acquire", firstOwner, "wrong"))).status, 401);
    assert.equal((await publicationLock(request("acquire", firstOwner, "maintainer-secret", "other/results"))).status, 400);
    assert.equal(values.size, 1); // No lease TTL or implicit takeover.
  } finally { globalThis.fetch = oldFetch; process.env = oldEnv; }
});
