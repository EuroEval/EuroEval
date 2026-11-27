import test from "node:test";
import assert from "node:assert/strict";
import { once } from "node:events";
import { createHash } from "node:crypto";
import { Readable } from "node:stream";
import http from "node:http";
import { createApiServer } from "../../dist-api/api-server/server.js";
import { redisFixture } from "./redis-rest-fixture.mjs";

const protocol = "volunteer-worker/v1";

test("standalone API preserves browser and worker workflows over HTTP", async (t) => {
  const originalEnv = { ...process.env };
  const originalFetch = globalThis.fetch;
  t.after(() => { process.env = originalEnv; globalThis.fetch = originalFetch; });
  for (const key of Object.keys(process.env)) {
    if (key.startsWith("VOLUNTEER_GITHUB_APP_")) delete process.env[key];
  }
  Object.assign(process.env, {
    UPSTASH_REDIS_REST_URL: "https://redis.test", UPSTASH_REDIS_REST_TOKEN: "redis-token",
    GITHUB_TOKEN: "fabricated-github-token", GITHUB_OAUTH_CLIENT_ID: "fake-client",
    WORKER_COORDINATOR_SECRET: "fake-coordinator", VOLUNTEER_PROMOTION_SECRET: "fake-promotion",
    EUROEVAL_VERSION: "20.0.0", VOLUNTEER_WORKER_VERSION: "1.0.0",
    VOLUNTEER_WORKER_IMAGE_DIGEST: `sha256:${"a".repeat(64)}`,
    VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64: `sha256:${"b".repeat(64)}`,
  });
  const redis = await redisFixture(t);
  const server = createApiServer({ evaluatedModels: Buffer.from(JSON.stringify({ "org/done": ["Greek"] })) });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const origin = `http://127.0.0.1:${server.address().port}`;
  t.after(async () => { server.closeAllConnections(); await new Promise((resolve) => server.close(resolve)); });
  const localOrigins = new Set([origin]);
  let created = 0;
  let indexUnavailable = false;
  globalThis.fetch = async (input, init = {}) => {
    const url = new URL(String(input));
    if (localOrigins.has(url.origin)) {
      if (indexUnavailable && url.pathname === "/evaluated-models.json") return new Response(null, { status: 503 });
      return originalFetch(input, init);
    }
    if (url.origin === "https://redis.test") return redis.respond(init);
    if (url.origin === "https://huggingface.co" && url.pathname.startsWith("/api/models/")) return Response.json({ private: false });
    if (url.href === "https://github.com/login/device/code") {
      return Response.json({ device_code: "device", user_code: "ABCD", verification_uri: "https://github.com/login/device", expires_in: 600, interval: 5 });
    }
    if (url.origin === "https://api.github.com" && url.pathname === "/search/issues") return Response.json({ items: [] });
    if (url.href.startsWith("https://api.github.com/repos/EuroEval/EuroEval/issues")) {
      if (init.method === "POST") {
        created++;
        return Response.json({ number: 42, html_url: "https://github.com/EuroEval/EuroEval/issues/42" });
      }
      return Response.json(url.searchParams.get("state") === "closed" && url.searchParams.get("page") === "3" ? [{ number: 43 }] : []);
    }
    throw new Error(`Unexpected external request: ${url.origin}${url.pathname}`);
  };
  const request = (path, init = {}) => originalFetch(`${origin}${path}`, init);
  const post = (path, body, headers = {}) => request(path, { method: "POST", body: typeof body === "string" ? body : JSON.stringify(body), headers: { "content-type": "application/json", ...headers } });

  await t.test("browser queries preserve parameters and override wildcard CORS", async () => {
    const response = await request("/api/issues?state=closed&page=3", { headers: { Origin: "https://euroeval.com" } });
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), [{ number: 43 }]);
    assert.equal(response.headers.get("access-control-allow-origin"), "https://euroeval.com");
    assert.match(response.headers.get("vary"), /Origin/);
    assert.match(response.headers.get("cache-control"), /max-age/);
  });
  await t.test("preflight is route-specific and does not weaken authorisation", async () => {
    const response = await request("/api/worker/coordinator-lock", { method: "OPTIONS", headers: {
      Origin: "https://euroeval.com", "access-control-request-method": "POST", "access-control-request-headers": "Content-Type, X-Coordinator-Secret",
    } });
    assert.equal(response.status, 204);
    assert.equal(await response.text(), "");
    assert.match(response.headers.get("access-control-allow-headers"), /x-coordinator-secret/);
    assert.equal((await post("/api/worker/coordinator-lock", {}, { Origin: "https://euroeval.com", "x-coordinator-secret": "wrong" })).status, 401);
    assert.equal((await post("/api/worker/promotion-status", {}, { "x-promotion-secret": "wrong" })).status, 401);
    assert.equal((await request("/api/issues", { method: "OPTIONS", headers: { Origin: "https://euroeval.com", "access-control-request-method": "DELETE" } })).status, 403);
    assert.equal((await post("/api/submit-evaluation", {}, { Origin: "https://attacker.test" })).status, 403);
    assert.equal(created, 0);
  });
  await t.test("unknown and wrong-method routes cannot expose files or helpers", async () => {
    for (const path of ["/api/worker/_lib", "/api-server/main.js", "/api/missing"]) {
      assert.equal((await request(path)).status, 404);
      assert.equal((await request(path, { method: "OPTIONS" })).status, 404);
    }
    const response = await request("/api/worker/promotion-status");
    assert.equal(response.status, 405);
    assert.match(response.headers.get("allow"), /POST/);
  });
  await t.test("worker release queries work without browser Origin headers", async () => {
    const response = await request("/api/worker/release-info?architecture=arm64");
    assert.equal(response.status, 200);
    assert.equal((await response.json()).image_digest, `sha256:${"b".repeat(64)}`);
    assert.equal(response.headers.get("cache-control"), "no-store");
  });
  await t.test("named Fetch exports preserve authentication and malformed-body responses", async () => {
    const credential = "fabricated-worker-credential";
    const hash = createHash("sha256").update(credential).digest("hex");
    redis.execute(["SET", `euroeval:worker:credential:${hash}`, JSON.stringify({ contributor: "alice" })]);
    for (const path of ["result", "finalise"]) {
      const denied = await post(`/api/worker/${path}`, {});
      assert.equal(denied.status, 401);
      const malformed = await post(`/api/worker/${path}`, "{", { authorization: `Bearer ${credential}`, Origin: "https://euroeval.com" });
      assert.equal(malformed.status, 400);
      assert.equal(malformed.headers.get("access-control-allow-origin"), "https://euroeval.com");
    }
  });
  await t.test("transport bounds declared and chunked bodies without shrinking promotion limits", async () => {
    const withinPromotionLimit = await post("/api/worker/promote", { padding: "x".repeat(100_000) }, { "x-promotion-secret": "fake-promotion" });
    assert.equal(withinPromotionLimit.status, 400);
    const big = Buffer.alloc(2 * 1024 * 1024 + 1, 120);
    assert.equal((await post("/api/worker/result", big.toString())).status, 413);
    const chunked = await request("/api/worker/result", { method: "POST", body: Readable.from([big.subarray(0, 1024), big.subarray(1024)]), duplex: "half" });
    assert.equal(chunked.status, 413);
    assert.equal((await request("/evaluated-models.json")).status, 200);
  });
  await t.test("submission uses the packaged index and fails closed when it is unavailable", async () => {
    const body = { model_id: "org/done", language_groups: ["Greek"] };
    const duplicate = await post("/api/submit-evaluation", body, { host: "attacker.test", Origin: "https://euroeval.com" });
    assert.equal(duplicate.status, 409);
    assert.equal(created, 0);
    indexUnavailable = true;
    try { assert.equal((await post("/api/submit-evaluation", body)).status, 503); }
    finally { indexUnavailable = false; }
    const accepted = await post("/api/submit-evaluation", { ...body, model_id: "org/new" });
    assert.equal(accepted.status, 200);
    assert.equal((await accepted.json()).number, 42);
    assert.equal(created, 1);
  });
  await t.test("IPv6 binding keeps submission index lookup independent of Host", async () => {
    const ipv6 = createApiServer({ evaluatedModels: Buffer.from(JSON.stringify({ "org/done": ["Greek"] })) });
    ipv6.listen(0, "::1");
    await once(ipv6, "listening");
    const local = `http://[::1]:${ipv6.address().port}`;
    localOrigins.add(local);
    try {
      const response = await originalFetch(`${local}/api/submit-evaluation`, {
        method: "POST", headers: { "content-type": "application/json", host: "attacker.test" },
        body: JSON.stringify({ model_id: "org/done", language_groups: ["Greek"] }),
      });
      assert.equal(response.status, 409);
    } finally {
      ipv6.closeAllConnections();
      await new Promise((resolve) => ipv6.close(resolve));
      localOrigins.delete(local);
    }
  });
  await t.test("forwarded-IP and localhost-referrer forgery cannot evade worker rate limits", async () => {
    for (let attempt = 0; attempt < 6; attempt++) {
      const response = await post("/api/worker/auth/start", { protocol_version: protocol }, {
        "x-forwarded-for": `10.0.0.${attempt}`, "x-vercel-forwarded-for": `10.0.1.${attempt}`,
        Referer: "http://localhost:5173", Origin: "https://euroeval.com",
      });
      assert.equal(response.status, attempt < 5 ? 200 : 429);
      if (attempt < 5) assert.equal((await response.json()).user_code, "ABCD");
      else assert.ok(Number(response.headers.get("retry-after")) > 0);
    }
  });
  await t.test("a disconnected upload does not stop subsequent requests", async () => {
    const aborted = http.request(`${origin}/api/worker/result`, { method: "POST" });
    aborted.on("error", () => {});
    aborted.write("{");
    aborted.destroy();
    assert.equal((await request("/evaluated-models.json")).status, 200);
  });
});
