import test from "node:test";
import assert from "node:assert/strict";
import poll from "../../../api/worker/auth/poll.ts";
import { authenticate } from "../../../api/worker/_lib/auth.ts";
import { appGrantEnabled } from "../../../api/worker/_lib/user-grants.ts";

const redisUrl = "https://redis.test";
const redisToken = "redis-token";

function setEnvironment(values) {
  const original = { ...process.env };
  Object.assign(process.env, values);
  return () => {
    for (const key of Object.keys(values)) {
      if (original[key] === undefined) delete process.env[key];
      else process.env[key] = original[key];
    }
  };
}

test("GitHub App grant configuration fails closed when partially configured", () => {
  const restoreEnvironment = setEnvironment({ VOLUNTEER_GITHUB_APP_CLIENT_ID: "app-client" });
  try {
    assert.throws(() => appGrantEnabled(), /configuration is incomplete/i);
  } finally {
    restoreEnvironment();
  }
});

test("device polling issues a credential to the repository maintainer", async () => {
  const restoreEnvironment = setEnvironment({
    GITHUB_OAUTH_CLIENT_SECRET: "client-secret",
    UPSTASH_REDIS_REST_URL: redisUrl,
    UPSTASH_REDIS_REST_TOKEN: redisToken,
  });
  const originalFetch = globalThis.fetch;
  const sessionId = "valid-session-id";
  const stored = new Map([[`euroeval:worker:device:${sessionId}`, JSON.stringify({
    device_code: "device-code", client_id: "client-id", interval: 5,
  })]]);
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input);
    if (url === redisUrl) {
      const command = JSON.parse(init.body);
      if (command[0] === "GET") return Response.json({ result: stored.get(command[1]) ?? null });
      if (command[0] === "INCR") return Response.json({ result: 1 });
      if (command[0] === "DEL") { stored.delete(command[1]); return Response.json({ result: 1 }); }
      if (command[0] === "SET") { stored.set(command[1], command[2]); return Response.json({ result: "OK" }); }
      if (command[0] === "EXPIRE") return Response.json({ result: 1 });
      throw new Error(`Unexpected Redis command: ${command[0]}`);
    }
    if (url === "https://github.com/login/oauth/access_token") {
      return Response.json({ access_token: "temporary-access-token" });
    }
    if (url === "https://api.github.com/user") return Response.json({ login: "saattrupdan" });
    if (url === "https://api.github.com/applications/client-id/grant") {
      assert.equal(init.method, "DELETE");
      return new Response(null, { status: 204 });
    }
    throw new Error(`Unexpected request: ${url}`);
  };

  try {
    const response = await poll(new Request("https://euroeval.test/api/worker/auth/poll", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", session_id: sessionId }),
    }));
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.status, "authorised");
    assert.equal(body.github_login, "saattrupdan");
    assert.equal(typeof body.credential, "string");
    assert.equal(stored.has(`euroeval:worker:device:${sessionId}`), false);
  } finally {
    globalThis.fetch = originalFetch;
    restoreEnvironment();
  }
});

test("device polling rejects an unidentifiable GitHub user", async () => {
  const restoreEnvironment = setEnvironment({
    GITHUB_OAUTH_CLIENT_SECRET: "client-secret",
    UPSTASH_REDIS_REST_URL: redisUrl,
    UPSTASH_REDIS_REST_TOKEN: redisToken,
  });
  const originalFetch = globalThis.fetch;
  const sessionId = "missing-login-session";
  const deviceKey = `euroeval:worker:device:${sessionId}`;
  const stored = new Map([[deviceKey, JSON.stringify({ device_code: "device-code", client_id: "client-id", interval: 5 })]]);
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input);
    if (url === redisUrl) {
      const command = JSON.parse(init.body);
      if (command[0] === "GET") return Response.json({ result: stored.get(command[1]) ?? null });
      if (command[0] === "INCR") return Response.json({ result: 1 });
      if (command[0] === "DEL") { stored.delete(command[1]); return Response.json({ result: 1 }); }
      if (command[0] === "EXPIRE") return Response.json({ result: 1 });
      throw new Error(`Unexpected Redis command: ${command[0]}`);
    }
    if (url === "https://github.com/login/oauth/access_token") return Response.json({ access_token: "temporary-access-token" });
    if (url === "https://api.github.com/user") return Response.json({});
    throw new Error(`Unexpected request: ${url}`);
  };

  try {
    const response = await poll(new Request("https://euroeval.test/api/worker/auth/poll", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", session_id: sessionId }),
    }));
    assert.equal(response.status, 401);
    assert.deepEqual(await response.json(), { error: "GitHub could not identify this worker." });
    assert.equal(stored.has(deviceKey), false);
  } finally {
    globalThis.fetch = originalFetch;
    restoreEnvironment();
  }
});

test("authenticate accepts a maintainer credential and rejects missing identities", async () => {
  const restoreEnvironment = setEnvironment({
    UPSTASH_REDIS_REST_URL: redisUrl,
    UPSTASH_REDIS_REST_TOKEN: redisToken,
  });
  const originalFetch = globalThis.fetch;
  const hashCredential = async (credential) => Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(credential))),
    (byte) => byte.toString(16).padStart(2, "0"),
  ).join("");
  const credentials = new Map([
    [`euroeval:worker:credential:${await hashCredential("maintainer-credential")}`, { contributor: "saattrupdan" }],
    [`euroeval:worker:credential:${await hashCredential("missing")}`, null],
    [`euroeval:worker:credential:${await hashCredential("empty")}`, { contributor: "" }],
  ]);
  globalThis.fetch = async (_input, init = {}) => {
    const command = JSON.parse(init.body);
    if (command[0] !== "GET") throw new Error(`Unexpected Redis command: ${command[0]}`);
    const worker = credentials.get(command[1]);
    return Response.json({ result: worker ? JSON.stringify(worker) : null });
  };

  try {
    const identity = await authenticate(new Request("https://euroeval.test", {
      headers: { authorization: "Bearer maintainer-credential" },
    }));
    assert.equal(identity.contributor, "saattrupdan");
    assert.equal(typeof identity.hash, "string");
    await assert.rejects(
      authenticate(new Request("https://euroeval.test", { headers: { authorization: "Bearer missing" } })),
      (error) => error.status === 401 && /invalid or revoked/.test(error.message),
    );
    await assert.rejects(
      authenticate(new Request("https://euroeval.test", { headers: { authorization: "Bearer empty" } })),
      (error) => error.status === 401 && /invalid or revoked/.test(error.message),
    );
    await assert.rejects(
      authenticate(new Request("https://euroeval.test")),
      (error) => error.status === 401 && /Authorization: Bearer/.test(error.message),
    );
  } finally {
    globalThis.fetch = originalFetch;
    restoreEnvironment();
  }
});
