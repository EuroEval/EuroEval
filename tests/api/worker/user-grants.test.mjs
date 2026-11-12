import test from "node:test";
import assert from "node:assert/strict";
import { appGrantEnabled, ensureVolunteerComment, grantCredentialHash, retainUserGrant, revokeUserGrant, verifyAppInstallation } from "../../../api/worker/_lib/user-grants.ts";
import { authenticate } from "../../../api/worker/_lib/auth.ts";
import startAuthorization from "../../../api/worker/auth/start.ts";

const redisUrl = "https://redis.test";
const contributor = { id: 42, login: "volunteer" };

function environment(values) {
  const previous = new Map(Object.keys(values).map((key) => [key, process.env[key]]));
  for (const [key, value] of Object.entries(values)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  return () => {
    for (const [key, value] of previous) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  };
}

async function fixture(options = {}) {
  const pair = await crypto.subtle.generateKey({ name: "RSASSA-PKCS1-v1_5", modulusLength: 2048, publicExponent: new Uint8Array([1, 0, 1]), hash: "SHA-256" }, true, ["sign", "verify"]);
  const privateDer = new Uint8Array(await crypto.subtle.exportKey("pkcs8", pair.privateKey));
  const privatePem = `-----BEGIN PRIVATE KEY-----\n${Buffer.from(privateDer).toString("base64")}\n-----END PRIVATE KEY-----`;
  const encryptionKey = crypto.getRandomValues(new Uint8Array(32));
  const restore = environment({
    UPSTASH_REDIS_REST_URL: redisUrl,
    UPSTASH_REDIS_REST_TOKEN: "redis-test-token",
    VOLUNTEER_GITHUB_APP_CLIENT_ID: "Iv1.test-client",
    VOLUNTEER_GITHUB_APP_CLIENT_SECRET: "test-client-secret",
    VOLUNTEER_GITHUB_APP_ID: "12345",
    VOLUNTEER_GITHUB_APP_PRIVATE_KEY: privatePem,
    VOLUNTEER_GITHUB_APP_INSTALLATION_ID: "67890",
    VOLUNTEER_GITHUB_APP_REPOSITORY_ID: "386992479",
    VOLUNTEER_GITHUB_APP_TOKEN_ENCRYPTION_KEY: Buffer.from(encryptionKey).toString("base64"),
  });
  const values = new Map();
  const api = { comments: [], requests: [], installation: { id: 67890, account: { login: "EuroEval", type: "Organization" }, repository_selection: "selected", permissions: { issues: "write", metadata: "read" } }, repositories: [{ id: 386992479, full_name: "EuroEval/EuroEval" }], commentPages: 0 };
  const oldFetch = globalThis.fetch;
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input);
    const method = init.method || "GET";
    api.requests.push({ url, method, headers: init.headers || {}, body: init.body ? JSON.parse(init.body) : undefined });
    if (url === redisUrl) return Response.json({ result: redisCommand(values, JSON.parse(init.body)) });
    if (url === "https://github.com/login/device/code") return Response.json({ device_code: "device-code", user_code: "ABCD-EFGH", verification_uri: "https://github.com/login/device", interval: 5, expires_in: 600 });
    if (url === "https://api.github.com/app/installations/67890") return Response.json(options.installation || api.installation);
    if (url === "https://api.github.com/app/installations/67890/access_tokens" && method === "POST") return Response.json({ token: "temporary-install-token", permissions: options.tokenPermissions || { issues: "write", metadata: "read" } });
    if (url.startsWith("https://api.github.com/installation/repositories")) return Response.json({ repositories: options.repositories || api.repositories });
    if (url === "https://api.github.com/installation/token" && method === "DELETE") { api.revokedInstallationTokens = (api.revokedInstallationTokens || 0) + 1; return new Response(null, { status: 204 }); }
    if (url === "https://api.github.com/user") return Response.json(options.identity || contributor);
    if (url === "https://api.github.com/applications/Iv1.test-client/token") {
      if (method === "DELETE") {
        api.credentialExistedAtRevoke = values.has(`euroeval:worker:credential:${await grantCredentialHash("broker-credential")}`);
        api.grantExistedAtRevoke = values.has(`euroeval:worker:user-grant:${await grantCredentialHash("broker-credential")}`);
        api.revokedUserTokens = [...(api.revokedUserTokens || []), JSON.parse(init.body).access_token];
        return new Response(null, { status: 204 });
      }
      const token = JSON.parse(init.body).access_token;
      return Response.json(options.tokenCheck?.(token) || { token, app: { client_id: "Iv1.test-client", id: 12345 }, user: contributor });
    }
    if (url === "https://api.github.com/repos/EuroEval/EuroEval/issues/7/comments?per_page=100&page=1") {
      api.commentPages++;
      return Response.json(options.comments || api.comments);
    }
    if (url === "https://api.github.com/repos/EuroEval/EuroEval/issues/7/comments" && method === "POST") {
      api.postCount = (api.postCount || 0) + 1;
      api.comments.push({ body: JSON.parse(init.body).body, user: contributor });
      return Response.json(api.comments.at(-1));
    }
    if (url === "https://api.github.com/repos/EuroEval/EuroEval/issues/7/comments?per_page=100&page=2") return Response.json(options.commentPageTwo || []);
    throw new Error(`Unexpected GitHub request: ${method} ${url}`);
  };
  return {
    api, values,
    close() { globalThis.fetch = oldFetch; restore(); },
    async retain({ credential = "broker-credential", access = "access-one", refresh = "refresh-one", expectedLogin = contributor.login, expires = Date.now() + 3600_000, refreshExpires = Date.now() + 86400_000 } = {}) {
      const hash = await grantCredentialHash(credential);
      await saveCredential(values, hash, expectedLogin);
      await retainUserGrant(access, refresh, expires, refreshExpires, credential, expectedLogin);
      return { contributor: expectedLogin, hash };
    },
  };
}

function redisCommand(values, command) {
  const [verb, ...args] = command;
  if (verb === "GET") return values.get(args[0]) ?? null;
  if (verb === "EXISTS") return values.has(args[0]) ? 1 : 0;
  if (verb === "SET") {
    if (args.includes("NX") && values.has(args[0])) return null;
    values.set(args[0], args[1]);
    return "OK";
  }
  if (verb === "DEL") { let count = 0; for (const key of args) count += values.delete(key) ? 1 : 0; return count; }
  if (verb === "EXPIRE") return values.has(args[0]) ? 1 : 0;
  if (verb === "INCR") return 1;
  if (verb !== "EVAL") throw new Error(`Unexpected Redis command: ${verb}`);
  const [script, countText, ...rest] = args;
  const count = Number(countText);
  const keys = rest.slice(0, count);
  const argv = rest.slice(count);
  if (script.includes("EXISTS, KEYS[1]")) { if (values.has(keys[0]) && values.has(keys[1])) { values.delete(keys[0]); return 1; } return 0; }
  if (script.includes("local old = redis.call('GET', KEYS[3])")) {
    if (values.get(keys[0]) !== argv[0]) return 0;
    if (!values.has(keys[1])) return -1;
    const old = values.get(keys[2]) ?? null;
    if (argv[1] !== "" && (!old || JSON.parse(old).generation !== argv[1])) return -2;
    values.set(keys[2], argv[2]);
    return 1;
  }
  if (script.includes("local old = redis.call('GET', KEYS[2])")) {
    if (values.get(keys[0]) !== argv[0]) return 0;
    const old = values.get(keys[1]) ?? null;
    if (argv[1] !== "" && old !== argv[1]) return -1;
    values.delete(keys[1]); values.delete(keys[2]);
    return 1;
  }
  if (script.includes("redis.call('EXPIRE'")) return values.get(keys[0]) === argv[0] ? 1 : 0;
  if (script.includes("redis.call('DEL'")) {
    if (values.get(keys[0]) !== argv[0]) return 0;
    values.delete(keys[0]); return 1;
  }
  throw new Error(`Unexpected Redis script ${script}`);
}

async function saveCredential(values, hash, contributorName = contributor.login, mode = "app") {
  values.set(`euroeval:worker:credential:${hash}`, JSON.stringify({ contributor: contributorName, grant_mode: mode }));
}

test("App device authorization requests offline access scoped to the exact repository", async () => {
  const fx = await fixture();
  try {
    const response = await startAuthorization(new Request("https://worker.test/api/worker/auth/start", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1" }),
    }));
    assert.equal(response.status, 200);
    const deviceRequest = fx.api.requests.find((request) => request.url === "https://github.com/login/device/code");
    assert.equal(deviceRequest.body.scope, "offline_access");
    assert.equal(deviceRequest.body.repository_id, 386992479);
    const started = await response.json();
    const persisted = JSON.parse(fx.values.get(`euroeval:worker:device:${started.session_id}`));
    assert.equal(persisted.app_mode, true);
    assert.equal(persisted.client_id, "Iv1.test-client");
  } finally { fx.close(); }
});

test("GitHub App config is opt-in, complete, and constrained to the production repository", async () => {
  const keys = ["VOLUNTEER_GITHUB_APP_CLIENT_ID", "VOLUNTEER_GITHUB_APP_CLIENT_SECRET", "VOLUNTEER_GITHUB_APP_ID", "VOLUNTEER_GITHUB_APP_PRIVATE_KEY", "VOLUNTEER_GITHUB_APP_INSTALLATION_ID", "VOLUNTEER_GITHUB_APP_REPOSITORY_ID", "VOLUNTEER_GITHUB_APP_TOKEN_ENCRYPTION_KEY"];
  const old = new Map(keys.map((key) => [key, process.env[key]]));
  try {
    keys.forEach((key) => delete process.env[key]);
    assert.equal(appGrantEnabled(), false);
    process.env.VOLUNTEER_GITHUB_APP_CLIENT_ID = "partial";
    assert.throws(() => appGrantEnabled(), /incomplete/i);
    const fx = await fixture();
    try {
      assert.equal(appGrantEnabled(), true);
      process.env.VOLUNTEER_GITHUB_APP_REPOSITORY_ID = "1";
      assert.throws(() => appGrantEnabled(), /repository configuration/i);
    } finally { fx.close(); }
  } finally {
    for (const [key, value] of old) if (value === undefined) delete process.env[key]; else process.env[key] = value;
  }
});

test("installation verification checks exact issue-write and metadata-read access and always revokes temporary tokens", async (t) => {
  for (const invalid of [
    { installation: { id: 67890, account: { login: "someone-else" }, repository_selection: "selected", permissions: { issues: "write", metadata: "read" } } },
    { installation: { id: 67890, account: { login: "EuroEval" }, repository_selection: "all", permissions: { issues: "write", metadata: "read" } } },
    { installation: { id: 67890, account: { login: "EuroEval" }, repository_selection: "selected", permissions: { issues: "write", metadata: "read", contents: "read" } } },
    { tokenPermissions: { issues: "write", metadata: "read", contents: "read" } },
    { repositories: [{ id: 386992479, full_name: "EuroEval/EuroEval" }, { id: 2, full_name: "EuroEval/other" }] },
    { repositories: [{ id: 386992479, full_name: "EuroEval/other" }] },
  ]) {
    const fx = await fixture(invalid);
    try {
      await assert.rejects(verifyAppInstallation(), /invalid|only EuroEval/i);
      if (!invalid.installation) assert.equal(fx.api.revokedInstallationTokens, 1);
    } finally { fx.close(); }
  }
  const fx = await fixture();
  try {
    await verifyAppInstallation();
    assert.equal(fx.api.revokedInstallationTokens, 1);
    const minted = fx.api.requests.find((request) => request.url.endsWith("/access_tokens"));
    assert.deepEqual(minted.body.permissions, { issues: "write", metadata: "read" });
  } finally { fx.close(); }
});

test("renewable grant is encrypted, credential-bound, identity-verified, and refreshes with rotation", async () => {
  const fx = await fixture();
  try {
    const identity = await fx.retain({ expires: Date.now() + 10_000 });
    const key = `euroeval:worker:user-grant:${identity.hash}`;
    const envelope = JSON.parse(fx.values.get(key));
    assert.equal(envelope.version, 1);
    assert.doesNotMatch(fx.values.get(key), /access-one|refresh-one/);
    const before = envelope.generation;
    await fx.retain({ credential: "broker-credential", expires: Date.now() + 10_000 });
    const renewedEnvelope = JSON.parse(fx.values.get(key));
    assert.notEqual(renewedEnvelope.iv, envelope.iv, "each encryption uses a fresh AES-GCM nonce");

    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (input, init = {}) => {
      if (String(input) === "https://github.com/login/oauth/access_token") {
        fx.api.refreshRequest = JSON.parse(init.body);
        return Response.json({ access_token: "access-two", refresh_token: "refresh-two", expires_in: 3600, refresh_token_expires_in: 86400 });
      }
      return originalFetch(input, init);
    };
    try {
      await ensureVolunteerComment(identity, 7, async () => {});
      const rotated = JSON.parse(fx.values.get(key));
      assert.notEqual(rotated.generation, renewedEnvelope.generation);
      assert.notEqual(rotated.generation, before);
      assert.deepEqual(fx.api.refreshRequest, { client_id: "Iv1.test-client", client_secret: "test-client-secret", grant_type: "refresh_token", refresh_token: "refresh-one" });
      assert.equal(fx.api.comments[0].body, "I'd like to evaluate this.");
      assert.equal(fx.api.comments[0].user.login, contributor.login);
    } finally { globalThis.fetch = originalFetch; }
  } finally { fx.close(); }
});

test("grant ciphertext rejects cross-credential substitution and tampering", async () => {
  const fx = await fixture();
  try {
    const first = await fx.retain({ credential: "credential-one" });
    const second = await fx.retain({ credential: "credential-two" });
    const firstKey = `euroeval:worker:user-grant:${first.hash}`;
    const secondKey = `euroeval:worker:user-grant:${second.hash}`;
    const firstEnvelope = fx.values.get(firstKey);
    const secondEnvelope = fx.values.get(secondKey);
    fx.values.set(firstKey, secondEnvelope);
    await assert.rejects(ensureVolunteerComment(first, 7, async () => {}), /renewed/i);
    fx.values.set(firstKey, firstEnvelope);
    const envelope = JSON.parse(firstEnvelope);
    envelope.ciphertext = `${envelope.ciphertext.slice(0, -1)}${envelope.ciphertext.endsWith("A") ? "B" : "A"}`;
    fx.values.set(firstKey, JSON.stringify(envelope));
    await assert.rejects(ensureVolunteerComment(first, 7, async () => {}), /renewed/i);
    assert.equal(fx.api.postCount || 0, 0);
  } finally { fx.close(); }
});

test("comment writes require a live matching grant and a still-owned eligible lease", async () => {
  const fx = await fixture();
  try {
    const identity = await fx.retain();
    await assert.rejects(ensureVolunteerComment({ ...identity, contributor: "unrelated" }, 7, async () => {}), /renewed/i);
    await assert.rejects(ensureVolunteerComment(identity, 7, async () => { throw new Error("lease expired or mutex lost"); }), /lease expired or mutex lost/);
    await assert.rejects(ensureVolunteerComment(identity, 8, async () => {}), /Unexpected GitHub request/i);
    const expired = await fx.retain({ credential: "expired-grant", expires: Date.now() + 10_000, refreshExpires: Date.now() + 20_000 });
    const now = Date.now;
    Date.now = () => now() + 30_000;
    try { await assert.rejects(ensureVolunteerComment(expired, 7, async () => {}), /renewed/i); }
    finally { Date.now = now; }
    assert.equal(fx.api.postCount || 0, 0);
  } finally { fx.close(); }
});

test("retention rejects app-token, user-id, and login mismatches without accepting organization membership as identity", async () => {
  const fx = await fixture();
  try {
    const credential = "external-user";
    await saveCredential(fx.values, await grantCredentialHash(credential));
    await retainUserGrant("access-one", "refresh-one", Date.now() + 3600_000, Date.now() + 86400_000, credential, "volunteer");
    assert.ok(fx.values.has(`euroeval:worker:user-grant:${await grantCredentialHash(credential)}`));
  } finally { fx.close(); }
  for (const options of [
    { identity: { id: 42, login: "attacker" } },
    { tokenCheck: () => ({ token: "different-token", app: { client_id: "Iv1.test-client", id: 12345 }, user: contributor }) },
    { tokenCheck: () => ({ token: "access-one", app: { client_id: "another-app", id: 12345 }, user: contributor }) },
    { tokenCheck: () => ({ token: "access-one", app: { client_id: "Iv1.test-client", id: 12345 }, user: { id: 99, login: "volunteer" } }) },
  ]) {
    const other = await fixture(options);
    try { await assert.rejects(other.retain(), /identity|match/i); }
    finally { other.close(); }
  }
});

test("comment reconciliation paginates before posting and finds a prior exact claimant comment", async () => {
  const firstPage = Array.from({ length: 100 }, (_, index) => ({ body: `unrelated ${index}`, user: { id: 5, login: "other" } }));
  const fx = await fixture({ comments: firstPage, commentPageTwo: [{ body: "I'd like to evaluate this.", user: contributor }] });
  try {
    const identity = await fx.retain();
    await ensureVolunteerComment(identity, 7, async () => {});
    assert.equal(fx.api.commentPages, 1);
    assert.equal(fx.api.postCount || 0, 0);
  } finally { fx.close(); }
});

test("parallel comment reconciliation does not duplicate exact authenticated comments", async () => {
  const fx = await fixture({ comments: [{ body: "I'd like to evaluate this.", user: contributor }] });
  try {
    const identity = await fx.retain();
    const concurrent = await Promise.allSettled([ensureVolunteerComment(identity, 7, async () => {}), ensureVolunteerComment(identity, 7, async () => {})]);
    assert.equal(concurrent.some((result) => result.status === "rejected"), true);
    await ensureVolunteerComment(identity, 7, async () => {});
    assert.equal(fx.api.postCount || 0, 0);
  } finally { fx.close(); }
});

test("comment scan only deduplicates the exact author and body and preserves pending-post uncertainty", async () => {
  const fx = await fixture({ comments: [{ body: "I'd like to evaluate this.", user: { id: 8, login: "other" } }, { body: "I'd like to evaluate this!", user: contributor }] });
  try {
    const identity = await fx.retain();
    await ensureVolunteerComment(identity, 7, async () => {});
    assert.equal(fx.api.postCount, 1);
    const dedupKey = `euroeval:worker:user-grant:${identity.hash}:comment:7`;
    fx.values.set(dedupKey, "pending");
    await ensureVolunteerComment(identity, 7, async () => {});
    assert.equal(fx.api.postCount, 1);
  } finally { fx.close(); }
});

test("revocation removes local credential and grant before remote revocation; legacy credentials reauthenticate only when App mode is off", async () => {
  const fx = await fixture();
  try {
    const identity = await fx.retain();
    await saveCredential(fx.values, identity.hash);
    await revokeUserGrant(identity);
    assert.equal(fx.values.has(`euroeval:worker:credential:${identity.hash}`), false);
    assert.equal(fx.values.has(`euroeval:worker:user-grant:${identity.hash}`), false);
    assert.deepEqual(fx.api.revokedUserTokens, ["access-one"]);
    assert.equal(fx.api.credentialExistedAtRevoke, false);
    assert.equal(fx.api.grantExistedAtRevoke, false);
    await assert.rejects(ensureVolunteerComment(identity, 7, async () => {}), /renewed/i);
  } finally { fx.close(); }
  const legacy = await fixture();
  try {
    const legacyHash = Buffer.from(await crypto.subtle.digest("SHA-256", new TextEncoder().encode("legacy-secret"))).toString("hex");
    await saveCredential(legacy.values, legacyHash, "legacy-user", "legacy");
    await assert.rejects(authenticate(new Request("https://worker.test", { headers: { authorization: "Bearer legacy-secret" } })), (error) => error.status === 401);
    await revokeUserGrant({ contributor: "legacy-user", hash: legacyHash });
    assert.equal(legacy.values.has(`euroeval:worker:credential:${legacyHash}`), false);
  } finally { legacy.close(); }
});
