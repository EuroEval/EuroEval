import { BrokerError, ConfigurationError, env, randomToken, sha256 } from "./protocol.js";
import { redis, redisDelete, redisGet, redisGetRaw, redisSet } from "./redis.js";

/** A broker credential and the verified GitHub contributor that owns it. */
export type UserGrant = { contributor: string; hash: string };
type AppConfig = {
  clientId: string; clientSecret: string; appId: string; privateKey: string;
  installationId: string; repositoryId: string; encryptionKey: Promise<CryptoKey>;
};
type GrantRecord = {
  version: 1; generation: string; userId: number; contributor: string; credentialHash: string;
  clientId: string; access: string; refresh: string; accessExpires: number; refreshExpires: number;
};
type StoredEnvelope = { version: 1; generation: string; iv: string; ciphertext: string };
const CONFIG_NAMES = [
  "VOLUNTEER_GITHUB_APP_CLIENT_ID", "VOLUNTEER_GITHUB_APP_CLIENT_SECRET", "VOLUNTEER_GITHUB_APP_ID",
  "VOLUNTEER_GITHUB_APP_PRIVATE_KEY", "VOLUNTEER_GITHUB_APP_INSTALLATION_ID",
  "VOLUNTEER_GITHUB_APP_REPOSITORY_ID", "VOLUNTEER_GITHUB_APP_TOKEN_ENCRYPTION_KEY",
] as const;
const GRANT_PREFIX = "euroeval:worker:user-grant:";
const CREDENTIAL_PREFIX = "euroeval:worker:credential:";
const LOCK_TTL = 30;
const COMMENT = "I'd like to evaluate this.";
const REPO = "EuroEval/EuroEval";
const REPOSITORY_ID = "386992479";
const REQUIRED_PERMISSIONS = { issues: "write", metadata: "read" } as const;
declare const process: { env: Record<string, string | undefined> };

/** Enable App grants only for a complete, valid App configuration. */
export function appGrantEnabled(): boolean {
  const present = CONFIG_NAMES.filter((name) => !!process.env[name]);
  if (present.length === 0) return false;
  if (present.length !== CONFIG_NAMES.length) throw new ConfigurationError("GitHub App grant configuration is incomplete.");
  const repositoryId = process.env.VOLUNTEER_GITHUB_APP_REPOSITORY_ID;
  if (repositoryId !== REPOSITORY_ID) throw new ConfigurationError("GitHub App repository configuration is invalid.");
  const key = process.env.VOLUNTEER_GITHUB_APP_TOKEN_ENCRYPTION_KEY || "";
  try {
    if (decodeBase64(key).length !== 32) throw new Error();
  } catch { throw new ConfigurationError("GitHub App token encryption key must be 32 bytes in base64."); }
  return true;
}

function cryptoBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}
function decodeBase64(value: string): Uint8Array {
  const normalized = value.replace(/-/g, "+").replace(/_/g, "/");
  const decoded = atob(normalized + "=".repeat((4 - normalized.length % 4) % 4));
  return Uint8Array.from(decoded, (char) => char.charCodeAt(0));
}
function appConfig(): AppConfig {
  if (!appGrantEnabled()) throw new ConfigurationError("GitHub App grants are not configured.");
  const key = decodeBase64(env("VOLUNTEER_GITHUB_APP_TOKEN_ENCRYPTION_KEY"));
  const pem = env("VOLUNTEER_GITHUB_APP_PRIVATE_KEY").replace(/\\n/g, "\n");
  return {
    clientId: env("VOLUNTEER_GITHUB_APP_CLIENT_ID"), clientSecret: env("VOLUNTEER_GITHUB_APP_CLIENT_SECRET"),
    appId: env("VOLUNTEER_GITHUB_APP_ID"), privateKey: pem,
    installationId: env("VOLUNTEER_GITHUB_APP_INSTALLATION_ID"), repositoryId: env("VOLUNTEER_GITHUB_APP_REPOSITORY_ID"),
    encryptionKey: crypto.subtle.importKey("raw", cryptoBuffer(key), "AES-GCM", false, ["encrypt", "decrypt"]),
  };
}
async function request(url: string, init: RequestInit = {}): Promise<Response> {
  return fetch(url, {
    ...init,
    headers: { accept: "application/vnd.github+json", "x-github-api-version": "2022-11-28", ...(init.headers || {}) },
  });
}
function base64url(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes)).replace(/=/g, "").replace(/\+/g, "-").replace(/\//g, "_");
}
function appPrivateKey(config: AppConfig): Uint8Array {
  const body = config.privateKey.replace(/-----BEGIN PRIVATE KEY-----|-----END PRIVATE KEY-----/g, "").replace(/\s/g, "");
  try { return decodeBase64(body); } catch { throw new ConfigurationError("GitHub App private key must be a valid PKCS8 key."); }
}
async function appJwt(config: AppConfig): Promise<string> {
  const key = await crypto.subtle.importKey("pkcs8", cryptoBuffer(appPrivateKey(config)), { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" }, false, ["sign"]);
  const now = Math.floor(Date.now() / 1000);
  const header = base64url(new TextEncoder().encode(JSON.stringify({ alg: "RS256", typ: "JWT" })));
  const payload = base64url(new TextEncoder().encode(JSON.stringify({ iat: now - 30, exp: now + 540, iss: config.appId })));
  const content = `${header}.${payload}`;
  const signature = await crypto.subtle.sign("RSASSA-PKCS1-v1_5", key, cryptoBuffer(new TextEncoder().encode(content)));
  return `${content}.${base64url(new Uint8Array(signature))}`;
}
function basicAuth(config: AppConfig): string { return `Basic ${btoa(`${config.clientId}:${config.clientSecret}`)}`; }
function exactPermissions(value: Record<string, string> | undefined): boolean {
  return !!value && value.issues === REQUIRED_PERMISSIONS.issues && value.metadata === REQUIRED_PERMISSIONS.metadata &&
    Object.keys(value).length === 2;
}

/** Verify the App installation, its restricted permissions, and its exact repository selection. */
export async function verifyAppInstallation(): Promise<void> {
  const config = appConfig();
  const jwt = await appJwt(config);
  const response = await request(`https://api.github.com/app/installations/${encodeURIComponent(config.installationId)}`, { headers: { authorization: `Bearer ${jwt}` } });
  if (!response.ok) throw new BrokerError(503, "GitHub App installation could not be verified.");
  const installation = await response.json() as {
    id?: number; account?: { login?: string; type?: string }; repository_selection?: string;
    permissions?: Record<string, string>;
  };
  if (String(installation.id) !== config.installationId || installation.account?.login?.toLowerCase() !== "euroeval" ||
      installation.repository_selection !== "selected" || !exactPermissions(installation.permissions)) {
    throw new BrokerError(503, "GitHub App installation permissions or owner are invalid.");
  }
  const minted = await request(`https://api.github.com/app/installations/${encodeURIComponent(config.installationId)}/access_tokens`, {
    method: "POST", headers: { authorization: `Bearer ${jwt}`, "content-type": "application/json" },
    body: JSON.stringify({ permissions: REQUIRED_PERMISSIONS }),
  });
  if (!minted.ok) throw new BrokerError(503, "GitHub App repository access could not be verified.");
  const data = await minted.json() as { token?: string; permissions?: Record<string, string>; repositories_url?: string; repositories?: Array<{ id?: number; full_name?: string }> };
  if (!data.token) throw new BrokerError(503, "GitHub App repository access could not be verified.");
  try {
    if (!exactPermissions(data.permissions)) throw new Error();
    const repositories: Array<{ id?: number; full_name?: string }> = [];
    const repositoriesUrl = data.repositories_url || "https://api.github.com/installation/repositories";
    for (let page = 1; page <= 100; page++) {
      const separator = repositoriesUrl.includes("?") ? "&" : "?";
      const listed = await request(`${repositoriesUrl}${separator}per_page=100&page=${page}`, { headers: { authorization: `Bearer ${data.token}` } });
      if (!listed.ok) throw new Error();
      const body = await listed.json() as { repositories?: Array<{ id?: number; full_name?: string }> };
      const batch = body.repositories || [];
      repositories.push(...batch);
      if (batch.length < 100) break;
      if (page === 100) throw new Error();
    }
    if (repositories.length !== 1 || String(repositories[0]?.id) !== config.repositoryId ||
        repositories[0]?.full_name?.toLowerCase() !== REPO.toLowerCase()) throw new Error();
  } catch { throw new BrokerError(503, "GitHub App installation must select only EuroEval/EuroEval."); }
  finally {
    const revoked = await request("https://api.github.com/installation/token", { method: "DELETE", headers: { authorization: `Bearer ${data.token}` } }).catch(() => null);
    if (!revoked?.ok && revoked?.status !== 404) throw new BrokerError(503, "Temporary GitHub App token could not be revoked.");
  }
}

async function seal(record: GrantRecord, config: AppConfig): Promise<string> {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const aad = new TextEncoder().encode(`${record.credentialHash}:${record.generation}`);
  const key = await config.encryptionKey;
  const ciphertext = await crypto.subtle.encrypt({ name: "AES-GCM", iv: cryptoBuffer(iv), additionalData: cryptoBuffer(aad) }, key, cryptoBuffer(new TextEncoder().encode(JSON.stringify(record))));
  return JSON.stringify({ version: 1, generation: record.generation, iv: base64url(iv), ciphertext: base64url(new Uint8Array(ciphertext)) });
}
async function unseal(value: string, config: AppConfig, hash: string): Promise<GrantRecord> {
  try {
    const envelope = JSON.parse(value) as StoredEnvelope;
    if (envelope.version !== 1 || !envelope.generation) throw new Error();
    const plain = await crypto.subtle.decrypt({ name: "AES-GCM", iv: cryptoBuffer(decodeBase64(envelope.iv)), additionalData: cryptoBuffer(new TextEncoder().encode(`${hash}:${envelope.generation}`)) }, await config.encryptionKey, cryptoBuffer(decodeBase64(envelope.ciphertext)));
    const record = JSON.parse(new TextDecoder().decode(plain)) as GrantRecord;
    if (record.version !== 1 || record.credentialHash !== hash || record.generation !== envelope.generation || record.clientId !== config.clientId) throw new Error();
    return record;
  } catch { throw new BrokerError(401, "GitHub App authorization must be renewed.", "github_grant_invalid"); }
}

async function withGrantLock<T>(hash: string, operation: (token: string) => Promise<T>): Promise<T> {
  const key = `${GRANT_PREFIX}${hash}:lock`;
  const token = randomToken(18);
  if (!(await redisSet(key, token, LOCK_TTL, true))) throw new BrokerError(409, "GitHub authorization is busy; retry.");
  let stopped = false;
  const renewScript = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('EXPIRE', KEYS[1], ARGV[2]) else return 0 end";
  const releaseScript = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) else return 0 end";
  const timer = setInterval(() => {
    if (!stopped) void redis("EVAL", renewScript, "1", key, token, String(LOCK_TTL)).catch(() => undefined);
  }, 10_000);
  (timer as unknown as { unref?: () => void }).unref?.();
  try { return await operation(token); }
  finally {
    stopped = true;
    clearInterval(timer);
    await redis("EVAL", releaseScript, "1", key, token).catch(() => undefined);
  }
}

const persistScript = "if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end; if not redis.call('EXISTS', KEYS[2]) then return -1 end; local old = redis.call('GET', KEYS[3]); if ARGV[2] ~= '' and (not old or cjson.decode(old).generation ~= ARGV[2]) then return -2 end; redis.call('SET', KEYS[3], ARGV[3], 'EX', ARGV[4]); return 1";
const removeScript = "if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end; local old = redis.call('GET', KEYS[2]); if ARGV[2] ~= '' and old ~= ARGV[2] then return -1 end; redis.call('DEL', KEYS[2], KEYS[3]); return 1";
async function persistRecord(hash: string, encrypted: string, ttl: number, expectedGeneration = ""): Promise<void> {
  await withGrantLock(hash, async (lock) => {
    const current = await redisGetRaw(`${GRANT_PREFIX}${hash}`);
    const generation = current ? (JSON.parse(current) as StoredEnvelope).generation : "";
    if (expectedGeneration && generation !== expectedGeneration) throw new BrokerError(401, "GitHub App authorization must be renewed.", "github_grant_invalid");
    const envelope = JSON.parse(encrypted) as StoredEnvelope;
    const result = Number(await redis("EVAL", persistScript, "3", `${GRANT_PREFIX}${hash}:lock`, `${CREDENTIAL_PREFIX}${hash}`, `${GRANT_PREFIX}${hash}`, lock, generation, encrypted, String(Math.max(1, Math.min(ttl, 30 * 24 * 60 * 60)))));
    if (result !== 1) throw new BrokerError(result === -1 ? 401 : 409, result === -1 ? "Worker credential is revoked." : "GitHub authorization changed; retry.", result === -1 ? "github_grant_invalid" : undefined);
    if (!envelope.generation) throw new BrokerError(503, "GitHub App authorization could not be retained securely.");
  });
}

async function verifyUserToken(config: AppConfig, accessToken: string, expected?: GrantRecord): Promise<{ id: number; login: string }> {
  const identityResponse = await request("https://api.github.com/user", { headers: { authorization: `Bearer ${accessToken}` } });
  if (!identityResponse.ok) throw new BrokerError(401, "GitHub volunteer identity could not be verified.", "github_grant_invalid");
  const identity = await identityResponse.json() as { id?: number; login?: string };
  const appResponse = await request(`https://api.github.com/applications/${encodeURIComponent(config.clientId)}/token`, {
    method: "POST", headers: { authorization: basicAuth(config), "content-type": "application/json" }, body: JSON.stringify({ access_token: accessToken }),
  });
  if (!appResponse.ok) throw new BrokerError(401, "GitHub App authorization could not be verified.", "github_grant_invalid");
  const check = await appResponse.json() as { token?: string; app?: { client_id?: string; id?: number }; user?: { id?: number; login?: string } };
  if (!identity.id || !identity.login || check.token !== accessToken || check.app?.client_id !== config.clientId ||
      (check.app.id !== undefined && String(check.app.id) !== config.appId) || check.user?.id !== identity.id ||
      check.user?.login?.toLowerCase() !== identity.login.toLowerCase() ||
      (expected && (identity.id !== expected.userId || identity.login.toLowerCase() !== expected.contributor.toLowerCase()))) {
    throw new BrokerError(401, "GitHub App authorization identity did not match.", "github_grant_invalid");
  }
  return { id: identity.id, login: identity.login };
}

/** Verify and retain a renewable App grant without exposing its tokens. */
export async function retainUserGrant(accessToken: string, refreshToken: string, accessExpires: number, refreshExpires: number, credential: string, expectedLogin: string): Promise<void> {
  const config = appConfig();
  await verifyAppInstallation();
  const user = await verifyUserToken(config, accessToken);
  if (user.login.toLowerCase() !== expectedLogin.toLowerCase() || !refreshToken || accessExpires <= Date.now() || refreshExpires <= Date.now()) {
    throw new BrokerError(401, "GitHub volunteer identity or renewable authorization did not match.", "github_grant_invalid");
  }
  const hash = await sha256(credential);
  const record: GrantRecord = { version: 1, generation: randomToken(18), userId: user.id, contributor: user.login,
    credentialHash: hash, clientId: config.clientId, access: accessToken, refresh: refreshToken, accessExpires, refreshExpires };
  await persistRecord(hash, await seal(record, config), Math.ceil((refreshExpires - Date.now()) / 1000));
}

async function liveGrant(identity: UserGrant, config: AppConfig): Promise<GrantRecord> {
  return withGrantLock(identity.hash, async (lockToken) => {
    const encrypted = await redisGetRaw(`${GRANT_PREFIX}${identity.hash}`);
    if (!encrypted) throw new BrokerError(401, "GitHub App authorization must be renewed.", "github_grant_invalid");
    const grant = await unseal(encrypted, config, identity.hash);
    if (grant.contributor.toLowerCase() !== identity.contributor.toLowerCase() || grant.refreshExpires <= Date.now()) {
      throw new BrokerError(401, "GitHub App authorization must be renewed.", "github_grant_invalid");
    }
    if (grant.accessExpires > Date.now() + 60_000) return grant;
    const response = await request("https://github.com/login/oauth/access_token", {
      method: "POST", headers: { accept: "application/json", "content-type": "application/json" },
      body: JSON.stringify({ client_id: config.clientId, client_secret: config.clientSecret, grant_type: "refresh_token", refresh_token: grant.refresh }),
    });
    if (!response.ok) throw new BrokerError(401, "GitHub App authorization must be renewed.", "github_grant_invalid");
    const rotated = await response.json() as { access_token?: string; refresh_token?: string; expires_in?: number; refresh_token_expires_in?: number; error?: string };
    if (rotated.error || !rotated.access_token || !rotated.refresh_token || !rotated.expires_in || !rotated.refresh_token_expires_in) {
      throw new BrokerError(401, "GitHub App authorization must be renewed.", "github_grant_invalid");
    }
    await verifyUserToken(config, rotated.access_token, grant);
    const next: GrantRecord = { ...grant, generation: randomToken(18), access: rotated.access_token,
      refresh: rotated.refresh_token, accessExpires: Date.now() + rotated.expires_in * 1000,
      refreshExpires: Date.now() + rotated.refresh_token_expires_in * 1000 };
    const ttl = Math.min(30 * 24 * 60 * 60, Math.ceil((next.refreshExpires - Date.now()) / 1000));
    const serialized = await seal(next, config);
    const result = Number(await redis("EVAL", persistScript, "3", `${GRANT_PREFIX}${identity.hash}:lock`, `${CREDENTIAL_PREFIX}${identity.hash}`, `${GRANT_PREFIX}${identity.hash}`, lockToken, grant.generation, serialized, String(ttl)));
    if (result !== 1) throw new BrokerError(401, "GitHub App authorization must be renewed.", "github_grant_invalid");
    return next;
  });
}

/** Remove local authority first, then revoke only the retained App token. */
export async function revokeUserGrant(identity: UserGrant): Promise<void> {
  if (!appGrantEnabled()) {
    await redisDelete(`${CREDENTIAL_PREFIX}${identity.hash}`);
    await redisDelete(`${GRANT_PREFIX}${identity.hash}`);
    return;
  }
  let config: AppConfig | undefined;
  try { config = appConfig(); } catch { /* local revocation must survive configuration loss */ }
  let remoteToken: string | undefined;
  await withGrantLock(identity.hash, async (lock) => {
    const key = `${GRANT_PREFIX}${identity.hash}`;
    const encrypted = await redisGetRaw(key);
    if (encrypted && config) {
      try { remoteToken = (await unseal(encrypted, config, identity.hash)).access; }
      catch { /* corrupted or undecryptable grants are still revoked locally */ }
    }
    if (remoteToken && config) {
      const protectedPending = await seal({ version: 1, generation: randomToken(18), userId: 0, contributor: "", credentialHash: identity.hash,
        clientId: config.clientId, access: remoteToken, refresh: "", accessExpires: Date.now(), refreshExpires: Date.now() + 30 * 86400_000 }, config);
      await redisSet(`${GRANT_PREFIX}${identity.hash}:revocation`, protectedPending, 30 * 86400);
    }
    await redisDelete(`${CREDENTIAL_PREFIX}${identity.hash}`);
    const removed = Number(await redis("EVAL", removeScript, "3", `${GRANT_PREFIX}${identity.hash}:lock`, key, `${CREDENTIAL_PREFIX}${identity.hash}`, lock, ""));
    if (removed !== 1) throw new BrokerError(503, "Credential revocation could not confirm ownership; retry revoke.");
  });
  const pendingKey = `${GRANT_PREFIX}${identity.hash}:revocation`;
  const stored = await redisGetRaw(pendingKey);
  if (!stored || !config) return;
  let pending: GrantRecord;
  try { pending = await unseal(stored, config, identity.hash); }
  catch { return; }
  const response = await request(`https://api.github.com/applications/${encodeURIComponent(config.clientId)}/token`, {
    method: "DELETE", headers: { authorization: basicAuth(config), "content-type": "application/json" }, body: JSON.stringify({ access_token: pending.access }),
  });
  if (response.ok || response.status === 404) await redisDelete(pendingKey);
  else throw new BrokerError(503, "GitHub App token revocation is pending; retry revoke.");
}

async function commentPages(issueNumber: number, token: string): Promise<{ comments: Array<{ body?: string; user?: { id?: number; login?: string } }>; complete: boolean }> {
  const comments: Array<{ body?: string; user?: { id?: number; login?: string } }> = [];
  for (let page = 1; page <= 100; page++) {
    const response = await request(`https://api.github.com/repos/${REPO}/issues/${issueNumber}/comments?per_page=100&page=${page}`, { headers: { authorization: `Bearer ${token}` } });
    if (!response.ok) throw new BrokerError(response.status === 401 ? 401 : 502, response.status === 401 ? "GitHub App authorization must be renewed." : "Unable to verify volunteer issue comments.", response.status === 401 ? "github_grant_invalid" : undefined);
    const chunk = await response.json() as Array<{ body?: string; user?: { id?: number; login?: string } }>;
    comments.push(...chunk);
    if (chunk.length < 100) return { comments, complete: true };
  }
  return { comments, complete: false };
}

/** Ensure a live, authenticated claimant's one public comment exists for this issue. */
export async function ensureVolunteerComment(identity: UserGrant, issueNumber: number, assertOwned: () => Promise<void>): Promise<void> {
  const config = appConfig();
  const refreshed = await liveGrant(identity, config);
  await withGrantLock(identity.hash, async (grantLock) => {
    const encrypted = await redisGetRaw(`${GRANT_PREFIX}${identity.hash}`);
    if (!encrypted) throw new BrokerError(401, "GitHub App authorization must be renewed.", "github_grant_invalid");
    const grant = await unseal(encrypted, config, identity.hash);
    if (grant.contributor.toLowerCase() !== identity.contributor.toLowerCase()) throw new BrokerError(401, "GitHub App authorization must be renewed.", "github_grant_invalid");
    if (grant.generation !== refreshed.generation || grant.accessExpires <= Date.now()) {
      throw new BrokerError(401, "GitHub App authorization must be renewed.", "github_grant_invalid");
    }
    const commentLockKey = `${GRANT_PREFIX}${identity.hash}:comment:${issueNumber}:lock`;
    const commentLock = randomToken(18);
    if (!(await redisSet(commentLockKey, commentLock, LOCK_TTL, true))) throw new BrokerError(409, "Volunteer comment is being reconciled; retry.");
    const renewCommentLock = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('EXPIRE', KEYS[1], ARGV[2]) else return 0 end";
    const renewal = setInterval(() => {
      void redis("EVAL", renewCommentLock, "1", commentLockKey, commentLock, String(LOCK_TTL)).catch(() => undefined);
    }, 10_000);
    (renewal as unknown as { unref?: () => void }).unref?.();
    try {
      const dedupKey = `${GRANT_PREFIX}${identity.hash}:comment:${issueNumber}`;
      const pending = await redisGetRaw(dedupKey);
      const scanned = await commentPages(issueNumber, grant.access);
      if (!scanned.complete) throw new BrokerError(502, "Issue comment history is too large to verify safely.");
      if (scanned.comments.some((comment) => comment.user?.id === grant.userId && comment.user.login?.toLowerCase() === grant.contributor.toLowerCase() && comment.body === COMMENT)) {
        await redisSet(dedupKey, "done", Math.max(60, Math.ceil((grant.refreshExpires - Date.now()) / 1000)));
        return;
      }
      if (pending) return; // A prior POST may have succeeded despite losing its response; never blindly repeat it.
      await assertOwned();
      const stillCredential = await redisGet(`${CREDENTIAL_PREFIX}${identity.hash}`);
      const latest = await redisGetRaw(`${GRANT_PREFIX}${identity.hash}`);
      if (!stillCredential || !latest || (JSON.parse(latest) as StoredEnvelope).generation !== grant.generation) {
        throw new BrokerError(401, "GitHub App authorization must be renewed.", "github_grant_invalid");
      }
      await redisSet(dedupKey, "pending", Math.max(60, Math.ceil((grant.refreshExpires - Date.now()) / 1000)));
      let postStarted = false;
      const clearPendingIfOwned = async (): Promise<void> => {
        const clear = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[2]) else return 0 end";
        await redis("EVAL", clear, "2", commentLockKey, dedupKey, commentLock).catch(() => undefined);
      };
      try {
        await assertOwned();
        if (await redisGetRaw(commentLockKey) !== commentLock ||
            (JSON.parse((await redisGetRaw(`${GRANT_PREFIX}${identity.hash}`)) || "null") as StoredEnvelope | null)?.generation !== grant.generation ||
            !(await redisGet(`${CREDENTIAL_PREFIX}${identity.hash}`))) {
          throw new BrokerError(401, "GitHub App authorization must be renewed.", "github_grant_invalid");
        }
        await assertOwned();
        if (await redisGetRaw(commentLockKey) !== commentLock) throw new BrokerError(409, "Volunteer comment ownership fence was lost.");
        postStarted = true;
        const posted = await request(`https://api.github.com/repos/${REPO}/issues/${issueNumber}/comments`, {
          method: "POST", headers: { authorization: `Bearer ${grant.access}`, "content-type": "application/json" }, body: JSON.stringify({ body: COMMENT }),
        });
        if (posted.status === 401) {
          await clearPendingIfOwned();
          throw new BrokerError(401, "GitHub App authorization must be renewed.", "github_grant_invalid");
        }
        if (!posted.ok) {
          await clearPendingIfOwned();
          throw new BrokerError(502, "Unable to post volunteer issue comment.");
        }
        const result = await posted.json() as { user?: { id?: number; login?: string } };
        if (result.user?.id !== grant.userId || result.user.login?.toLowerCase() !== grant.contributor.toLowerCase()) throw new BrokerError(502, "GitHub did not confirm the volunteer comment author.");
        await redisSet(dedupKey, "done", Math.max(60, Math.ceil((grant.refreshExpires - Date.now()) / 1000)));
      } catch (error) {
        if (!postStarted) await clearPendingIfOwned();
        throw error;
      }
    } finally {
      clearInterval(renewal);
      const release = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) else return 0 end";
      await redis("EVAL", release, "1", commentLockKey, commentLock).catch(() => undefined);
    }
  });
}

/** Revoke a verified-or-partially-verified App access token after failed onboarding. */
export async function discardUserAccessToken(accessToken: string): Promise<void> {
  const config = appConfig();
  const response = await request(`https://api.github.com/applications/${encodeURIComponent(config.clientId)}/token`, {
    method: "DELETE", headers: { authorization: basicAuth(config), "content-type": "application/json" },
    body: JSON.stringify({ access_token: accessToken }),
  });
  if (!response.ok && response.status !== 404) throw new BrokerError(503, "GitHub App token cleanup failed.");
}

/** Derive the broker credential's grant key. */
export async function grantCredentialHash(credential: string): Promise<string> { return sha256(credential); }
