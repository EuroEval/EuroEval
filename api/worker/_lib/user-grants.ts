import { BrokerError, ConfigurationError, env, randomToken, sha256 } from "./protocol.js";
import { redis, redisDelete, redisGet, redisSet } from "./redis.js";

/** Server-side GitHub App user grants used only for public volunteer comments. */
export type UserGrant = { contributor: string; hash: string };
type AppConfig = { clientId: string; clientSecret: string; appId: string; privateKey: string; installationId: string; repositoryId: string; encryptionKey: CryptoKey };
type GrantRecord = { version: 1; generation: string; userId: number; contributor: string; credentialHash: string; clientId: string; access: string; refresh: string; accessExpires: number; refreshExpires: number };
const CONFIG_NAMES = ["VOLUNTEER_GITHUB_APP_CLIENT_ID", "VOLUNTEER_GITHUB_APP_CLIENT_SECRET", "VOLUNTEER_GITHUB_APP_ID", "VOLUNTEER_GITHUB_APP_PRIVATE_KEY", "VOLUNTEER_GITHUB_APP_INSTALLATION_ID", "VOLUNTEER_GITHUB_APP_REPOSITORY_ID", "VOLUNTEER_GITHUB_APP_TOKEN_ENCRYPTION_KEY"] as const;

/** Return true when app authorization is opted in; reject incomplete configuration. */
export function appGrantEnabled(): boolean {
  const present = CONFIG_NAMES.filter((name) => !!process.env[name]);
  if (!present.length) return false;
  if (present.length !== CONFIG_NAMES.length) throw new ConfigurationError("GitHub App grant configuration is incomplete.");
  return true;
}

declare const process: { env: Record<string, string | undefined> };

function appConfig(): AppConfig {
  if (!appGrantEnabled()) throw new ConfigurationError("GitHub App grants are not configured.");
  const encodedKey = env("VOLUNTEER_GITHUB_APP_TOKEN_ENCRYPTION_KEY");
  const raw = Uint8Array.from(atob(encodedKey), (char) => char.charCodeAt(0));
  if (raw.length !== 32) throw new ConfigurationError("GitHub App token encryption key must be 32 bytes in base64.");
  const pem = env("VOLUNTEER_GITHUB_APP_PRIVATE_KEY").replace(/\\n/g, "\n");
  const der = Uint8Array.from(atob(pem.replace(/-----[^-]+-----/g, "").replace(/\s/g, "")), (char) => char.charCodeAt(0));
  return {
    clientId: env("VOLUNTEER_GITHUB_APP_CLIENT_ID"), clientSecret: env("VOLUNTEER_GITHUB_APP_CLIENT_SECRET"),
    appId: env("VOLUNTEER_GITHUB_APP_ID"), privateKey: pem,
    installationId: env("VOLUNTEER_GITHUB_APP_INSTALLATION_ID"), repositoryId: env("VOLUNTEER_GITHUB_APP_REPOSITORY_ID"),
    encryptionKey: crypto.subtle.importKey("raw", raw, "AES-GCM", false, ["encrypt", "decrypt"]) as unknown as CryptoKey,
  };
}

async function request(url: string, init: RequestInit = {}): Promise<Response> {
  return fetch(url, { ...init, headers: { accept: "application/vnd.github+json", "x-github-api-version": "2022-11-28", ...(init.headers || {}) } });
}
function base64url(bytes: Uint8Array): string { return btoa(String.fromCharCode(...bytes)).replace(/=/g, "").replace(/\\+/g, "-").replace(/\\//g, "_"); }
async function appJwt(config: AppConfig): Promise<string> {
  const pem = config.privateKey.replace(/-----[^-]+-----/g, "").replace(/\s/g, "");
  const der = Uint8Array.from(atob(pem), (char) => char.charCodeAt(0));
  const key = await crypto.subtle.importKey("pkcs8", der, { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" }, false, ["sign"]);
  const now = Math.floor(Date.now() / 1000);
  const head = base64url(new TextEncoder().encode(JSON.stringify({ alg: "RS256", typ: "JWT" })));
  const payload = base64url(new TextEncoder().encode(JSON.stringify({ iat: now - 30, exp: now + 540, iss: config.appId })));
  const data = `${head}.${payload}`;
  return `${data}.${base64url(new Uint8Array(await crypto.subtle.sign("RSASSA-PKCS1-v1_5", key, new TextEncoder().encode(data))))}`;
}

/** Validate installation/repository permission before enabling a user-token grant. */
export async function verifyAppInstallation(): Promise<void> {
  const config = appConfig();
  const response = await request(`https://api.github.com/app/installations/${encodeURIComponent(config.installationId)}`, { headers: { authorization: `Bearer ${await appJwt(config)}` } });
  if (!response.ok) throw new BrokerError(503, "GitHub App installation could not be verified.");
  const installation = await response.json() as { id?: number; repository_selection?: string; permissions?: { issues?: string; metadata?: string } };
  if (String(installation.id) !== config.installationId || installation.permissions?.issues !== "write" || installation.permissions?.metadata !== "read") throw new BrokerError(503, "GitHub App installation permissions are insufficient.");
  const tokenResponse = await request(`https://api.github.com/app/installations/${encodeURIComponent(config.installationId)}/access_tokens`, { method: "POST", headers: { authorization: `Bearer ${await appJwt(config)}`, "content-type": "application/json" }, body: JSON.stringify({ repository_ids: [Number(config.repositoryId)] }) });
  if (!tokenResponse.ok) throw new BrokerError(503, "GitHub App repository access could not be verified.");
  const tokenData = await tokenResponse.json() as { token?: string; permissions?: { issues?: string; metadata?: string }; repositories?: Array<{ id: number }> };
  if (!tokenData.token || tokenData.permissions?.issues !== "write" || tokenData.permissions?.metadata !== "read" || !tokenData.repositories?.some((repo) => String(repo.id) === config.repositoryId)) throw new BrokerError(503, "GitHub App repository permissions are insufficient.");
}

/** Encrypt all token material with fresh AES-GCM nonces and credential-bound AAD. */
async function seal(record: GrantRecord, config: AppConfig): Promise<string> {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const aad = new TextEncoder().encode(`${record.credentialHash}:${record.generation}`);
  const key = await config.encryptionKey;
  const ciphertext = await crypto.subtle.encrypt({ name: "AES-GCM", iv, additionalData: aad }, key, new TextEncoder().encode(JSON.stringify(record)));
  return JSON.stringify({ version: 1, generation: record.generation, iv: base64url(iv), ciphertext: base64url(new Uint8Array(ciphertext)) });
}
async function unseal(value: string, config: AppConfig, hash: string): Promise<GrantRecord> {
  try {
    const envelope = JSON.parse(value) as { version: number; generation: string; iv: string; ciphertext: string };
    if (envelope.version !== 1) throw new Error();
    const bytes = (input: string) => Uint8Array.from(atob(input.replace(/-/g, "+").replace(/_/g, "/") + "===".slice((input.length + 3) % 4)), (char) => char.charCodeAt(0));
    const key = await config.encryptionKey;
    const plain = await crypto.subtle.decrypt({ name: "AES-GCM", iv: bytes(envelope.iv), additionalData: new TextEncoder().encode(`${hash}:${envelope.generation}`) }, key, bytes(envelope.ciphertext));
    const record = JSON.parse(new TextDecoder().decode(plain)) as GrantRecord;
    if (record.credentialHash !== hash || record.generation !== envelope.generation || record.clientId !== config.clientId) throw new Error();
    return record;
  } catch { throw new BrokerError(401, "GitHub App authorization must be renewed.", "github_grant_invalid"); }
}

/** Complete and verify an App device-flow authorization; never expose GitHub tokens. */
export async function retainUserGrant(accessToken: string, refreshToken: string, accessExpires: number, refreshExpires: number, credential: string, expectedLogin: string): Promise<void> {
  const config = appConfig();
  await verifyAppInstallation();
  const identity = await request("https://api.github.com/user", { headers: { authorization: `Bearer ${accessToken}` } });
  if (!identity.ok) throw new BrokerError(401, "GitHub could not verify the volunteer identity.");
  const user = await identity.json() as { id?: number; login?: string };
  if (!user.id || !user.login || user.login.toLowerCase() !== expectedLogin.toLowerCase()) throw new BrokerError(401, "GitHub volunteer identity did not match.");
  const appCheck = await request(`https://api.github.com/applications/${encodeURIComponent(config.clientId)}/token`, { method: "POST", headers: { authorization: `Basic ${btoa(`${config.clientId}:${config.clientSecret}`)}`, "content-type": "application/json" }, body: JSON.stringify({ access_token: accessToken }) });
  if (!appCheck.ok) throw new BrokerError(401, "GitHub App authorization could not be verified.");
  const checked = await appCheck.json() as { token?: string; app?: { client_id?: string; id?: number }; user?: { id?: number; login?: string } };
  if (checked.token !== accessToken || checked.app?.client_id !== config.clientId || checked.user?.id !== user.id || checked.user?.login?.toLowerCase() !== user.login.toLowerCase()) throw new BrokerError(401, "GitHub App authorization identity did not match.");
  const hash = await sha256(credential);
  const record: GrantRecord = { version: 1, generation: randomToken(18), userId: user.id, contributor: user.login, credentialHash: hash, clientId: config.clientId, access: accessToken, refresh: refreshToken, accessExpires, refreshExpires };
  const encrypted = await seal(record, config);
  const result = await redis("SET", `euroeval:worker:user-grant:${hash}`, encrypted, "EX", String(Math.max(60, Math.ceil((refreshExpires - Date.now()) / 1000))));
  if (result !== "OK") throw new BrokerError(503, "GitHub App authorization could not be retained securely.");
}

/** Remove a retained grant locally before attempting remote revocation. */
export async function revokeUserGrant(identity: UserGrant): Promise<void> {
  const key = `euroeval:worker:user-grant:${identity.hash}`;
  const encrypted = await redisGet<string>(key);
  await redisDelete(key);
  if (!encrypted) return;
  try {
    const config = appConfig();
    const grant = await unseal(encrypted, config, identity.hash);
    const response = await request(`https://api.github.com/applications/${encodeURIComponent(config.clientId)}/token`, {
      method: "DELETE", headers: { authorization: `Basic ${btoa(`${config.clientId}:${config.clientSecret}`)}`, "content-type": "application/json" },
      body: JSON.stringify({ access_token: grant.access }),
    });
    if (!response.ok && response.status !== 404) throw new BrokerError(503, "GitHub App token revocation failed after local revocation.");
  } catch (error) {
    if (error instanceof ConfigurationError || error instanceof BrokerError && error.status === 401) return;
    throw error;
  }
}

/** Add the fixed public volunteer comment using only this contributor's retained grant. */
export async function ensureVolunteerComment(identity: UserGrant, issueNumber: number, assertOwned: () => Promise<void>): Promise<void> {
  const config = appConfig();
  const stored = await redisGet<string>(`euroeval:worker:user-grant:${identity.hash}`);
  if (!stored) throw new BrokerError(401, "GitHub App authorization must be renewed.", "github_grant_invalid");
  const grant = await unseal(stored, config, identity.hash);
  if (grant.contributor.toLowerCase() !== identity.contributor.toLowerCase() || grant.accessExpires <= Date.now()) throw new BrokerError(401, "GitHub App authorization must be renewed.", "github_grant_invalid");
  const headers = { authorization: `Bearer ${grant.access}`, "x-github-api-version": "2022-11-28" };
  const comments: Array<{ body?: string; user?: { id?: number; login?: string } }> = [];
  for (let page = 1; page <= 10; page++) {
    const response = await request(`https://api.github.com/repos/EuroEval/EuroEval/issues/${issueNumber}/comments?per_page=100&page=${page}`, { headers });
    if (!response.ok) throw new BrokerError(response.status === 401 ? 401 : 502, response.status === 401 ? "GitHub App authorization must be renewed." : "Unable to verify volunteer issue comments.", response.status === 401 ? "github_grant_invalid" : undefined);
    const chunk = await response.json() as Array<{ body?: string; user?: { id?: number; login?: string } }>;
    comments.push(...chunk);
    if (chunk.length < 100) break;
  }
  if (comments.some((comment) => comment.user?.id === grant.userId && comment.body === "I'd like to evaluate this.")) return;
  await assertOwned();
  const posted = await request(`https://api.github.com/repos/EuroEval/EuroEval/issues/${issueNumber}/comments`, { method: "POST", headers: { ...headers, "content-type": "application/json" }, body: JSON.stringify({ body: "I'd like to evaluate this." }) });
  if (!posted.ok) throw new BrokerError(posted.status === 401 ? 401 : 502, posted.status === 401 ? "GitHub App authorization must be renewed." : "Unable to post volunteer issue comment.", posted.status === 401 ? "github_grant_invalid" : undefined);
  const comment = await posted.json() as { user?: { id?: number } };
  if (comment.user?.id !== grant.userId) throw new BrokerError(502, "GitHub did not confirm the volunteer comment author.");
}

/** Associate a credential with its verified GitHub login for grant records. */
export async function grantCredentialHash(credential: string): Promise<string> { return sha256(credential); }
