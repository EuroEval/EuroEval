declare const process: { env: Record<string, string | undefined> };

import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, env, enforceRateLimit, fetchWithRetry, json, method, randomToken, readJson,
  redis, redisDelete, redisGet, redisSet, sha256, requireProtocol, appGrantEnabled, retainUserGrant, revokeUserGrant, discardUserAccessToken,
} from "../_lib.js";

export const config = { runtime: "edge" };

type Device = { device_code: string; client_id: string; interval: number; app_mode?: boolean; next_poll_at?: number };

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  let completionLock: { key: string; token: string } | undefined;
  try {
    const body = await readJson(req, 8 * 1024);
    requireProtocol(body);
    if (typeof body.session_id !== "string" || !/^[A-Za-z0-9_-]{12,}$/.test(body.session_id)) throw new BrokerError(400, "session_id is required.");
    const key = `euroeval:worker:device:${body.session_id}`;
    await enforceRateLimit(`euroeval:worker:limit:auth-poll:${body.session_id}`, 120, 900);
    const device = await redisGet<Device>(key);
    if (!device) throw new BrokerError(400, "Device authorisation has expired or was already used.");
    if (device.next_poll_at && device.next_poll_at > Date.now()) throw new BrokerError(429, `Poll again in ${Math.ceil((device.next_poll_at - Date.now()) / 1000)} seconds.`);
    const lockKey = `${key}:completion-lock`;
    const lockToken = randomToken(18);
    if (!(await redisSet(lockKey, lockToken, 30, true))) throw new BrokerError(429, "Authorization is being completed; retry shortly.");
    completionLock = { key: lockKey, token: lockToken };
    const response = await fetchWithRetry("https://github.com/login/oauth/access_token", {
      method: "POST",
      headers: { accept: "application/json", "content-type": "application/json" },
      body: JSON.stringify({ client_id: device.client_id || env("GITHUB_OAUTH_CLIENT_ID"), device_code: device.device_code, grant_type: "urn:ietf:params:oauth:grant-type:device_code" }),
    });
    const data = await response.json() as { access_token?: string; refresh_token?: string; expires_in?: number; refresh_token_expires_in?: number; error?: string; error_description?: string };
    if (data.error === "authorization_pending" || data.error === "slow_down") {
      const interval = data.error === "slow_down" ? Math.max(5, device.interval + 5) : device.interval;
      await redisSet(key, JSON.stringify({ ...device, interval, next_poll_at: Date.now() + interval * 1000 }), 900);
      return json(202, { protocol_version: PROTOCOL_VERSION, status: "pending", retry_after: interval }, { "retry-after": String(interval) });
    }
    if (!response.ok || !data.access_token) {
      await redisDelete(key);
      throw new BrokerError(401, "GitHub device authorisation was denied or expired.");
    }
    if (device.app_mode && !appGrantEnabled()) throw new ConfigurationError("GitHub App grant configuration changed during authorization.");
    const userResponse = await fetchWithRetry("https://api.github.com/user", { headers: { accept: "application/vnd.github+json", authorization: `Bearer ${data.access_token}`, "x-github-api-version": "2022-11-28" } });
    // Deliberately keep the token in this scope only. It is never returned or persisted.
    if (!userResponse.ok) { await redisDelete(key); throw new BrokerError(401, "GitHub could not identify this worker."); }
    const user = await userResponse.json() as { login?: string };
    if (!user.login) { await redisDelete(key); throw new BrokerError(401, "GitHub could not identify this worker."); }
    if (device.app_mode) {
      if (!data.refresh_token || !data.expires_in || !data.refresh_token_expires_in) {
        throw new BrokerError(401, "GitHub App authorization did not provide renewable tokens.");
      }
      const credential = randomToken(32);
      const hash = await sha256(credential);
      const ttl = 30 * 24 * 60 * 60;
      const credentialKey = `euroeval:worker:credential:${hash}`;
      if (!(await redisSet(credentialKey, JSON.stringify({ contributor: user.login, issued_at: new Date().toISOString(), grant_mode: "app" }), ttl, true))) {
        throw new BrokerError(503, "Unable to issue worker credential safely.");
      }
      try {
        await retainUserGrant(data.access_token, data.refresh_token, Date.now() + data.expires_in * 1000,
          Date.now() + data.refresh_token_expires_in * 1000, credential, user.login);
        const complete = "if redis.call('EXISTS', KEYS[1]) == 0 or redis.call('EXISTS', KEYS[2]) == 0 then return 0 end; return redis.call('DEL', KEYS[1])";
        const consumed = Number(await redis("EVAL", complete, "2", key, credentialKey));
        if (consumed !== 1) throw new BrokerError(400, "Device authorisation has expired or was already used.");
      } catch (error) {
        await revokeUserGrant({ hash, contributor: user.login }).catch(() => undefined);
        await discardUserAccessToken(data.access_token).catch(() => undefined);
        throw error;
      }
      return json(200, { protocol_version: PROTOCOL_VERSION, status: "authorised", credential, github_login: user.login, expires_in: ttl });
    }
    const clientSecret = process.env.GITHUB_OAUTH_CLIENT_SECRET;
    if (!clientSecret) throw new ConfigurationError("GITHUB_OAUTH_CLIENT_SECRET is required to revoke the device grant before issuing a credential.");
    const revoked = await fetchWithRetry(`https://api.github.com/applications/${device.client_id}/grant`, {
      method: "DELETE", headers: { accept: "application/vnd.github+json", "content-type": "application/json", authorization: `Basic ${btoa(`${device.client_id}:${clientSecret}`)}` },
      body: JSON.stringify({ access_token: data.access_token }),
    });
    if (!revoked.ok) throw new BrokerError(503, "GitHub grant revocation failed; no broker credential was issued.");
    const credential = randomToken(32);
    const hash = await sha256(credential);
    const ttl = 30 * 24 * 60 * 60;
    const credentialKey = `euroeval:worker:credential:${hash}`;
    if (!(await redisSet(credentialKey, JSON.stringify({ contributor: user.login, issued_at: new Date().toISOString() }), ttl, true))) {
      throw new BrokerError(503, "Unable to issue worker credential safely.");
    }
    const complete = "if redis.call('EXISTS', KEYS[1]) == 0 or redis.call('EXISTS', KEYS[2]) == 0 then return 0 end; return redis.call('DEL', KEYS[1])";
    const consumed = Number(await redis("EVAL", complete, "2", key, credentialKey));
    if (consumed !== 1) {
      await redisDelete(credentialKey);
      throw new BrokerError(400, "Device authorisation has expired or was already used.");
    }
    return json(200, { protocol_version: PROTOCOL_VERSION, status: "authorised", credential, github_login: user.login, expires_in: ttl });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { error: error instanceof BrokerError || error instanceof ConfigurationError ? error.message : "Unable to poll authorisation." });
  } finally {
    if (completionLock) {
      const release = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) else return 0 end";
      await redis("EVAL", release, "1", completionLock.key, completionLock.token).catch(() => undefined);
    }
  }
}
