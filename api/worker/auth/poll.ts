declare const process: { env: Record<string, string | undefined> };

import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, env, enforceRateLimit, fetchWithRetry, json, method, randomToken, readJson,
  redisDelete, redisGet, redisSet, sha256, requireProtocol,
} from "../_lib";

export const config = { runtime: "edge" };

type Device = { device_code: string; client_id: string; interval: number; next_poll_at?: number };

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const body = await readJson(req, 8 * 1024);
    requireProtocol(body);
    if (typeof body.session_id !== "string" || !/^[A-Za-z0-9_-]{12,}$/.test(body.session_id)) throw new BrokerError(400, "session_id is required.");
    const key = `euroeval:worker:device:${body.session_id}`;
    await enforceRateLimit(`euroeval:worker:limit:auth-poll:${body.session_id}`, 120, 900);
    const device = await redisGet<Device>(key);
    if (!device) throw new BrokerError(400, "Device authorisation has expired or was already used.");
    if (device.next_poll_at && device.next_poll_at > Date.now()) throw new BrokerError(429, `Poll again in ${Math.ceil((device.next_poll_at - Date.now()) / 1000)} seconds.`);
    const response = await fetchWithRetry("https://github.com/login/oauth/access_token", {
      method: "POST",
      headers: { accept: "application/json", "content-type": "application/json" },
      body: JSON.stringify({ client_id: device.client_id || env("GITHUB_OAUTH_CLIENT_ID"), device_code: device.device_code, grant_type: "urn:ietf:params:oauth:grant-type:device_code" }),
    });
    const data = await response.json() as { access_token?: string; error?: string; error_description?: string };
    if (data.error === "authorization_pending" || data.error === "slow_down") {
      const interval = data.error === "slow_down" ? Math.max(5, device.interval + 5) : device.interval;
      await redisSet(key, JSON.stringify({ ...device, interval, next_poll_at: Date.now() + interval * 1000 }), 900);
      return json(202, { protocol_version: PROTOCOL_VERSION, status: "pending", retry_after: interval });
    }
    if (!response.ok || !data.access_token) {
      await redisDelete(key);
      throw new BrokerError(401, data.error_description || "GitHub device authorisation was denied or expired.");
    }
    const userResponse = await fetchWithRetry("https://api.github.com/user", { headers: { accept: "application/vnd.github+json", authorization: `Bearer ${data.access_token}`, "x-github-api-version": "2022-11-28" } });
    // Deliberately keep the token in this scope only. It is never returned or persisted.
    if (!userResponse.ok) { await redisDelete(key); throw new BrokerError(401, "GitHub could not identify this worker."); }
    const user = await userResponse.json() as { login?: string };
    if (!user.login || user.login.toLowerCase() === "saattrupdan") { await redisDelete(key); throw new BrokerError(403, "The repository owner cannot register as a volunteer worker."); }
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
    await redisSet(`euroeval:worker:credential:${hash}`, JSON.stringify({ contributor: user.login, issued_at: new Date().toISOString() }), ttl);
    await redisDelete(key);
    return json(200, { protocol_version: PROTOCOL_VERSION, status: "authorised", credential, github_login: user.login, expires_in: ttl });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { error: error instanceof Error ? error.message : "Unable to poll authorisation." });
  }
}
