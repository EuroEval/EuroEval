import {
  ConfigurationError, BrokerError, PROTOCOL_VERSION, env, json, method, randomToken, redisSet,
  readJson, requireProtocol,
} from "../_lib";

export const config = { runtime: "edge" };

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const body = await readJson(req, 4 * 1024);
    requireProtocol(body);
    const clientId = env("GITHUB_OAUTH_CLIENT_ID");
    const response = await fetch("https://github.com/login/device/code", {
      method: "POST",
      headers: { accept: "application/json", "content-type": "application/json" },
      body: JSON.stringify({ client_id: clientId, scope: "read:user" }),
    });
    if (!response.ok) throw new BrokerError(503, `GitHub device authorisation failed with HTTP ${response.status}.`);
    const data = await response.json() as { device_code?: string; user_code?: string; verification_uri?: string; verification_uri_complete?: string; interval?: number; expires_in?: number };
    if (!data.device_code || !data.user_code || !data.verification_uri) throw new BrokerError(503, "GitHub returned an incomplete device authorisation response.");
    const deviceId = randomToken(18);
    const ttl = Math.max(60, Math.min(900, data.expires_in || 600));
    await redisSet(`euroeval:worker:device:${deviceId}`, JSON.stringify({ device_code: data.device_code, client_id: clientId, interval: data.interval || 5 }), ttl);
    return json(200, { protocol_version: PROTOCOL_VERSION, session_id: deviceId, user_code: data.user_code, verification_uri: data.verification_uri, verification_uri_complete: data.verification_uri_complete, interval: data.interval || 5, expires_in: ttl });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { error: error instanceof Error ? error.message : "Unable to start authorisation." });
  }
}
