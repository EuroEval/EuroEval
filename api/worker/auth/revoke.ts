import { BrokerError, ConfigurationError, PROTOCOL_VERSION, authenticate, json, method, redisDelete, revokeUserGrant } from "../_lib.js";

export const config = { runtime: "edge" };

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const identity = await authenticate(req, true);
    await redisDelete(`euroeval:worker:credential:${identity.hash}`);
    await revokeUserGrant(identity);
    return json(200, { protocol_version: PROTOCOL_VERSION, status: "revoked" });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { error: error instanceof Error ? error.message : "Unable to revoke credential." });
  }
}
