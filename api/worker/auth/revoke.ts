import { BrokerError, ConfigurationError, PROTOCOL_VERSION, authCredential, authenticate, json, method, redisGet, redisGetRaw, revokeUserGrant, sha256 } from "../_lib.js";

export const config = { runtime: "edge" };

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const credential = authCredential(req);
    const hash = await sha256(credential);
    const worker = await redisGet<{ contributor: string }>(`euroeval:worker:credential:${hash}`);
    if (worker?.contributor) {
      await revokeUserGrant(await authenticate(req, true));
    } else {
      const pending = await redisGetRaw(`euroeval:worker:user-grant:${hash}:revocation`);
      if (!pending) throw new BrokerError(401, "Worker credential is invalid or revoked.");
      await revokeUserGrant({ hash, contributor: "" });
    }
    return json(200, { protocol_version: PROTOCOL_VERSION, status: "revoked" });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { error: error instanceof BrokerError || error instanceof ConfigurationError ? error.message : "Unable to revoke credential." });
  }
}
