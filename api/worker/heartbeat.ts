import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, authenticate, getLeaseById, json, leaseTtl, method,
  readJson, saveLease,
} from "./_lib";

export const config = { runtime: "edge" };

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const identity = await authenticate(req);
    const body = await readJson(req, 8 * 1024);
    if (typeof body.lease_id !== "string") throw new BrokerError(400, "lease_id is required.");
    const lease = await getLeaseById(body.lease_id);
    if (!lease || lease.worker !== identity.hash.slice(0, 24) || lease.expires_at <= Date.now()) throw new BrokerError(409, "Lease is absent, expired, or belongs to another worker.");
    lease.expires_at = Date.now() + leaseTtl() * 1000;
    if (!(await saveLease(lease))) throw new BrokerError(409, "Lease was replaced or expired before it could be renewed.");
    return json(200, { protocol_version: PROTOCOL_VERSION, lease_id: lease.lease_id, expires_at: lease.expires_at });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { error: error instanceof Error ? error.message : "Unable to renew lease." });
  }
}
