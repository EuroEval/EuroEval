import { BrokerError, ConfigurationError, env, json, method, readJson } from "./_lib.js";
import { canonicalPublicationBucket, publicationKey, publicationScripts } from "./_lib/publication.js";
import { redis } from "./_lib/redis.js";
import { enforceRateLimit } from "./_lib/auth.js";

export const config = { runtime: "edge" };

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const supplied = req.headers.get("authorization") || "";
    const secret = env("CANONICAL_PUBLICATION_SECRET");
    if (!supplied.startsWith("Bearer ") || supplied.slice(7).trim() !== secret) {
      throw new BrokerError(401, "Canonical publication authentication failed.");
    }
    const body = await readJson(req, 16 * 1024);
    if (!["acquire", "assert", "release", "status"].includes(String(body.action))) throw new BrokerError(400, "Invalid publication action.");
    const bucket = body.bucket;
    const owner = body.owner;
    if (typeof bucket !== "string" || bucket !== canonicalPublicationBucket()) throw new BrokerError(400, "Only the configured canonical results bucket is permitted.");
    if (body.action === "status") {
      const result = await redis("GET", publicationKey(bucket));
      return json(200, { bucket, ...(typeof owner === "string" ? { owner } : {}), status: result ? "owned" : "unlocked" });
    }
    if (typeof owner !== "string" || !/^[A-Za-z0-9_-]{24,256}$/.test(owner)) throw new BrokerError(400, "owner must be an opaque random token.");
    const action = body.action as "acquire" | "assert" | "release";
    if (action === "acquire") {
      const key = publicationKey(bucket);
      const currentOwner = await redis("GET", key);
      if (currentOwner !== owner) {
        await enforceRateLimit("euroeval:worker:publication-rate", 60, 60);
      }
    }
    const status = await redis("EVAL", publicationScripts[action], "1", publicationKey(bucket), owner);
    if (status === "busy") throw new BrokerError(409, "Canonical publication lock is held by another owner.");
    return json(200, { bucket, owner, status });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { error: error instanceof Error ? error.message : "Unable to access publication lock." });
  }
}
