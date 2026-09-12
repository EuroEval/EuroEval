import { BrokerError, env, sha256 } from "./protocol.ts";
import { redis, redisGet } from "./redis.ts";
import type { WorkerIdentity } from "./protocol.ts";

export function authCredential(req: Request): string {
  const header = req.headers.get("authorization") || "";
  if (!header.startsWith("Bearer ") || header.length < 10) throw new BrokerError(401, "Use the worker credential as Authorization: Bearer <credential>.");
  return header.slice(7).trim();
}
export async function authenticate(req: Request): Promise<WorkerIdentity> {
  const credential = authCredential(req); const hash = await sha256(credential);
  const worker = await redisGet<{ contributor: string }>(`euroeval:worker:credential:${hash}`);
  if (!worker?.contributor || worker.contributor.toLowerCase() === "saattrupdan") throw new BrokerError(401, "Worker credential is invalid or revoked.");
  return { hash, contributor: worker.contributor };
}
export async function enforceRateLimit(key: string, limit: number, windowSeconds: number): Promise<void> {
  const bucket = `${key}:${Math.floor(Date.now() / (windowSeconds * 1000))}`;
  const count = await redis("INCR", bucket);
  if (count === 1 || count === "1") await redis("EXPIRE", bucket, String(windowSeconds));
  if (Number(count) > limit) throw new BrokerError(429, "Rate limit exceeded; retry later.");
}
export function coordinatorSecret(req: Request): void {
  const expected = env("WORKER_COORDINATOR_SECRET");
  const supplied = req.headers.get("x-coordinator-secret") || authCredential(req);
  if (supplied !== expected) throw new BrokerError(401, "Coordinator authentication failed.");
}

export function promotionSecret(req: Request): void {
  const expected = env("VOLUNTEER_PROMOTION_SECRET");
  const supplied = req.headers.get("x-promotion-secret") || authCredential(req);
  if (supplied !== expected) throw new BrokerError(401, "Promotion authentication failed.");
}
