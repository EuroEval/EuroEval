import { BrokerError, optionalEnv } from "./protocol.js";
import { redis } from "./redis.js";

export const DEFAULT_PUBLICATION_BUCKET = "EuroEval/results";

export function canonicalPublicationBucket(): string {
  return optionalEnv("HF_RESULTS_BUCKET", DEFAULT_PUBLICATION_BUCKET);
}

export function publicationKey(bucket = canonicalPublicationBucket()): string {
  return `euroeval:worker:canonical-publication:${bucket}`;
}

export async function publicationOwned(bucket: string, owner: string): Promise<boolean> {
  const result = await redis("EVAL", "return redis.call('GET',KEYS[1]) == ARGV[1] and 1 or 0",
    "1", publicationKey(bucket), owner);
  return result === 1 || result === "1";
}

export async function assertPublicationOwner(bucket: unknown, owner: unknown): Promise<{ key: string; owner: string }> {
  if (typeof bucket !== "string" || bucket !== canonicalPublicationBucket() ||
      typeof owner !== "string" || !/^[A-Za-z0-9_-]{24,256}$/.test(owner)) {
    throw new BrokerError(400, "publication_bucket and publication_owner are required and must match the canonical bucket.");
  }
  return { key: publicationKey(bucket), owner };
}

/** Redis compare/set/delete operations are atomic and deliberately have no TTL. */
export const publicationScripts = {
  acquire: `local value=redis.call('GET',KEYS[1]); if not value then redis.call('SET',KEYS[1],ARGV[1]); return 'owned' end; if value == ARGV[1] then return 'owned' end; return 'busy'`,
  assert: `if redis.call('GET',KEYS[1]) == ARGV[1] then return 'owned' end; return 'busy'`,
  release: `if redis.call('GET',KEYS[1]) == ARGV[1] then redis.call('DEL',KEYS[1]); return 'released' end; if not redis.call('GET',KEYS[1]) then return 'released' end; return 'busy'`,
};
