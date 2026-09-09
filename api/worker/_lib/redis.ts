import { BrokerError, env, optionalEnv, randomToken, sha256 } from "./protocol.ts";
import type { Lease } from "./protocol.ts";

async function redisRequest(command: unknown[]): Promise<unknown> {
  let lastError: unknown;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const response = await fetch(env("UPSTASH_REDIS_REST_URL"), {
        method: "POST",
        headers: { authorization: `Bearer ${env("UPSTASH_REDIS_REST_TOKEN")}`, "content-type": "application/json" },
        body: JSON.stringify(command),
      });
      if (!response.ok) throw new BrokerError(503, `Upstash returned HTTP ${response.status}.`);
      const payload = (await response.json()) as { result?: unknown; error?: string };
      if (payload.error) throw new BrokerError(503, `Upstash error: ${payload.error}`);
      return payload.result;
    } catch (error) {
      lastError = error;
      if (attempt < 2) await new Promise((resolve) => setTimeout(resolve, 100 * (attempt + 1)));
    }
  }
  throw lastError instanceof Error ? lastError : new BrokerError(503, "Upstash request failed.");
}
export async function redis(command: string, ...args: string[]): Promise<unknown> {
  return redisRequest([command, ...args]);
}
export async function redisSet(key: string, value: string, ttl: number, nx = false): Promise<boolean> {
  const result = await redis("SET", key, value, "EX", String(ttl), ...(nx ? ["NX"] : []));
  return result === "OK";
}
export async function redisGet<T>(key: string): Promise<T | null> {
  const value = await redis("GET", key);
  if (typeof value !== "string") return null;
  try { return JSON.parse(value) as T; } catch { return null; }
}
export async function redisDelete(key: string): Promise<void> { await redis("DEL", key); }

const LEASE_TTL_MAX = 6 * 60 * 60;
export function leaseTtl(): number {
  const configured = Number(optionalEnv("VOLUNTEER_LEASE_SECONDS", "1800"));
  return Number.isFinite(configured) ? Math.max(60, Math.min(LEASE_TTL_MAX, Math.floor(configured))) : 1800;
}

export function issueLeaseKey(issue: number, language: string): string { return `euroeval:worker:lease:${issue}:${language}`; }
function resultReservationsKey(leaseId: string): string { return `euroeval:worker:reservations:${leaseId}`; }

export async function getLeaseForIssue(issue: number, language: string): Promise<Lease | null> {
  return redisGet<Lease>(issueLeaseKey(issue, language));
}

/**
 * Reclaim an expired lease and its result-identity reservations atomically.
 * Only identity keys whose stored owner and digest match the lease's persisted
 * reservation list are deleted; staged HF objects are intentionally untouched.
 */
export async function reclaimExpiredLease(lease: Lease): Promise<boolean> {
  const raw = await redis("SMEMBERS", resultReservationsKey(lease.lease_id));
  if (!Array.isArray(raw) || raw.length > MAX_RESULT_RESERVATIONS) return false;
  const records: Array<{ identity: string; digest: string }> = [];
  for (const value of raw) {
    if (typeof value !== "string") return false;
    try {
      const record = JSON.parse(value) as Record<string, unknown>;
      if (typeof record.identity !== "string" || !record.identity ||
          typeof record.digest !== "string" || !/^[0-9a-f]{64}$/.test(record.digest)) return false;
      records.push({ identity: record.identity, digest: record.digest });
    } catch { return false; }
  }
  const identityKeys = await Promise.all(records.map((record) => sha256(record.identity)));
  const keys = [issueLeaseKey(lease.issue_number, lease.language), leaseKey(lease.lease_id),
    resultReservationsKey(lease.lease_id), ...identityKeys.map((digest) => `euroeval:worker:record-identity:${digest}`)];
  const expiresAt = Date.parse(lease.expires_at);
  if (!Number.isFinite(expiresAt) || expiresAt > Date.now()) return false;
  const script = `local current=redis.call('GET',KEYS[1]); if not current then return 0 end;
    local item=cjson.decode(current); if item.lease_id ~= ARGV[1] or item.expires_at ~= ARGV[2] or tonumber(ARGV[3]) > tonumber(redis.call('TIME')[1]) then return 0 end;
    local byid=redis.call('GET',KEYS[2]); if not byid or cjson.decode(byid).lease_id ~= ARGV[1] then return 0 end;
    local records=cjson.decode(ARGV[4]);
    for i=4,#KEYS do local value=redis.call('GET',KEYS[i]); if value then local owned=cjson.decode(value); local record=records[i-3];
      if owned.lease_id == ARGV[1] and owned.digest == record.digest and owned.identity == record.identity then redis.call('DEL',KEYS[i]) end;
    end end;
    redis.call('DEL',KEYS[3]); redis.call('DEL',KEYS[1]); redis.call('DEL',KEYS[2]); return 1`;
  const result = await redis("EVAL", script, String(keys.length), ...keys, lease.lease_id,
    lease.expires_at, String(Math.floor(expiresAt / 1000)), JSON.stringify(records));
  return result === 1 || result === "1";
}

export async function acquireIssueMutex(issue: number): Promise<string | null> {
  const token = randomToken(12);
  return await redisSet(`euroeval:worker:mutex:${issue}`, token, 30, true) ? token : null;
}
export async function releaseIssueMutex(issue: number, token: string): Promise<void> {
  await redis("EVAL", "if redis.call('GET',KEYS[1]) == ARGV[1] then return redis.call('DEL',KEYS[1]) else return 0 end", "1", `euroeval:worker:mutex:${issue}`, token);
}

/** Renew an issue mutex only while its opaque token still owns it. */
export async function renewIssueMutex(issue: number, token: string): Promise<boolean> {
  const result = await redis("EVAL", "if redis.call('GET',KEYS[1]) == ARGV[1] then return redis.call('EXPIRE',KEYS[1],ARGV[2]) else return 0 end", "1", `euroeval:worker:mutex:${issue}`, token, "30");
  return result === 1 || result === "1";
}

const MAX_RESULT_RESERVATIONS = 1024;

/**
 * Reserve an identity and record it in the owning lease's bounded list in one
 * Lua transaction. The list is deliberately maintained before the HF upload so
 * a worker crash cannot strand an uploading identity after lease expiry.
 */
export async function reserveResultIdentity(
  key: string, reservation: string, digest: string, leaseId: string, ttl: number,
  reservationsKey: string, reservationEntry: string,
): Promise<"reserved" | "retry" | "duplicate" | "busy" | "full"> {
  const result = await redis("EVAL", `local function track()
      if redis.call('SISMEMBER',KEYS[2],ARGV[5]) == 0 and redis.call('SCARD',KEYS[2]) >= tonumber(ARGV[6]) then return 0 end;
      redis.call('SADD',KEYS[2],ARGV[5]); redis.call('EXPIRE',KEYS[2],ARGV[7]); return 1
    end;
    local current=redis.call('GET',KEYS[1]);
    if not current then if track() == 0 then return 'full' end; redis.call('SET',KEYS[1],ARGV[1],'EX',ARGV[4]); return 'reserved' end;
    local item=cjson.decode(current);
    if item.digest ~= ARGV[2] then return 'busy' end;
    if item.status == 'uploaded' then if track() == 0 then return 'full' end; return 'duplicate' end;
    if item.lease_id == ARGV[3] then if track() == 0 then return 'full' end; return 'retry' end;
    return 'busy'`, "2", key, reservationsKey, reservation, digest, leaseId, String(ttl), reservationEntry,
    String(MAX_RESULT_RESERVATIONS), String(30 * 24 * 60 * 60));
  return result as "reserved" | "retry" | "duplicate" | "busy" | "full";
}

export async function completeResultIdentity(key: string, reservation: string, digest: string, leaseId: string, ttl: number): Promise<boolean> {
  const result = await redis("EVAL", `local current=redis.call('GET',KEYS[1]);
    if not current then return 0 end; local item=cjson.decode(current);
    if item.digest ~= ARGV[1] or item.lease_id ~= ARGV[2] then return 0 end;
    redis.call('SET',KEYS[1],ARGV[3],'EX',ARGV[4]); return 1`, "1", key, digest, leaseId, reservation, String(ttl));
  return result === 1 || result === "1";
}

export async function abortResultIdentity(key: string, digest: string, leaseId: string): Promise<void> {
  await redis("EVAL", `local current=redis.call('GET',KEYS[1]); if not current then return 0 end;
    local item=cjson.decode(current); if item.digest == ARGV[1] and item.lease_id == ARGV[2] and item.status == 'uploading' then return redis.call('DEL',KEYS[1]) end; return 0`, "1", key, digest, leaseId);
}

export async function releaseResultReservation(key: string, digest: string, leaseId: string): Promise<boolean> {
  const result = await redis("EVAL", `local current=redis.call('GET',KEYS[1]); if not current then return 1 end;
    local item=cjson.decode(current); if item.digest ~= ARGV[1] or item.lease_id ~= ARGV[2] then return 0 end;
    return redis.call('DEL',KEYS[1])`, "1", key, digest, leaseId);
  return result === 1 || result === "1";
}

/**
 * Remove every uploading result reservation for a submission in one Redis
 * transaction. Missing records are already clean; a mismatch aborts without
 * deleting anything so a retry cannot accidentally release another lease.
 */
export async function releaseResultReservations(
  records: Array<{ identity: string; digest: string }>,
  leaseId: string,
): Promise<boolean> {
  const keys = [
    ...(await Promise.all(records.map((record) => sha256(record.identity))))
      .map((digest) => `euroeval:worker:record-identity:${digest}`),
    `euroeval:worker:reservations:${leaseId}`,
  ];
  const script = `local records=cjson.decode(ARGV[1]);
    for i=1,#KEYS-1 do local current=redis.call('GET',KEYS[i]);
      if current then local item=cjson.decode(current); local record=records[i];
        if item.digest ~= record.digest or item.lease_id ~= ARGV[2] then return 0 end;
      end;
    end;
    for i=1,#KEYS-1 do redis.call('DEL',KEYS[i]) end;
    redis.call('DEL',KEYS[#KEYS]); return 1`;
  const result = await redis("EVAL", script, String(keys.length), ...keys,
    JSON.stringify(records), leaseId);
  return result === 1 || result === "1";
}
export function leaseKey(leaseId: string): string {
return `euroeval:worker:lease-id:${leaseId}`; }
export async function getLeaseById(leaseId: string): Promise<Lease | null> { return redisGet<Lease>(leaseKey(leaseId)); }
export async function putLease(lease: Lease): Promise<boolean> {
  const ttl = Math.max(60, Math.ceil((Date.parse(lease.expires_at) - Date.now()) / 1000));
  const result = await redis("EVAL", "if redis.call('EXISTS',KEYS[1]) == 1 or redis.call('EXISTS',KEYS[2]) == 1 then return 0 end; redis.call('SET',KEYS[1],ARGV[1],'EX',ARGV[2]); redis.call('SET',KEYS[2],ARGV[1],'EX',ARGV[2]); return 1", "2", issueLeaseKey(lease.issue_number, lease.language), leaseKey(lease.lease_id), JSON.stringify(lease), String(ttl));
  return result === 1 || result === "1";
}
export async function saveLease(lease: Lease): Promise<boolean> {
  const ttl = Math.max(60, Math.ceil((Date.parse(lease.expires_at) - Date.now()) / 1000));
  const result = await redis("EVAL", "local current=redis.call('GET',KEYS[1]); if not current then return 0 end; local item=cjson.decode(current); if item.lease_id ~= ARGV[1] then return 0 end; redis.call('SET',KEYS[1],ARGV[2],'EX',ARGV[3]); redis.call('SET',KEYS[2],ARGV[2],'EX',ARGV[3]); return 1", "2", issueLeaseKey(lease.issue_number, lease.language), leaseKey(lease.lease_id), lease.lease_id, JSON.stringify(lease), String(ttl));
  return result === 1 || result === "1";
}
export async function deleteLease(lease: Lease): Promise<void> {
  await redis("EVAL", "local current=redis.call('GET',KEYS[1]); if current then local item=cjson.decode(current); if item.lease_id == ARGV[1] then redis.call('DEL',KEYS[1]) end end; local byid=redis.call('GET',KEYS[2]); if byid then local item=cjson.decode(byid); if item.lease_id == ARGV[1] then redis.call('DEL',KEYS[2]) end end; return 1", "2", issueLeaseKey(lease.issue_number, lease.language), leaseKey(lease.lease_id), lease.lease_id);
}
