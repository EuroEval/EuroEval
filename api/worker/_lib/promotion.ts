import { BrokerError, sha256 } from "./protocol.ts";
import { redis, redisGet, redisSet } from "./redis.ts";
import type { PromotionRecord, PromotionReservation } from "./protocol.ts";

export const PROMOTION_RESERVATION_TTL = 48 * 60 * 60;
export const PROMOTION_TERMINAL_TTL = 30 * 24 * 60 * 60;

export function promotionReservationKey(issueNumber: number, submissionId: string): string {
  return `euroeval:worker:promotion:${issueNumber}:${submissionId}`;
}

export async function promotionIdentityKey(path: string): Promise<string> {
  return `euroeval:worker:promotion-identity:${await sha256(path)}`;
}

export function parsePromotionRecords(value: unknown): PromotionRecord[] {
  if (!Array.isArray(value) || value.length === 0 || value.length > 1024) {
    throw new BrokerError(400, "records must contain 1 to 1024 items.");
  }
  const result = value.map((item) => {
    if (!item || typeof item !== "object") {
      throw new BrokerError(400, "Invalid promotion record.");
    }
    const record = item as Record<string, unknown>;
    if (typeof record.identity !== "string" || record.identity.length === 0 ||
        record.identity.length > 4096 || typeof record.canonical_path !== "string" ||
        !safeCanonicalPath(record.canonical_path) || typeof record.digest !== "string" ||
        !/^[0-9a-f]{64}$/.test(record.digest)) {
      throw new BrokerError(400, "Promotion records require a safe path and SHA256 digest.");
    }
    const identity = parseIdentity(record.identity);
    if (canonicalPath(identity) !== record.canonical_path) {
      throw new BrokerError(400, "Promotion record path does not match its identity.");
    }
    return {
      identity: record.identity,
      canonical_path: record.canonical_path,
      digest: record.digest,
    };
  });
  const paths = result.map((item) => item.canonical_path);
  if (new Set(paths).size !== paths.length) {
    throw new BrokerError(409, "Promotion paths are not unique.");
  }
  return result.sort((a, b) => a.identity.localeCompare(b.identity));
}

function safeCanonicalPath(path: string): boolean {
  if (path.length > 1024 || path.startsWith("/") || path.endsWith("/") ||
      path.includes("\\") || path.split("/").length !== 2 ||
      [...path].some((character) => character.charCodeAt(0) < 32)) return false;
  return !path.split("/").some((part) => part === "" || part === "." || part === "..");
}

function parseIdentity(value: string): [string, string, boolean | null, boolean | null] {
  let parsed: unknown;
  try { parsed = JSON.parse(value); } catch { throw new BrokerError(400, "Invalid promotion identity."); }
  if (!Array.isArray(parsed) || parsed.length !== 4 || typeof parsed[0] !== "string" ||
      !parsed[0] || typeof parsed[1] !== "string" || !parsed[1] ||
      (typeof parsed[2] !== "boolean" && parsed[2] !== null) ||
      (typeof parsed[3] !== "boolean" && parsed[3] !== null)) {
    throw new BrokerError(400, "Invalid promotion identity.");
  }
  return parsed as [string, string, boolean | null, boolean | null];
}

function canonicalPath(identity: [string, string, boolean | null, boolean | null]): string {
  const split = identity[2] === true ? "val" : identity[2] === false ? "test" : "none";
  const shot = identity[3] === true ? "fewshot" : identity[3] === false ? "zeroshot" : "none";
  return `${identity[0].replace(/\//g, "_")}/${identity[1].replace(/\//g, "_")}__${split}__${shot}.json`;
}

export async function reservePromotionReservation(
  reservation: PromotionReservation,
): Promise<"reserved" | "busy" | "conflict"> {
  const keys = [reservationReservationKey(reservation),
    ...(await Promise.all(reservation.records.map((record) => promotionIdentityKey(record.canonical_path))))];
  const script = `local current=redis.call('GET',KEYS[1]);
    local reservation=cjson.decode(ARGV[1]); local records=cjson.decode(ARGV[3]);
    if current then local item=cjson.decode(current); if item.token ~= ARGV[2] then return 'busy' end;
      if (item.decision_digest and not reservation.decision_digest) or
          (not item.decision_digest and reservation.decision_digest) or
          (item.decision_digest and item.decision_digest ~= reservation.decision_digest) then return 'conflict' end;
      if #item.records ~= #reservation.records then return 'conflict' end;
      for i=1,#item.records do
        if item.records[i].identity ~= reservation.records[i].identity or
            item.records[i].canonical_path ~= reservation.records[i].canonical_path or
            item.records[i].digest ~= reservation.records[i].digest then return 'conflict' end
      end;
      if item.status == 'terminal' then return 'reserved' end;
    end;
    for i=2,#KEYS do local value=redis.call('GET',KEYS[i]); if value then
      local item=cjson.decode(value); if item.digest ~= records[i-1].digest or item.identity ~= records[i-1].identity then return 'conflict' end;
      if item.status ~= 'terminal' and item.token ~= ARGV[2] then return 'busy' end;
    end end;
    redis.call('SET',KEYS[1],ARGV[1],'EX',ARGV[4]);
    for i=2,#KEYS do local value=redis.call('GET',KEYS[i]); if not value then
      local record=records[i-1]; redis.call('SET',KEYS[i],cjson.encode({token=ARGV[2],digest=record.digest,status='reserved',issue_number=reservation.issue_number,submission_id=reservation.submission_id,identity=record.identity,canonical_path=record.canonical_path}),'EX',ARGV[4]);
    else local item=cjson.decode(value); if item.status ~= 'terminal' then
      redis.call('EXPIRE',KEYS[i],ARGV[4]);
    end end end; return 'reserved'`;
  const result = await redis("EVAL", script, String(keys.length), ...keys,
    JSON.stringify(reservation), reservation.token, JSON.stringify(reservation.records),
    String(PROMOTION_RESERVATION_TTL));
  return result as "reserved" | "busy" | "conflict";
}

function reservationReservationKey(reservation: PromotionReservation): string {
  return promotionReservationKey(reservation.issue_number, reservation.submission_id);
}

export async function bindPromotionDecision(
  reservation: PromotionReservation,
  decisionDigest: string,
): Promise<"bound" | "expired" | "mismatch"> {
  if (!/^[0-9a-f]{64}$/.test(decisionDigest)) {
    throw new BrokerError(400, "decision_digest must be a SHA256 digest.");
  }
  const key = reservationReservationKey(reservation);
  const script = `local current=redis.call('GET',KEYS[1]); if not current then return 'expired' end;
    local item=cjson.decode(current); if item.token ~= ARGV[1] then return 'expired' end;
    if item.status == 'terminal' then
      if item.decision_digest ~= ARGV[2] then return 'mismatch' end;
      return 'bound';
    end;
    if item.status ~= 'reserved' then return 'expired' end;
    if item.decision_digest and item.decision_digest ~= ARGV[2] then return 'mismatch' end;
    item.decision_digest=ARGV[2]; redis.call('SET',KEYS[1],cjson.encode(item),'EX',ARGV[3]); return 'bound'`;
  const result = await redis("EVAL", script, "1", key, reservation.token, decisionDigest,
    String(PROMOTION_RESERVATION_TTL));
  return result as "bound" | "expired" | "mismatch";
}

export async function completePromotionReservation(reservation: PromotionReservation): Promise<boolean> {
  const keys = [reservationReservationKey(reservation),
    ...(await Promise.all(reservation.records.map((record) => promotionIdentityKey(record.canonical_path))))];
  const script = `local current=redis.call('GET',KEYS[1]); if not current then return 0 end;
    local currentReservation=cjson.decode(current); if currentReservation.token ~= ARGV[2] or currentReservation.decision_digest ~= ARGV[4] then return 0 end;
    local records=cjson.decode(ARGV[3]);
    for i=2,#KEYS do local value=redis.call('GET',KEYS[i]); if value then local item=cjson.decode(value);
      if item.digest ~= records[i-1].digest or item.identity ~= records[i-1].identity or (item.status ~= 'terminal' and item.token ~= ARGV[2]) then return 0 end end end;
    if currentReservation.status == 'terminal' then return 1 end;
    redis.call('SET',KEYS[1],ARGV[1]);
    for i=2,#KEYS do local value=redis.call('GET',KEYS[i]); if not value then
      local record=records[i-1]; redis.call('SET',KEYS[i],cjson.encode({token=ARGV[2],digest=record.digest,status='terminal',issue_number=currentReservation.issue_number,submission_id=currentReservation.submission_id,identity=record.identity,canonical_path=record.canonical_path}));
    elseif cjson.decode(value).status ~= 'terminal' then local item=cjson.decode(value); item.status='terminal'; redis.call('SET',KEYS[i],cjson.encode(item)); end end; return 1`;
  const result = await redis("EVAL", script, String(keys.length), ...keys,
    JSON.stringify({ ...reservation, status: "terminal" }), reservation.token,
    JSON.stringify(reservation.records), reservation.decision_digest || "");
  return result === 1 || result === "1";
}

export async function getPromotionReservation(
  issueNumber: number,
  submissionId: string,
): Promise<PromotionReservation | null> {
  return redisGet<PromotionReservation>(promotionReservationKey(issueNumber, submissionId));
}

export async function savePromotionReservation(
  reservation: PromotionReservation,
  terminal: boolean,
): Promise<void> {
  const key = promotionReservationKey(reservation.issue_number, reservation.submission_id);
  const value = JSON.stringify({ ...reservation, status: terminal ? "terminal" : reservation.status });
  if (terminal) {
    const result = await redis(
      "EVAL",
      `local current=redis.call('GET',KEYS[1]); if not current then return 0 end;
       local item=cjson.decode(current); if item.token ~= ARGV[3] or item.decision_digest ~= ARGV[4] then return 0 end;
       if item.status == 'terminal' then return 1 end;
       redis.call('SET',KEYS[1],ARGV[1],'EX',ARGV[2]); return 1`,
      "1",
      key,
      value,
      String(PROMOTION_TERMINAL_TTL),
      reservation.token,
      reservation.decision_digest || "",
    );
    if (result !== 1 && result !== "1") {
      throw new BrokerError(409, "Promotion reservation was replaced during promotion.");
    }
    return;
  }
  await redis("SET", key, value, "EX", String(PROMOTION_RESERVATION_TTL));
}
