import { BrokerError, ConfigurationError } from "./protocol.js";
import type { TrustedScope } from "./model.js";

const COVERAGE_DEADLINE_MS = 4_000;
const OBJECT_LIMIT_BYTES = 1_048_576;
const COVERAGE_PARALLELISM = 6;
const BUCKET_REPO = "EuroEval/results";

export type Coverage = {
  allowed_identity_suffix_sets: string[][];
  original_identity_suffix_sets: string[][];
  covered_identity_suffixes: string[];
};

/** Resolve canonical result paths and narrow trusted identity alternatives. */
export async function narrowCoverage(
  scope: TrustedScope,
  modelId: string,
  revision: string,
  requestedVersion: string,
  fetcher: typeof fetch = fetch,
): Promise<Coverage> {
  const deadline = Date.now() + COVERAGE_DEADLINE_MS;
  const cache = new Map<string, Promise<boolean>>();
  const suffixes = [...new Set(scope.allowed_identity_suffix_sets.flat())];
  const check = (suffix: string): Promise<boolean> => {
    const identity = parseIdentity(suffix);
    const key = resultPath(modelId, identity);
    let result = cache.get(key);
    if (!result) {
      result = lookup({
        key,
        modelId,
        revision,
        requestedVersion,
        deadline,
        fetcher,
      });
      cache.set(key, result);
    }
    return result;
  };
  const covered = new Set<string>();
  for (let offset = 0; offset < suffixes.length; offset += COVERAGE_PARALLELISM) {
    if (Date.now() >= deadline) throw new BrokerError(503, "Canonical coverage timed out; retry claim.");
    const batch = suffixes.slice(offset, offset + COVERAGE_PARALLELISM);
    const outcomes = await Promise.all(batch.map(async (suffix) => [suffix, await check(suffix)] as const));
    outcomes.filter(([, exists]) => exists).forEach(([suffix]) => covered.add(suffix));
  }
  const original = scope.allowed_identity_suffix_sets.map((set) => [...set]);
  const completeAlternative = original.some((set) =>
    set.every((item) => covered.has(item)));
  const narrowed = completeAlternative ? [] : original
    .map((set) => set.filter((item) => !covered.has(item)))
    .filter((set) => set.length > 0);
  return {
    allowed_identity_suffix_sets: narrowed,
    original_identity_suffix_sets: original,
    covered_identity_suffixes: [...covered],
  };
}

async function lookup(args: {
  key: string;
  modelId: string;
  revision: string;
  requestedVersion: string;
  deadline: number;
  fetcher: typeof fetch;
}): Promise<boolean> {
  const remaining = args.deadline - Date.now();
  if (remaining <= 0) throw new BrokerError(503, "Canonical coverage timed out; retry claim.");
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => {
      controller.abort();
      reject(new BrokerError(503, "Canonical coverage timed out; retry claim."));
    }, remaining);
  });
  try {
    const path = encodePath(args.key);
    const response = await Promise.race([
      args.fetcher(
        `https://huggingface.co/buckets/${BUCKET_REPO}/resolve/${path}`,
        { signal: controller.signal, headers: { Range: `bytes=0-${OBJECT_LIMIT_BYTES - 1}` } },
      ),
      timeout,
    ]);
    if (response.status === 404) return false;
    if (!response.ok) throw new BrokerError(503, "Canonical coverage is unavailable; retry claim.");
    const length = Number(response.headers.get("content-length"));
    if (Number.isFinite(length) && length > OBJECT_LIMIT_BYTES) {
      throw new BrokerError(503, "Canonical result exceeds the coverage limit; retry claim.");
    }
    const text = await Promise.race([
      readBounded(response, OBJECT_LIMIT_BYTES),
      timeout,
    ]);
    let record: Record<string, any>;
    try {
      record = JSON.parse(text);
    } catch {
      throw new BrokerError(503, "Canonical result is malformed; retry claim.");
    }
    if (!record || typeof record !== "object" || Array.isArray(record) ||
        !["0.2.1", "0.3.0"].includes(record.schema_version) ||
        !Array.isArray(record.evaluation_results)) {
      throw new BrokerError(503, "Canonical result is malformed; retry claim.");
    }
    const model = record.model_info;
    const library = record.eval_library;
    const details = library?.additional_details;
    if (!model || typeof model !== "object" || !library ||
        typeof library !== "object" || library.name !== "euroeval" ||
        typeof library.version !== "string" || !library.version ||
        !details || typeof details !== "object" ||
        typeof details.dataset !== "string" ||
        !isBoolOrNull(details.validation_split) ||
        !isBoolOrNull(details.few_shot)) {
      throw new BrokerError(503, "Canonical result is malformed; retry claim.");
    }
    const aliases = [model.id, model.name, ...(Array.isArray(model.aliases) ? model.aliases : [])];
    if (!aliases.includes(args.modelId) || model.revision !== args.revision ||
        library.name !== "euroeval") return false;
    const pathIdentity = parseIdentityFromPath(args.key);
    if (typeof details?.dataset !== "string" ||
        details.dataset.replace(/\//g, "_") !== pathIdentity[0] ||
        boolOrNull(details.validation_split) !== pathIdentity[1] ||
        boolOrNull(details.few_shot) !== pathIdentity[2]) return false;
    const storedVersion = compareVersion(library.version, args.requestedVersion);
    if (storedVersion === null) throw new BrokerError(503, "Canonical result version is malformed; retry claim.");
    return storedVersion >= 0;
  } catch (error) {
    if (error instanceof BrokerError) throw error;
    throw new BrokerError(503, "Canonical coverage lookup failed; retry claim.");
  } finally {
    if (timer) clearTimeout(timer);
  }
}

async function readBounded(response: Response, limit: number): Promise<string> {
  if (!response.body) throw new BrokerError(503, "Canonical result body is unavailable; retry claim.");
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > limit) {
        await reader.cancel();
        throw new BrokerError(503, "Canonical result exceeds the coverage limit; retry claim.");
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new TextDecoder().decode(bytes);
}

function parseIdentity(suffix: string): [string, boolean | null, boolean | null] {
  let parsed: unknown;
  try { parsed = JSON.parse(suffix); } catch { throw new ConfigurationError("Trusted scope identity is malformed."); }
  if (!Array.isArray(parsed) || typeof parsed[0] !== "string" || !parsed[0] ||
      ![true, false, null].includes(parsed[1]) || ![true, false, null].includes(parsed[2])) {
    throw new ConfigurationError("Trusted scope identity is malformed.");
  }
  return parsed as [string, boolean | null, boolean | null];
}

function resultPath(modelId: string, identity: [string, boolean | null, boolean | null]): string {
  const [dataset, split, shot] = identity;
  const safe = (value: string): string => value.replace(/\//g, "_");
  const splitLabel = split === true ? "val" : split === false ? "test" : "none";
  const shotLabel = shot === true ? "fewshot" : shot === false ? "zeroshot" : "none";
  return `${safe(modelId)}/${safe(dataset)}__${splitLabel}__${shotLabel}.json`;
}

function isBoolOrNull(value: unknown): boolean {
  return value === true || value === false || value === null || value === "true" ||
    value === "false" || value === "null";
}

function boolOrNull(value: unknown): unknown {
  if (value === true || value === "true") return true;
  if (value === false || value === "false") return false;
  return value === null || value === "null" ? null : "invalid";
}

function parseIdentityFromPath(path: string): [string, boolean | null, boolean | null] {
  const filename = path.slice(path.lastIndexOf("/") + 1).replace(/\.json$/, "");
  const match = filename.match(/^(.*)__(val|test|none)__(fewshot|zeroshot|none)$/);
  if (!match) throw new BrokerError(503, "Canonical result path is malformed; retry claim.");
  return [match[1], match[2] === "val" ? true : match[2] === "test" ? false : null,
    match[3] === "fewshot" ? true : match[3] === "zeroshot" ? false : null];
}

function encodePath(path: string): string {
  return path.split("/").map(encodeURIComponent).join("/");
}

function compareVersion(left: string, right: string): number | null {
  const parse = (value: string): number[] | null => {
    const match = value.replace(/\.dev$/, ".dev0").match(/^(\d+(?:\.\d+)*)(?:(a|b|rc|\.dev)(\d*))?$/i);
    if (!match) return null;
    const numbers = match[1].split(".").map(Number);
    const phase = match[2]?.toLowerCase();
    const stage = phase === ".dev" ? -4 : phase === "a" ? -3 : phase === "b" ? -2 : phase === "rc" ? -1 : 0;
    return [...numbers, stage, Number(match[3] || 0)];
  };
  const a = parse(left);
  const b = parse(right);
  if (!a || !b) return null;
  const width = Math.max(a.length, b.length);
  for (let index = 0; index < width; index++) {
    const difference = (a[index] || 0) - (b[index] || 0);
    if (difference) return Math.sign(difference);
  }
  return 0;
}
