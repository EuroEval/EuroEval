/* Shared broker implementation. This is the volunteer-worker/v1 JSON protocol. */
declare const process: { env: Record<string, string | undefined> };
export const PROTOCOL_VERSION = "volunteer-worker/v1" as const;
export const REPO = "EuroEval/EuroEval";
export const REQUEST_LABEL = "model evaluation request";
export const TITLE_PREFIX = "[MODEL EVALUATION REQUEST]";
export const VOLUNTEER_MARKER_RE =
  /<!--[\s\S]*?euroeval-volunteer-worker:v1\s+([\s\S]*?)-->/i;
export const VM_MARKER_RE = /<!--[\s\S]*?vm-id:\s*([^\s>]+)\s*-->/i;

const GROUPS: Record<string, string[]> = {
  "Baltic languages (Latvian, Lithuanian)": ["lv", "lt"],
  "Finnic languages (Estonian, Finnish)": ["et", "fi"],
  "Romance languages (Catalan, French, Italian, Portuguese, Romanian, Spanish)": ["ca", "fr", "it", "pt", "ro", "es"],
  "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)": ["da", "fo", "is", "no", "sv"],
  "Slavic languages (Belarusian, Bulgarian, Bosnian, Croatian, Czech, Polish, Serbian, Slovak, Slovenian, Ukrainian)": ["be", "bg", "bs", "hr", "cs", "pl", "sr", "sk", "sl", "uk"],
  "West Germanic languages (Dutch, English, German, Luxembourgish)": ["nl", "en", "de", "lb"],
  Albanian: ["sq"],
  Greek: ["el"],
  Hungarian: ["hu"],
};

export interface VolunteerLeaseMarker {
  winner: string;
  leases: Array<{ language: string; worker: string; contributor: string; expires_at: number }>;
}

export interface WorkerIdentity {
  hash: string;
  contributor: string;
}

export interface Lease {
  issue_number: number;
  language: string;
  worker: string;
  contributor: string;
  model_id: string;
  revision: string;
  image: string;
  worker_version: string;
  expires_at: number;
  lease_id: string;
  result_count?: number;
}

export class ConfigurationError extends Error {}
export class BrokerError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export function env(name: string): string {
  const value = process.env[name];
  if (!value) throw new ConfigurationError(`Deployment is missing ${name}.`);
  return value;
}

export function optionalEnv(name: string, fallback: string): string {
  return process.env[name] || fallback;
}

export function json(status: number, body: unknown, extra?: HeadersInit): Response {
  return new Response(status === 204 ? null : JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "access-control-allow-origin": "*",
      "access-control-allow-methods": "POST, OPTIONS",
      "access-control-allow-headers": "content-type, authorization",
      ...(extra || {}),
    },
  });
}

export function method(req: Request): Response | null {
  if (req.method === "OPTIONS") return json(204, {});
  if (req.method !== "POST") return json(405, { error: "Method not allowed" });
  return null;
}

export async function readJson(req: Request, limit: number): Promise<Record<string, unknown>> {
  const declared = req.headers.get("content-length");
  if (declared && Number(declared) > limit) throw new BrokerError(413, "Request body is too large.");
  const text = await req.text();
  if (new TextEncoder().encode(text).byteLength > limit) throw new BrokerError(413, "Request body is too large.");
  let value: unknown;
  try { value = JSON.parse(text); } catch { throw new BrokerError(400, "Request body must be valid JSON."); }
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new BrokerError(400, "Request body must be a JSON object.");
  return value as Record<string, unknown>;
}

function randomBytes(length: number): Uint8Array {
  const bytes = new Uint8Array(length);
  crypto.getRandomValues(bytes);
  return bytes;
}
export function randomToken(length = 32): string {
  return bytesToBase64(randomBytes(length));
}
function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
export async function sha256(value: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return Array.from(new Uint8Array(digest), (x) => x.toString(16).padStart(2, "0")).join("");
}

async function redisRequest(command: unknown[]): Promise<unknown> {
  const response = await fetch(env("UPSTASH_REDIS_REST_URL"), {
    method: "POST",
    headers: { authorization: `Bearer ${env("UPSTASH_REDIS_REST_TOKEN")}`, "content-type": "application/json" },
    body: JSON.stringify(command),
  });
  if (!response.ok) throw new BrokerError(503, `Upstash returned HTTP ${response.status}.`);
  const payload = (await response.json()) as { result?: unknown; error?: string };
  if (payload.error) throw new BrokerError(503, `Upstash error: ${payload.error}`);
  return payload.result;
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

export function parseVolunteerMarker(body: string | null): VolunteerLeaseMarker | null {
  if (!body) return null;
  const match = body.match(VOLUNTEER_MARKER_RE);
  if (!match) return null;
  try {
    const parsed = JSON.parse(match[1].trim()) as Partial<VolunteerLeaseMarker>;
    if (typeof parsed.winner !== "string" || !parsed.winner || !Array.isArray(parsed.leases)) return null;
    const leases = parsed.leases.filter((lease): lease is VolunteerLeaseMarker["leases"][number] =>
      !!lease && typeof lease === "object" && typeof lease.language === "string" && typeof lease.worker === "string" &&
      typeof lease.contributor === "string" && Number.isFinite(lease.expires_at));
    if (leases.length !== parsed.leases.length) return null;
    return { winner: parsed.winner, leases };
  } catch { return null; }
}
export function hasVolunteerMarker(body: string | null): boolean { return !!body?.match(VOLUNTEER_MARKER_RE); }
export function renderVolunteerMarker(marker: VolunteerLeaseMarker): string {
  return `<!-- euroeval-volunteer-worker:v1 ${JSON.stringify(marker)} -->`;
}
export function replaceVolunteerMarker(body: string, marker: VolunteerLeaseMarker | null): string {
  const without = body.replace(VOLUNTEER_MARKER_RE, "").replace(/\n{3,}/g, "\n\n").trimEnd();
  return marker ? `${without}\n\n${renderVolunteerMarker(marker)}\n` : `${without}\n`;
}

export function selectedLanguages(body: string | null): string[] {
  if (!body) return [];
  const languages: string[] = [];
  for (const [group, codes] of Object.entries(GROUPS)) {
    const pattern = new RegExp(`-\\s*\\[[xX]\\]\\s*${escapeRegExp(group)}(?:\\s|$)`, "m");
    if (pattern.test(body)) languages.push(...codes);
  }
  return languages;
}
function escapeRegExp(value: string): string { return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

export function extractModelId(title: string, body: string | null): string | null {
  const match = body?.match(/(?:^|\n)#{1,6}\s*Model ID\s*\n+([^\n]+)/i);
  if (!match && !title.startsWith(`${TITLE_PREFIX} `)) return null;
  const value = (match?.[1] || title.slice(TITLE_PREFIX.length)).trim().replace(/^[`*_]+|[`*_]+$/g, "");
  return /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+(?::[A-Z0-9_]+)?$/.test(value) ? value : null;
}

interface GithubIssue { number: number; title: string; body: string | null; state: string; assignees?: Array<{ login: string }>; labels?: Array<{ name?: string }>; pull_request?: unknown; }
export async function github(path: string, init: RequestInit = {}): Promise<any> {
  const token = env("GITHUB_TOKEN");
  const response = await fetch(`https://api.github.com${path}`, {
    ...init,
    headers: { accept: "application/vnd.github+json", "x-github-api-version": "2022-11-28", authorization: `Bearer ${token}`, ...(init.headers || {}) },
  });
  if (!response.ok) throw new BrokerError(response.status === 404 ? 404 : 502, `GitHub returned HTTP ${response.status}: ${(await response.text()).slice(0, 300)}`);
  if (response.status === 204) return null;
  return response.json();
}
export async function getIssue(number: number): Promise<GithubIssue> { return github(`/repos/${REPO}/issues/${number}`) as Promise<GithubIssue>; }
export const fetchIssue = getIssue;
export async function listOpenIssues(): Promise<GithubIssue[]> {
  const result: GithubIssue[] = [];
  for (let page = 1; page <= 10; page++) {
    const chunk = await github(`/repos/${REPO}/issues?state=open&labels=${encodeURIComponent(REQUEST_LABEL)}&per_page=100&page=${page}`) as GithubIssue[];
    for (const issue of chunk) if (!issue.pull_request) result.push(issue);
    if (chunk.length < 100) break;
  }
  return result;
}
export async function patchIssue(number: number, body: string): Promise<void> {
  await github(`/repos/${REPO}/issues/${number}`, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ body }) });
}
export async function assignIssue(number: number, login: string): Promise<void> {
  await github(`/repos/${REPO}/issues/${number}/assignees`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ assignees: [login] }) });
}
export async function unassignIssue(number: number, login: string): Promise<void> {
  await github(`/repos/${REPO}/issues/${number}/assignees`, { method: "DELETE", headers: { "content-type": "application/json" }, body: JSON.stringify({ assignees: [login] }) });
}
export async function addIssueLabel(number: number, label: string): Promise<void> {
  await github(`/repos/${REPO}/issues/${number}/labels`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ labels: [label] }) });
}
export async function issueComments(number: number): Promise<Array<{ body?: string }>> {
  return github(`/repos/${REPO}/issues/${number}/comments?per_page=100`) as Promise<Array<{ body?: string }>>;
}
export async function commentIssue(number: number, body: string): Promise<void> {
  await github(`/repos/${REPO}/issues/${number}/comments`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ body }) });
}

export interface ResolvedModel { id: string; revision: string; config: Record<string, unknown>; weight_bytes: number; }
export async function resolveModel(modelId: string): Promise<ResolvedModel> {
  const encoded = modelId.split("/").map(encodeURIComponent).join("/");
  const infoResponse = await fetch(`https://huggingface.co/api/models/${encoded}?blobs=true`);
  if (!infoResponse.ok) throw new BrokerError(422, `Hugging Face model ${modelId} could not be resolved.`);
  const info = await infoResponse.json() as { id?: string; sha?: string; gated?: boolean | string; private?: boolean; siblings?: Array<{ rfilename?: string; size?: number }> };
  if (info.private || info.gated === true || (typeof info.gated === "string" && info.gated !== "false")) throw new BrokerError(422, "The model must be public and ungated.");
  if (!info.sha) throw new BrokerError(422, "Hugging Face did not provide an immutable model revision.");
  const files = info.siblings || [];
  const weights = files.filter((file) => file.rfilename?.toLowerCase().endsWith(".safetensors"));
  const unsafe = files.some((file) => /\.(bin|pt|pth|ckpt|gguf|onnx|h5|msgpack)$/i.test(file.rfilename || ""));
  if (!weights.length || unsafe || files.some((file) => file.rfilename?.endsWith(".py"))) throw new BrokerError(422, "The model must use safetensors only and contain no custom repository Python.");
  const configResponse = await fetch(`https://huggingface.co/${encoded}/raw/${encodeURIComponent(info.sha)}/config.json`);
  if (!configResponse.ok) throw new BrokerError(422, "The model config could not be resolved at its immutable revision.");
  const config = await configResponse.json() as Record<string, unknown>;
  if (config.auto_map || config.custom_code || config.trust_remote_code) throw new BrokerError(422, "Models with auto_map or custom code are not supported.");
  const weightBytes = weights.reduce((total, file) => total + (typeof file.size === "number" && Number.isSafeInteger(file.size) ? file.size : 0), 0);
  if (!weightBytes) throw new BrokerError(422, "Hugging Face did not provide safetensors sizes for a fit estimate.");
  return { id: modelId, revision: info.sha, config, weight_bytes: weightBytes };
}
export function fitsGpu(model: ResolvedModel, hardware: Record<string, unknown>): boolean {
  const free = typeof hardware.free_vram_bytes === "number" ? hardware.free_vram_bytes :
    typeof hardware.free_vram_gib === "number" ? hardware.free_vram_gib * 1024 ** 3 : NaN;
  return Number.isFinite(free) && free > 0 && model.weight_bytes * 1.35 <= free;
}

export function authCredential(req: Request): string {
  const header = req.headers.get("authorization") || "";
  if (!header.startsWith("Bearer ") || header.length < 10) throw new BrokerError(401, "Use the worker credential as Authorization: Bearer <credential>.");
  return header.slice(7).trim();
}
export async function authenticate(req: Request): Promise<WorkerIdentity> {
  const credential = authCredential(req);
  const hash = await sha256(credential);
  const worker = await redisGet<{ contributor: string }>(`euroeval:worker:credential:${hash}`);
  if (!worker?.contributor || worker.contributor.toLowerCase() === "saattrupdan") throw new BrokerError(401, "Worker credential is invalid or revoked.");
  return { hash, contributor: worker.contributor };
}
export function issueLeaseKey(issue: number, language: string): string { return `euroeval:worker:lease:${issue}:${language}`; }
export function leaseKey(leaseId: string): string { return `euroeval:worker:lease-id:${leaseId}`; }
export async function getLeaseById(leaseId: string): Promise<Lease | null> { return redisGet<Lease>(leaseKey(leaseId)); }
export async function putLease(lease: Lease): Promise<boolean> {
  const ttl = Math.max(60, Math.ceil((lease.expires_at - Date.now()) / 1000));
  // NX is the atomic per-language lease boundary. GitHub remains the canonical queue.
  const first = await redisSet(issueLeaseKey(lease.issue_number, lease.language), JSON.stringify(lease), ttl, true);
  if (!first) return false;
  await redisSet(leaseKey(lease.lease_id), JSON.stringify(lease), ttl);
  return true;
}
export async function saveLease(lease: Lease): Promise<boolean> {
  const ttl = Math.max(60, Math.ceil((lease.expires_at - Date.now()) / 1000));
  const result = await redis("EVAL", "local current=redis.call('GET',KEYS[1]); if not current then return 0 end; local item=cjson.decode(current); if item.lease_id ~= ARGV[1] then return 0 end; redis.call('SET',KEYS[1],ARGV[2],'EX',ARGV[3]); redis.call('SET',KEYS[2],ARGV[2],'EX',ARGV[3]); return 1", "2", issueLeaseKey(lease.issue_number, lease.language), leaseKey(lease.lease_id), lease.lease_id, JSON.stringify(lease), String(ttl));
  return result === 1 || result === "1";
}
export async function deleteLease(lease: Lease): Promise<void> {
  await redis("EVAL", "local current=redis.call('GET',KEYS[1]); if current then local item=cjson.decode(current); if item.lease_id == ARGV[1] then redis.call('DEL',KEYS[1]) end end; local byid=redis.call('GET',KEYS[2]); if byid then local item=cjson.decode(byid); if item.lease_id == ARGV[1] then redis.call('DEL',KEYS[2]) end end; return 1", "2", issueLeaseKey(lease.issue_number, lease.language), leaseKey(lease.lease_id), lease.lease_id);
}

export function validateRecord(record: unknown, expected: { modelId: string; revision: string; language: string }): { identity: string; failed: number } {
  if (!record || typeof record !== "object" || Array.isArray(record)) throw new BrokerError(422, "record must be an EEE JSON object.");
  const value = record as Record<string, any>;
  if (typeof value.schema_version !== "string" || !value.model_info || !value.eval_library || !Array.isArray(value.evaluation_results)) throw new BrokerError(422, "record is not a complete EEE record.");
  const modelInfo = value.model_info as Record<string, any>;
  const expectedId = `${expected.modelId}@${expected.revision}`;
  if (modelInfo.id !== expectedId) throw new BrokerError(422, `record model_info.id must be ${expectedId}.`);
  const details = (value.eval_library as Record<string, any>).additional_details;
  if (!details || typeof details.dataset !== "string" || !details.dataset) throw new BrokerError(422, "record has no canonical dataset identity.");
  const recordLanguage = details.language ?? value.language;
  if (recordLanguage !== undefined && recordLanguage !== expected.language) throw new BrokerError(422, "record language does not match the lease.");
  let failed = 0;
  const envelopeFailed = details.num_failed_instances ?? details.failed_instances;
  if (envelopeFailed !== undefined) {
    const count = typeof envelopeFailed === "number" ? envelopeFailed : Number(envelopeFailed);
    if (!Number.isFinite(count) || count < 0 || count !== 0 || (Array.isArray(envelopeFailed) && envelopeFailed.length)) throw new BrokerError(422, "records with failed instances cannot be submitted.");
  }
  const scores: unknown[] = [];
  for (const result of value.evaluation_results) {
    if (!result || typeof result !== "object") throw new BrokerError(422, "evaluation_results contains an invalid item.");
    const scoreDetails = (result as any).score_details;
    const score = scoreDetails?.score;
    if (typeof score !== "number" || !Number.isFinite(score) || score < 0 || score > 100) throw new BrokerError(422, "every evaluation score must be finite and between 0 and 100.");
    const resultDetails = scoreDetails?.details;
    const rawFailed = resultDetails?.num_failed_instances;
    if (rawFailed !== undefined) {
      const count = typeof rawFailed === "number" ? rawFailed : Number(rawFailed);
      if (!Number.isFinite(count) || count < 0 || count !== 0) throw new BrokerError(422, "records with failed instances cannot be submitted.");
      failed += count;
    }
    if (Array.isArray(resultDetails?.failed_instances) && resultDetails.failed_instances.length) throw new BrokerError(422, "records with failed instances cannot be submitted.");
    scores.push(score);
  }
  const topLevelFailed = value.num_failed_instances;
  if (topLevelFailed !== undefined && Number(topLevelFailed) !== 0) throw new BrokerError(422, "records with failed instances cannot be submitted.");
  if (!value.evaluation_results.length) throw new BrokerError(422, "record contains no evaluation results.");
  const identity = JSON.stringify([modelInfo.id, details.dataset, details.validation_split ?? null, details.few_shot ?? null]);
  return { identity, failed };
}

export async function uploadStaging(path: string, content: string): Promise<void> {
  const bucket = env("HF_STAGING_BUCKET");
  const token = env("HF_TOKEN");
  const endpoint = env("HF_STAGING_UPLOAD_URL");
  let url: URL;
  try { url = new URL(endpoint); } catch { throw new ConfigurationError("HF_STAGING_UPLOAD_URL must be an absolute HTTPS URL."); }
  if (url.protocol !== "https:") throw new ConfigurationError("HF_STAGING_UPLOAD_URL must use HTTPS.");
  const response = await fetch(url, { method: "POST", headers: { authorization: `Bearer ${token}`, "content-type": "application/json" }, body: JSON.stringify({ bucket, path, content }) });
  if (!response.ok) throw new BrokerError(502, `Hugging Face staging upload failed with HTTP ${response.status}.`);
}

export async function recordDigest(record: unknown): Promise<string> {
  return sha256(JSON.stringify(record));
}
export function contributorLabel(login: string): string { return login === "saattrupdan" ? "anonymous worker" : login; }
