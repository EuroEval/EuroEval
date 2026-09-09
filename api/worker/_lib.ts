/* Shared broker implementation. This is the volunteer-worker/v1 JSON protocol. */
import { uploadFile } from "@huggingface/hub";
declare const process: { env: Record<string, string | undefined> };
export const PROTOCOL_VERSION = "volunteer-worker/v1" as const;
export const REPO = "EuroEval/EuroEval";
export const REQUEST_LABEL = "model evaluation request";
export const TITLE_PREFIX = "[MODEL EVALUATION REQUEST]";
export const VOLUNTEER_MARKER_RE =
  /<!--[\s\S]*?euroeval-volunteer-worker:v1\s+([\s\S]*?)-->/i;
export const VM_MARKER_RE = /<!--[\s\S]*?vm-id:\s*([^\s>]+)\s*-->/i;

export const GROUPS: Record<string, string[]> = {
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

export interface VolunteerSubmission {
  submission_id: string;
  language: string;
  manifest_path: string;
  submitted_at: string;
}

export interface VolunteerLeaseMarker {
  protocol_version: typeof PROTOCOL_VERSION;
  coordinator: string;
  submission: "active" | "submitted" | "completed" | "released";
  leases: Array<{ lease_id: string; language: string; worker: string; contributor: string; expires_at: string }>;
  submissions?: VolunteerSubmission[];
  completed_languages?: string[];
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
  model_revision: string;
  euroeval_version: string;
  image_digest: string;
  worker_version: string;
  expires_at: string;
  lease_id: string;
  result_count?: number;
  model_profile: string;
  expected_scope: {
    policy_version: string;
    language_group: string;
    identity_suffixes: string[];
    count: number;
    warnings: string[];
  };
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

export function requireProtocol(body: Record<string, unknown>): void {
  if (body.protocol_version !== PROTOCOL_VERSION) throw new BrokerError(400, "protocol_version must be volunteer-worker/v1.");
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
export async function fetchWithRetry(input: string, init: RequestInit = {}): Promise<Response> {
  let response: Response | undefined;
  for (let attempt = 0; attempt < 3; attempt++) {
    try { response = await fetch(input, init); } catch (error) {
      if (attempt === 2) throw error;
      await new Promise((resolve) => setTimeout(resolve, 100 * (attempt + 1))); continue;
    }
    if (![408, 425, 429, 500, 502, 503, 504].includes(response.status) || attempt === 2) return response;
    await new Promise((resolve) => setTimeout(resolve, 100 * (attempt + 1)));
  }
  return response as Response;
}

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

export function parseVolunteerMarker(body: string | null): VolunteerLeaseMarker | null {
  if (!body) return null;
  const matches = [...body.matchAll(new RegExp(VOLUNTEER_MARKER_RE.source, "gi"))];
  if (matches.length !== 1) return null;
  try {
    const parsed = JSON.parse(matches[0][1].trim()) as Partial<VolunteerLeaseMarker>;
    if (parsed.protocol_version !== PROTOCOL_VERSION || typeof parsed.coordinator !== "string" ||
        !parsed.coordinator || !["active", "submitted", "completed", "released"].includes(parsed.submission || "") ||
        !Array.isArray(parsed.leases)) return null;
    const leases = parsed.leases.filter((lease): lease is VolunteerLeaseMarker["leases"][number] =>
      !!lease && typeof lease === "object" && typeof lease.lease_id === "string" &&
      typeof lease.language === "string" && typeof lease.worker === "string" &&
      typeof lease.contributor === "string" && typeof lease.expires_at === "string" &&
      !Number.isNaN(Date.parse(lease.expires_at)));
    if (leases.length !== parsed.leases.length || new Set(leases.map((item) => item.lease_id)).size !== leases.length || new Set(leases.map((item) => item.language)).size !== leases.length) return null;
    const submissions = Array.isArray(parsed.submissions) ? parsed.submissions.filter((item): item is VolunteerSubmission =>
      !!item && typeof item === "object" && typeof item.submission_id === "string" &&
      typeof item.language === "string" && typeof item.manifest_path === "string" &&
      typeof item.submitted_at === "string") : undefined;
    if (parsed.submissions !== undefined && (submissions === undefined || submissions.length !== parsed.submissions.length || new Set(submissions.map((item) => item.submission_id)).size !== submissions.length || new Set(submissions.map((item) => item.language)).size !== submissions.length)) return null;
    const completed = Array.isArray(parsed.completed_languages) ? parsed.completed_languages.filter((item): item is string => typeof item === "string" && !!item) : undefined;
    if (parsed.completed_languages !== undefined && (completed === undefined || completed.length !== parsed.completed_languages.length)) return null;
    return { protocol_version: PROTOCOL_VERSION, coordinator: parsed.coordinator,
      submission: parsed.submission as VolunteerLeaseMarker["submission"], leases,
      ...(submissions === undefined ? {} : { submissions }), ...(completed === undefined ? {} : { completed_languages: completed }) };
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
  const response = await fetchWithRetry(`https://api.github.com${path}`, {
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

export interface ResolvedModel {
  id: string;
  revision: string;
  config: Record<string, unknown>;
  weight_bytes: number;
  repo_bytes: number;
  model_profile: string;
}
export async function resolveModel(modelId: string): Promise<ResolvedModel> {
  const encoded = modelId.split("/").map(encodeURIComponent).join("/");
  const infoResponse = await fetchWithRetry(`https://huggingface.co/api/models/${encoded}?blobs=true`);
  if (!infoResponse.ok) throw new BrokerError(422, `Hugging Face model ${modelId} could not be resolved.`);
  const info = await infoResponse.json() as { id?: string; sha?: string; gated?: boolean | string; private?: boolean; siblings?: Array<{ rfilename?: string; size?: number }> };
  if (info.id && info.id.toLowerCase() !== modelId.toLowerCase()) throw new BrokerError(422, "Hugging Face returned metadata for a different model.");
  if (info.private || info.gated === true || (typeof info.gated === "string" && info.gated !== "false")) throw new BrokerError(422, "The model must be public and ungated.");
  if (!info.sha) throw new BrokerError(422, "Hugging Face did not provide an immutable model revision.");
  const files = info.siblings || [];
  const weights = files.filter((file) => file.rfilename?.toLowerCase().endsWith(".safetensors"));
  const unsafe = files.some((file) => /\.(bin|pt|pth|ckpt|gguf|onnx|h5|msgpack)$/i.test(file.rfilename || ""));
  if (!weights.length || unsafe || files.some((file) => file.rfilename?.toLowerCase().endsWith(".py"))) throw new BrokerError(422, "The model must use safetensors only and contain no custom repository Python.");
  const configResponse = await fetchWithRetry(`https://huggingface.co/${encoded}/raw/${encodeURIComponent(info.sha)}/config.json`);
  if (!configResponse.ok) throw new BrokerError(422, "The model config could not be resolved at its immutable revision.");
  const config = await configResponse.json() as Record<string, unknown>;
  if (config.auto_map || config.custom_code || config.trust_remote_code) throw new BrokerError(422, "Models with auto_map or custom code are not supported.");
  const weightBytes = weights.reduce((total, file) => total + (typeof file.size === "number" && Number.isSafeInteger(file.size) ? file.size : 0), 0);
  const repoBytes = files.reduce((total, file) => total + (typeof file.size === "number" && Number.isSafeInteger(file.size) && file.size >= 0 ? file.size : 0), 0);
  if (!weightBytes || !repoBytes || files.some((file) => typeof file.size !== "number" || !Number.isSafeInteger(file.size) || file.size < 0)) throw new BrokerError(422, "Hugging Face did not provide complete model file sizes for a fit estimate.");
  const modelProfile = modelProfileFor(config);
  if (!modelProfile) throw new BrokerError(422, "The model architecture is not admitted to the volunteer worker.");
  return { id: modelId, revision: info.sha, config, weight_bytes: weightBytes, repo_bytes: repoBytes, model_profile: modelProfile };
}

export function fitsGpu(model: ResolvedModel, hardware: Record<string, unknown>): boolean {
  if (typeof hardware.free_disk_bytes !== "number" || !Number.isFinite(hardware.free_disk_bytes) || hardware.free_disk_bytes < model.repo_bytes) return false;
  if (!Array.isArray(hardware.gpus)) return false;
  return hardware.gpus.some((gpu) => {
    if (!gpu || typeof gpu !== "object") return false;
    const item = gpu as Record<string, unknown>;
    const free = item.free_memory_bytes;
    return typeof free === "number" && Number.isFinite(free) && free > 0 && model.weight_bytes * 1.35 <= free;
  });
}

const ARCHITECTURE_PROFILES: Record<string, string> = {
  bert: "bert", roberta: "roberta", xlm_roberta: "roberta", eurobert: "eurobert",
  llama: "llama", mistral: "mistral", qwen2: "qwen", qwen3: "qwen", gemma: "gemma",
  phi3: "phi", phi: "phi", falcon: "falcon", gpt2: "gpt2",
};
function modelProfileFor(config: Record<string, unknown>): string | null {
  const architectures = config.architectures;
  if (!Array.isArray(architectures) || architectures.length !== 1 || typeof architectures[0] !== "string") return null;
  const name = architectures[0].replace(/For[A-Za-z]+$/, "").replace(/Model$/, "").toLowerCase();
  const key = Object.keys(ARCHITECTURE_PROFILES).find((candidate) => name.includes(candidate));
  return key ? ARCHITECTURE_PROFILES[key] : null;
}
export function languageGroup(language: string): string | null {
  return Object.entries(GROUPS).find(([, codes]) => codes.includes(language))?.[0] || null;
}

type ScopeEntry = { euroeval_version: string; model_profile: string; language_group: string; identity_suffixes: string[]; count?: number; warnings?: string[] };
type ScopePolicy = { policy_version: string; policies: ScopeEntry[] };
export function expectedScope(euroevalVersion: string, profile: string, language: string): ScopeEntry & { policy_version: string } {
  const group = languageGroup(language);
  if (!group) throw new BrokerError(422, "The leased language has no trusted language group.");
  const raw = process.env.VOLUNTEER_SCOPE_POLICY_JSON;
  if (!raw) throw new BrokerError(503, "No trusted volunteer expected-scope policy is configured.");
  let policy: ScopePolicy;
  try { policy = JSON.parse(raw) as ScopePolicy; } catch { throw new ConfigurationError("VOLUNTEER_SCOPE_POLICY_JSON is invalid JSON."); }
  if (typeof policy.policy_version !== "string" || !Array.isArray(policy.policies)) throw new ConfigurationError("VOLUNTEER_SCOPE_POLICY_JSON has an invalid schema.");
  const match = policy.policies.find((item) => item.euroeval_version === euroevalVersion && item.model_profile === profile && item.language_group === group);
  if (!match || !Array.isArray(match.identity_suffixes) || !match.identity_suffixes.length) throw new BrokerError(422, "This model profile and language group has no trusted expected scope.");
  if (match.identity_suffixes.some((item) => typeof item !== "string") || new Set(match.identity_suffixes).size !== match.identity_suffixes.length || match.count !== undefined && match.count !== match.identity_suffixes.length) throw new ConfigurationError("Trusted expected scope contains invalid or duplicate identities.");
  if (match.warnings && (!Array.isArray(match.warnings) || match.warnings.some((item) => typeof item !== "string"))) throw new ConfigurationError("Trusted expected scope contains invalid warnings.");
  return { ...match, policy_version: policy.policy_version, count: match.identity_suffixes.length, warnings: match.warnings || [] };
}

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
export function issueLeaseKey(issue: number, language: string): string { return `euroeval:worker:lease:${issue}:${language}`; }
export async function acquireIssueMutex(issue: number): Promise<string | null> {
  const token = randomToken(12);
  return await redisSet(`euroeval:worker:mutex:${issue}`, token, 30, true) ? token : null;
}
export async function releaseIssueMutex(issue: number, token: string): Promise<void> {
  await redis("EVAL", "if redis.call('GET',KEYS[1]) == ARGV[1] then return redis.call('DEL',KEYS[1]) else return 0 end", "1", `euroeval:worker:mutex:${issue}`, token);
}
export function leaseKey(leaseId: string): string { return `euroeval:worker:lease-id:${leaseId}`; }
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

export function validateRecord(record: unknown, expected: { modelId: string; revision: string; language: string; euroevalVersion?: string }): { identity: string; failed: number } {
  if (!record || typeof record !== "object" || Array.isArray(record)) throw new BrokerError(422, "record must be an EEE JSON object.");
  const value = record as Record<string, any>;
  const modelInfo = value.model_info as Record<string, any> | undefined;
  const library = value.eval_library as Record<string, any> | undefined;
  if (value.schema_version !== "0.2.1" || !modelInfo || !library || !Array.isArray(value.evaluation_results)) throw new BrokerError(422, "record is not a supported EEE record.");
  if (modelInfo.id !== expected.modelId) throw new BrokerError(422, `record model_info.id must be ${expected.modelId}.`);
  if (modelInfo.revision !== undefined && modelInfo.revision !== expected.revision) throw new BrokerError(422, "record model_info.revision does not match the lease.");
  if (library.name !== "euroeval" || typeof library.version !== "string" || !library.version || expected.euroevalVersion && library.version.replace(/\.dev\d+$/, "") !== expected.euroevalVersion.replace(/\.dev\d+$/, "")) throw new BrokerError(422, "record has an invalid eval_library name or version.");
  const details = library.additional_details as Record<string, any> | undefined;
  if (!details || typeof details.dataset !== "string" || !details.dataset || typeof details.task !== "string" || !details.task) throw new BrokerError(422, "record has no canonical dataset/task identity.");
  let raw: unknown;
  try { raw = typeof details.raw_results === "string" ? JSON.parse(details.raw_results) : details.raw_results; } catch { throw new BrokerError(422, "additional_details.raw_results is not valid JSON."); }
  if (!Array.isArray(raw)) throw new BrokerError(422, "additional_details.raw_results is not valid JSON.");
  const failedValue = (item: unknown): number => typeof item === "number" ? item : typeof item === "string" && item.trim() ? Number(item) : 0;
  const failedIndicator = (item: unknown): boolean => {
    if (Array.isArray(item)) return item.length !== 0;
    if (typeof item === "string" && (item.trim().startsWith("[") || item.trim().startsWith("{"))) {
      try { const parsed = JSON.parse(item) as unknown; return Array.isArray(parsed) ? parsed.length !== 0 : !!parsed; } catch { return true; }
    }
    return !Number.isFinite(failedValue(item)) || failedValue(item) !== 0;
  };
  const containsFailure = (item: unknown): boolean => {
    if (Array.isArray(item)) return item.some(containsFailure);
    if (!item || typeof item !== "object") return false;
    return Object.entries(item as Record<string, unknown>).some(([key, value]) =>
      key === "num_failed_instances" || key === "failed_instances" ? failedIndicator(value) : containsFailure(value));
  };
  if (containsFailure(raw)) throw new BrokerError(422, "records with failed instances cannot be submitted.");
  const group = languageGroup(expected.language);
  let languages: unknown;
  try { languages = typeof details.languages === "string" ? JSON.parse(details.languages) : details.languages; } catch { throw new BrokerError(422, "additional_details.languages is not valid JSON."); }
  if (!Array.isArray(languages) || !languages.length || !languages.every((item) => typeof item === "string" && group !== null && GROUPS[group].includes(item))) throw new BrokerError(422, "record languages must be a non-empty subset of the leased language group.");
  const recordLanguage = details.language ?? value.language;
  if (recordLanguage !== undefined && (typeof recordLanguage !== "string" || !languages.includes(recordLanguage))) throw new BrokerError(422, "record language is inconsistent with languages.");
  const normaliseBool = (item: unknown): boolean | null | undefined => item === null || item === undefined ? null : item === true || item === "true" ? true : item === false || item === "false" ? false : undefined;
  const split = normaliseBool(details.validation_split); const shot = normaliseBool(details.few_shot);
  if (split === undefined || shot === undefined) throw new BrokerError(422, "validation_split and few_shot must be boolean or null.");
  for (const item of [value.num_failed_instances, value.failed_instances, details.num_failed_instances, details.failed_instances]) if (item !== undefined && failedIndicator(item)) throw new BrokerError(422, "records with failed instances cannot be submitted.");
  for (const result of value.evaluation_results) {
    if (!result || typeof result !== "object") throw new BrokerError(422, "evaluation_results contains an invalid item.");
    const evaluation = result as Record<string, any>; const scoreDetails = evaluation.score_details as Record<string, any>;
    if (typeof evaluation.evaluation_name !== "string" || !evaluation.evaluation_name || !scoreDetails || typeof scoreDetails !== "object") throw new BrokerError(422, "every evaluation result needs a metric and score details.");
    const source = evaluation.source_data as Record<string, any> | undefined;
    if (!source || source.dataset_name !== details.dataset) throw new BrokerError(422, "evaluation source_data is inconsistent with the record dataset.");
    const metric = evaluation.metric_config as Record<string, any> | undefined;
    if (!metric || typeof metric.lower_is_better !== "boolean" || metric.score_type !== undefined && typeof metric.score_type !== "string") throw new BrokerError(422, "every evaluation result needs a valid metric config.");
    for (const bound of [metric.min_score, metric.max_score]) if (bound !== undefined && (typeof bound !== "number" || !Number.isFinite(bound))) throw new BrokerError(422, "metric bounds must be finite numbers.");
    const score = scoreDetails.score;
    if (typeof score !== "number" || !Number.isFinite(score)) throw new BrokerError(422, "every evaluation score must be finite.");
    if (typeof metric.min_score === "number" && score < metric.min_score || typeof metric.max_score === "number" && score > metric.max_score) throw new BrokerError(422, "evaluation score violates its declared metric bounds.");
    const uncertainty = scoreDetails.uncertainty as Record<string, any> | undefined;
    const interval = uncertainty?.confidence_interval as Record<string, any> | undefined;
    if (uncertainty && typeof uncertainty !== "object") throw new BrokerError(422, "evaluation uncertainty is invalid.");
    if (uncertainty && Object.values(uncertainty).some((item) => typeof item === "number" && !Number.isFinite(item))) throw new BrokerError(422, "evaluation uncertainty is invalid.");
    if (interval && (typeof interval.lower !== "number" || typeof interval.upper !== "number" || !Number.isFinite(interval.lower) || !Number.isFinite(interval.upper) || interval.lower > interval.upper || score < interval.lower || score > interval.upper)) throw new BrokerError(422, "evaluation uncertainty is invalid.");
    const resultDetails = scoreDetails.details as Record<string, any> | undefined;
    if (resultDetails && ((resultDetails.num_failed_instances !== undefined && failedValue(resultDetails.num_failed_instances) !== 0) || Array.isArray(resultDetails.failed_instances) && resultDetails.failed_instances.length)) throw new BrokerError(422, "records with failed instances cannot be submitted.");
  }
  if (!value.evaluation_results.length) throw new BrokerError(422, "record contains no results.");
  return { identity: JSON.stringify([modelInfo.id, details.dataset, split, shot]), failed: 0 };
}

let stagingMetadata: { bucket: string; private: boolean; checkedAt: number } | null = null;
export async function uploadStaging(path: string, content: string): Promise<void> {
  const bucket = env("HF_STAGING_BUCKET"); const token = env("HF_TOKEN");
  if (!/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(bucket)) throw new ConfigurationError("HF_STAGING_BUCKET must be a namespace/bucket id.");
  if (!stagingMetadata || stagingMetadata.bucket !== bucket || Date.now() - stagingMetadata.checkedAt > 60_000) {
    const info = await fetchWithRetry(`https://huggingface.co/api/buckets/${bucket}`, { headers: { accept: "application/json", authorization: `Bearer ${token}` } });
    if (!info.ok) throw new BrokerError(502, `Hugging Face bucket lookup failed with HTTP ${info.status}.`);
    const metadata = await info.json() as { private?: boolean };
    stagingMetadata = { bucket, private: metadata.private === true, checkedAt: Date.now() };
  }
  if (!stagingMetadata.private) throw new ConfigurationError("HF_STAGING_BUCKET is not private; refusing to upload worker results.");
  try {
    await uploadFile({ repo: `buckets/${bucket}`, file: { path, content: new Blob([content], { type: "application/json" }) }, accessToken: token, useXet: true });
  } catch (error) { throw new BrokerError(502, `Hugging Face bucket upload failed: ${error instanceof Error ? error.message : "unknown error"}`); }
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`).join(",")}}`;
  }
  return JSON.stringify(value);
}
export async function recordDigest(record: unknown): Promise<string> {
  return sha256(canonicalJson(record));
}
export function contributorLabel(login: string): string { return login === "saattrupdan" ? "anonymous worker" : login; }
