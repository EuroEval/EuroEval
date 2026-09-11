/* Shared broker implementation. This is the volunteer-worker/v1 JSON protocol. */
declare const process: { env: Record<string, string | undefined> };
export const PROTOCOL_VERSION = "volunteer-worker/v1" as const;
export const REPO = "EuroEval/EuroEval";
const MARKER_VERSION = 1 as const;
const MARKER_DOMAIN = "euroeval-volunteer-marker";
const CREDIT_DOMAIN = "euroeval-volunteer-credit";
export const REQUEST_LABEL = "model evaluation request";
export const TITLE_PREFIX = "[MODEL EVALUATION REQUEST]";
export const VOLUNTEER_MARKER_RE =
  /<!--[ \t]*euroeval-volunteer-worker:v1[ \t]+([^<]*?)-->/i;
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
  verified_contributor: string;
  result_count: number;
  status: "submitted" | "accepted" | "rejected";
}

export interface PromotionRecord {
  identity: string;
  canonical_path: string;
  digest: string;
}

/**
 * Broker-owned promotion state. Decision metadata is selected by the broker on
 * the first reservation and remains immutable for retries.
 */
export interface PromotionReservation {
  issue_number: number;
  submission_id: string;
  outcome: "accepted" | "rejected";
  records: PromotionRecord[];
  token: string;
  decision_reviewer: string;
  decision_created_at: string;
  /** SHA256 of the exact durable decision artifact, bound once by the broker. */
  decision_digest?: string;
  /** Legacy metadata retained only when reading pre-migration reservations. */
  decision_nonce?: string;
  status: "reserved" | "terminal";
}

export interface VolunteerLeaseMarker {
  protocol_version: typeof PROTOCOL_VERSION;
  coordinator: string;
  submission: "active" | "submitted" | "accepted" | "rejected";
  leases: Array<{ lease_id: string; language: string; worker: string; contributor: string; expires_at: string }>;
  submissions?: VolunteerSubmission[];
  completed_languages?: string[];
  signature?: string;
}

export interface FinalCredit {
  version: typeof MARKER_VERSION;
  immutable: true;
  winner: string;
  accepted_counts: Array<{ login: string; count: number }>;
  completed_languages: string[];
  /** Server-bound metadata for the immutable decision artifact. */
  decision_reviewer?: string;
  decision_created_at?: string;
  /** Legacy field retained so old signed credits remain verifiable. */
  decision_nonce?: string;
  signature: string;
}

export interface WorkerIdentity {
  hash: string;
  contributor: string;
}

export interface ModelMetadataEvidence {
  pipeline_tag: string;
  architectures: string[];
  model_type: "encoder" | "generative";
  is_encoder_decoder: boolean | null;
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
  gpu_memory_utilisation: number;
  selected_gpu_index: number;
  selected_gpu_uuid: string;
  expires_at: string;
  lease_id: string;
  result_count?: number;
  model_type: "encoder" | "generative";
  model_metadata: ModelMetadataEvidence;
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
      "access-control-allow-headers": "content-type, authorization, x-coordinator-secret, x-promotion-secret",
      ...(status === 429 ? { "retry-after": "60" } : {}),
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

async function hmac(secret: string, value: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"],
  );
  return bytesToBase64(new Uint8Array(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(value))));
}

function markerPayload(issueNumber: number, marker: VolunteerLeaseMarker): string {
  const { signature: _signature, ...unsigned } = marker;
  return canonicalJson({ domain: MARKER_DOMAIN, version: MARKER_VERSION, issue_number: issueNumber, marker: unsigned });
}

function creditPayload(issueNumber: number, credit: Omit<FinalCredit, "signature">): string {
  return canonicalJson({ domain: CREDIT_DOMAIN, version: MARKER_VERSION, issue_number: issueNumber, credit });
}

export function markerSecret(): string {
  return env("VOLUNTEER_MARKER_SECRET");
}

export async function signVolunteerMarker(issueNumber: number, marker: VolunteerLeaseMarker): Promise<VolunteerLeaseMarker> {
  return { ...marker, signature: await hmac(markerSecret(), markerPayload(issueNumber, marker)) };
}

export async function verifyVolunteerMarker(issueNumber: number, marker: VolunteerLeaseMarker): Promise<boolean> {
  return !!marker.signature && marker.signature === await hmac(markerSecret(), markerPayload(issueNumber, marker));
}

export async function signFinalCredit(issueNumber: number, credit: Omit<FinalCredit, "signature">): Promise<FinalCredit> {
  return { ...credit, signature: await hmac(markerSecret(), creditPayload(issueNumber, credit)) };
}

export async function verifyFinalCredit(issueNumber: number, credit: FinalCredit): Promise<boolean> {
  const { signature: _signature, ...unsigned } = credit;
  return !!credit.signature && credit.signature === await hmac(markerSecret(), creditPayload(issueNumber, unsigned));
}

export function parseFinalCredit(body: string | null): FinalCredit | null {
  if (!body) return null;
  const starts = [...body.matchAll(/<!--[ \t]*euroeval-volunteer-credit:v1[ \t]+/gi)];
  if (starts.length !== 1) return null;
  const start = starts[0].index! + starts[0][0].length;
  const end = body.indexOf("-->", start);
  if (end < 0) return null;
  try {
    const value = JSON.parse(body.slice(start, end).trim()) as FinalCredit;
    if (value.version !== MARKER_VERSION || value.immutable !== true || typeof value.winner !== "string" ||
        !Array.isArray(value.accepted_counts) || !Array.isArray(value.completed_languages) ||
        value.decision_reviewer !== undefined &&
          (typeof value.decision_reviewer !== "string" || !value.decision_reviewer) ||
        value.decision_created_at !== undefined &&
          (typeof value.decision_created_at !== "string" ||
            !value.decision_created_at || Number.isNaN(Date.parse(value.decision_created_at))) ||
        value.decision_reviewer !== undefined !== (value.decision_created_at !== undefined) ||
        value.decision_nonce !== undefined && (typeof value.decision_nonce !== "string" || !value.decision_nonce) ||
        typeof value.signature !== "string") return null;
    return value;
  } catch { return null; }
}

export function removeFinalCredit(body: string): string {
  return body.replace(/<!--[ \t]*euroeval-volunteer-credit:v1[\s\S]*?-->/gi, "").replace(/\n{3,}/g, "\n\n").trimEnd();
}

export function canonicalJson(value: unknown): string {
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
