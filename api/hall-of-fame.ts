declare const process: { env: Record<string, string | undefined> };

export const config = { runtime: "edge" };
const REPO = "EuroEval/EuroEval";
const LABEL = "model evaluation request";
const TITLE_PREFIX = "[MODEL EVALUATION REQUEST]";
const EXCLUDE = new Set(["saattrupdan"]);
const MAX_PAGES = 10;
const PER_PAGE = 100;
const CREDIT_MARKER_RE = /<!--[\s\S]*?euroeval-volunteer-credit:v1\s+({[\s\S]*?})\s*-->/i;
const VOLUNTEER_MARKER_CANDIDATE_RE = /<!--[ \t]*euroeval-volunteer-worker:v1/i;
import { redis, sha256 } from "./worker/_lib.ts";

interface RawAssignee { login: string; avatar_url?: string; }
interface RawIssue { title: string; body: string | null; assignee: RawAssignee | null; assignees: RawAssignee[]; }
interface EvaluatorCount { login: string; count: number; avatarUrl: string; }
const MODEL_ID_BODY_RE = /(?:^|\n)#{1,6}\s*Model ID\s*\n+([^\n]+)/i;

function extractModelId(title: string, body: string | null): string | null {
  if (body) { const m = body.match(MODEL_ID_BODY_RE); if (m) { const id = m[1].trim().replace(/^[`*_]+|[`*_]+$/g, "").trim(); if (id && id !== "<model-name>") return id; } }
  const prefix = `${TITLE_PREFIX} `; if (!title.startsWith(prefix)) return null;
  const rest = title.slice(prefix.length).trim(); return rest && rest !== "<model-name>" ? rest : null;
}

/** Pick the highest accepted identity count, with a stable login tie-break. */
export function calculateWinner(accepted: Array<{ github_login: string; identity: string }>): string | null {
  const counts = new Map<string, Set<string>>();
  for (const item of accepted) { if (!item?.github_login || !item.identity) continue; const set = counts.get(item.github_login) || new Set<string>(); set.add(item.identity); counts.set(item.github_login, set); }
  return [...counts.entries()].sort((a, b) => b[1].size - a[1].size || a[0].toLowerCase().localeCompare(b[0].toLowerCase()) || a[0].localeCompare(b[0]))[0]?.[0] || null;
}

function immutableWinner(body: string | null): { login: string; avatarUrl?: string } | null {
  const match = body?.match(CREDIT_MARKER_RE); if (!match) return null;
  try {
    const value = JSON.parse(match[1]) as { winner?: unknown; avatar_url?: unknown; immutable?: unknown };
    return value.immutable === true && typeof value.winner === "string" ? { login: value.winner, avatarUrl: typeof value.avatar_url === "string" ? value.avatar_url : undefined } : null;
  } catch { return null; }
}

export function creditLogins(issue: RawIssue): string[] {
  const winner = immutableWinner(issue.body);
  if (winner && !EXCLUDE.has(winner.login)) return [winner.login];
  // A recognisable volunteer marker is an explicit protocol record, not a
  // maintainer assignment.  Falling back to assignees would credit the Hall of
  // Fame for malformed, pending, or rejected submissions.
  if (VOLUNTEER_MARKER_CANDIDATE_RE.test(issue.body || "")) return [];
  const assignees = issue.assignees && issue.assignees.length > 0 ? issue.assignees : issue.assignee ? [issue.assignee] : [];
  const seen = new Set<string>();
  return assignees.flatMap((assignee) => { if (EXCLUDE.has(assignee.login) || seen.has(assignee.login)) return []; seen.add(assignee.login); return [assignee.login]; });
}

function json(status: number, body: unknown, extra?: HeadersInit): Response { return new Response(status === 204 ? null : typeof body === "string" ? body : JSON.stringify(body), { status, headers: { "content-type": "application/json", "access-control-allow-origin": "*", "access-control-allow-methods": "GET, OPTIONS", "access-control-allow-headers": "content-type", "cache-control": "public, max-age=300, s-maxage=3600, stale-while-revalidate=86400", ...(extra ?? {}) } }); }
async function fetchPage(page: number, headers: Record<string, string>): Promise<RawIssue[]> {
  const url = `https://api.github.com/repos/${REPO}/issues?state=closed&per_page=${PER_PAGE}&page=${page}&labels=${encodeURIComponent(LABEL)}`;
  let lastErr: unknown = null;
  for (let attempt = 0; attempt < 3; attempt++) { try { const response = await fetch(url, { headers }); if (response.ok) return await response.json() as RawIssue[]; if (response.status >= 500 || response.status === 429) lastErr = new Error(`GitHub ${response.status}`); else throw new Error(`GitHub ${response.status}: ${await response.text()}`); } catch (error) { lastErr = error; } await new Promise((resolve) => setTimeout(resolve, 200 * (attempt + 1))); }
  throw lastErr instanceof Error ? lastErr : new Error("GitHub fetch failed");
}
export default async function handler(req: Request): Promise<Response> {
  if (req.method === "OPTIONS") return json(204, ""); if (req.method !== "GET") return json(405, { error: "Method not allowed" });
  if (!(await withinPublicRateLimit(req))) return json(429, { error: "Too many requests." }, { "retry-after": "60" });
  const token = process.env.GITHUB_TOKEN; const headers: Record<string, string> = { accept: "application/vnd.github+json", "x-github-api-version": "2022-11-28" }; if (token) headers.authorization = `Bearer ${token}`;
  const counts = new Map<string, EvaluatorCount>();
  try { for (let page = 1; page <= MAX_PAGES; page++) { const chunk = await fetchPage(page, headers); for (const issue of chunk) { if (!extractModelId(issue.title, issue.body)) continue; for (const login of creditLogins(issue)) {            const credit = immutableWinner(issue.body);
            const avatarUrl = issue.assignees?.find((assignee) => assignee.login === login)?.avatar_url || (issue.assignee?.login === login ? issue.assignee.avatar_url : "") || (credit?.login === login ? credit.avatarUrl : undefined) || `https://github.com/${encodeURIComponent(login)}.png?size=64`; const current = counts.get(login); if (current) current.count += 1; else counts.set(login, { login, count: 1, avatarUrl }); } } if (chunk.length < PER_PAGE) break; } } catch (error) { return json(502, { error: (error as Error).message }); }
  return json(200, Array.from(counts.values()).sort((a, b) => b.count - a.count || a.login.toLowerCase().localeCompare(b.login.toLowerCase())));
}

async function withinPublicRateLimit(req: Request): Promise<boolean> {
  if (!process.env.UPSTASH_REDIS_REST_URL || !process.env.UPSTASH_REDIS_REST_TOKEN) return false;
  const address = req.headers.get("x-forwarded-for")?.split(",", 1)[0]?.trim() || "anonymous";
  const key = `euroeval:public:hall:${await sha256(address)}:${Math.floor(Date.now() / 60000)}`;
  try {
    const count = await redis("INCR", key);
    if (count === 1 || count === "1") await redis("EXPIRE", key, "60");
    return Number(count) <= 60;
  } catch {
    return false;
  }
}
