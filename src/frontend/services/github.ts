export const REPO = "EuroEval/EuroEval";
export const LABEL = "model evaluation request";
export const FAILED_LABEL = "evaluation-failed";
export const GATED_LABEL = "gated";
export const RESULTS_READY_LABEL = "results-ready";
export const GGUF_LABEL = "gguf";
export const TITLE_PREFIX = "[MODEL EVALUATION REQUEST]";

export const LANGUAGE_GROUPS = [
  "Baltic languages (Latvian, Lithuanian)",
  "Finnic languages (Estonian, Finnish)",
  "Romance languages (Catalan, French, Italian, Portuguese, Romanian, Spanish)",
  "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)",
  "Slavic languages (Belarusian, Bulgarian, Bosnian, Croatian, Czech, Polish, Serbian, Slovak, Slovenian, Ukrainian)",
  "West Germanic languages (Dutch, English, German)",
  "Albanian",
  "Greek",
  "Hungarian",
] as const;

export type LanguageGroup = (typeof LANGUAGE_GROUPS)[number];

export interface RawIssue {
  number: number;
  title: string;
  html_url: string;
  body: string | null;
  assignee: { login: string } | null;
  assignees: Array<{ login: string }>;
  created_at: string;
  labels: Array<{ name: string } | string>;
}

export type QueueStatus =
  | "Awaiting publish"
  | "Evaluating"
  | "Waiting"
  | "Gated model"
  | "Error";

export interface QueueEntry {
  number: number;
  url: string;
  modelId: string;
  languageGroups: string[];
  status: QueueStatus;
  evaluator: string | null;
  createdAt: string;
}





export function hasLabel(issue: RawIssue, name: string): boolean {
  return issue.labels.some((l) => (typeof l === "string" ? l : l.name) === name);
}

const MODEL_ID_BODY_RE = /(?:^|\n)#{1,6}\s*Model ID\s*\n+([^\n]+)/i;

export function extractModelId(
  title: string,
  body: string | null = null,
): string | null {
  if (body) {
    const m = body.match(MODEL_ID_BODY_RE);
    if (m) {
      const id = m[1].trim().replace(/^[`*_]+|[`*_]+$/g, "").trim();
      if (id && id !== "<model-name>") return id;
    }
  }
  const prefix = `${TITLE_PREFIX} `;
  if (!title.startsWith(prefix)) return null;
  const rest = title.slice(prefix.length).trim();
  return rest && rest !== "<model-name>" ? rest : null;
}

export function extractLanguageGroups(body: string | null): string[] {
  if (!body) return [];
  return LANGUAGE_GROUPS.filter((g) => {
    const escaped = g.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`-\\s*\\[[xX]\\]\\s*${escaped}`);
    return re.test(body);
  });
}

export function issueStatus(
  issue: RawIssue,
  failed: boolean,
  gated: boolean,
  resultsReady: boolean,
): QueueStatus {
  if (resultsReady) return "Awaiting publish";
  if (issue.assignee || issue.assignees.length > 0) return "Evaluating";
  if (gated) return "Gated model";
  if (failed) return "Error";
  return "Waiting";
}

export function toQueueEntry(issue: RawIssue): QueueEntry | null {
  const modelId = extractModelId(issue.title, issue.body);
  if (!modelId) return null;
  // The queue can't run GGUF models. We rely on the GGUF_LABEL applied by the
  // /api/issues refresh path (an accurate, metadata-based Hub check, cached as
  // a label so it runs once per issue) rather than guessing from the model id
  // — many GGUF repos carry no "GGUF" token in their name.
  if (hasLabel(issue, GGUF_LABEL)) return null;
  const failed = hasLabel(issue, FAILED_LABEL);
  const resultsReady = hasLabel(issue, RESULTS_READY_LABEL);
  const gated = hasLabel(issue, GATED_LABEL);
  const evaluator =
    issue.assignee?.login ?? issue.assignees[0]?.login ?? null;
  return {
    number: issue.number,
    url: issue.html_url,
    modelId,
    languageGroups: extractLanguageGroups(issue.body),
    status: issueStatus(issue, failed, gated, resultsReady),
    evaluator,
    createdAt: issue.created_at,
  };
}

export async function listOpenEvalIssues(
  { fresh = false }: { fresh?: boolean } = {},
): Promise<QueueEntry[]> {
  // The default request shares a stable URL so Vercel's edge cache (s-maxage +
  // stale-while-revalidate in api/issues.ts) can serve it without re-hitting
  // GitHub. `fresh: true` bypasses both the browser and edge caches via a
  // cache-buster + no-store — needed right after submitting a new issue, where
  // a 30s-stale list would miss the just-created entry.
  const buster = fresh ? `&_=${Date.now()}` : "";
  const init = fresh ? { cache: "no-store" as const } : undefined;
  const raw: RawIssue[] = [];
  const maxPages = 10;
  for (let page = 1; page <= maxPages; page++) {
    const r = await fetch(`/api/issues?state=open&page=${page}${buster}`, init);
    if (!r.ok) {
      throw new Error(`Failed to load queue (${r.status}).`);
    }
    const chunk = (await r.json()) as RawIssue[];
    raw.push(...chunk);
    if (chunk.length < 100) break;
  }
  const order: Record<QueueStatus, number> = {
    "Awaiting publish": 0,
    Evaluating: 1,
    Waiting: 2,
    "Gated model": 3,
    Error: 4,
  };
  return raw
    .map(toQueueEntry)
    .filter((e): e is QueueEntry => e !== null)
    .sort((a, b) => {
      if (a.status !== b.status) return order[a.status] - order[b.status];
      return b.createdAt.localeCompare(a.createdAt);
    });
}

export interface EvaluatorCount {
  login: string;
  count: number;
  avatarUrl: string;
}

export async function listEvaluators(): Promise<EvaluatorCount[]> {
  const r = await fetch("/api/hall-of-fame");
  if (!r.ok) {
    throw new Error(`Failed to load hall of fame (${r.status}).`);
  }
  return (await r.json()) as EvaluatorCount[];
}

export interface SubmitResult {
  ok: boolean;
  status: number;
  url?: string;
  number?: number;
  error?: string;
}

export async function submitEvalRequest(
  modelId: string,
  languageGroups: string[],
): Promise<SubmitResult> {
  const r = await fetch("/api/submit-evaluation", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model_id: modelId, language_groups: languageGroups }),
  });
  const data = (await r.json().catch(() => ({}))) as {
    url?: string;
    number?: number;
    error?: string;
  };
  return { ok: r.ok, status: r.status, ...data };
}
