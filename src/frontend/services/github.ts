export const REPO = "EuroEval/EuroEval";
export const LABEL = "model evaluation request";
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
  | "Evaluating"
  | "Waiting"
  | "Gated model"
  | "Error"
  | "Waiting for bug fix";

export interface QueueEntry {
  number: number;
  url: string;
  modelId: string;
  languageGroups: string[];
  status: QueueStatus;
  evaluator: string | null;
  erroredOnVersion: string | null;
  createdAt: string;
}

declare const __PACKAGE_VERSION__: string | undefined;
const currentVersion =
  typeof __PACKAGE_VERSION__ !== "undefined" ? __PACKAGE_VERSION__ : "";

const ERROR_MARKER_RE = /<!--\s*errored-on:\s*v([^\s>-]+)\s*-->/;
const GATED_MARKER_RE = /<!--\s*gated-model\s*-->/;

export function extractErroredOnVersion(body: string | null): string | null {
  if (!body) return null;
  const m = body.match(ERROR_MARKER_RE);
  return m ? m[1] : null;
}

export function isGatedModel(body: string | null): boolean {
  return !!body && GATED_MARKER_RE.test(body);
}

function versionTuple(v: string): number[] {
  return v
    .split(".")
    .map((p) => parseInt(p, 10))
    .filter((n) => !Number.isNaN(n));
}

/** Return whether ``a`` is strictly newer than ``b`` in dotted-numeric order. */
export function isVersionNewer(a: string, b: string): boolean {
  const ta = versionTuple(a);
  const tb = versionTuple(b);
  const n = Math.max(ta.length, tb.length);
  for (let i = 0; i < n; i++) {
    const av = ta[i] ?? 0;
    const bv = tb[i] ?? 0;
    if (av !== bv) return av > bv;
  }
  return false;
}

export function extractModelId(title: string): string | null {
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
  erroredOn: string | null,
  gated: boolean,
): QueueStatus {
  if (issue.assignee || issue.assignees.length > 0) return "Evaluating";
  if (gated) return "Gated model";
  if (erroredOn) {
    return currentVersion && isVersionNewer(currentVersion, erroredOn)
      ? "Error"
      : "Waiting for bug fix";
  }
  return "Waiting";
}

export function toQueueEntry(issue: RawIssue): QueueEntry | null {
  const modelId = extractModelId(issue.title);
  if (!modelId) return null;
  const erroredOnVersion = extractErroredOnVersion(issue.body);
  const gated = isGatedModel(issue.body);
  const evaluator =
    issue.assignee?.login ?? issue.assignees[0]?.login ?? null;
  return {
    number: issue.number,
    url: issue.html_url,
    modelId,
    languageGroups: extractLanguageGroups(issue.body),
    status: issueStatus(issue, erroredOnVersion, gated),
    evaluator,
    erroredOnVersion,
    createdAt: issue.created_at,
  };
}

export async function listOpenEvalIssues(): Promise<QueueEntry[]> {
  const url = `https://api.github.com/repos/${REPO}/issues?state=open&per_page=100&labels=${encodeURIComponent(LABEL)}`;
  const r = await fetch(url, {
    headers: { accept: "application/vnd.github+json" },
  });
  if (!r.ok) {
    throw new Error(`Failed to load queue (${r.status}).`);
  }
  const raw = (await r.json()) as RawIssue[];
  const order: Record<QueueStatus, number> = {
    Evaluating: 0,
    Waiting: 1,
    "Gated model": 2,
    Error: 3,
    "Waiting for bug fix": 4,
  };
  return raw
    .map(toQueueEntry)
    .filter((e): e is QueueEntry => e !== null)
    .sort((a, b) => {
      if (a.status !== b.status) return order[a.status] - order[b.status];
      return a.createdAt.localeCompare(b.createdAt);
    });
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
