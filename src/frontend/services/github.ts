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

export type QueueStatus = "Evaluating" | "Waiting";

export interface QueueEntry {
  number: number;
  url: string;
  modelId: string;
  languageGroups: string[];
  status: QueueStatus;
  createdAt: string;
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

export function issueStatus(issue: RawIssue): QueueStatus {
  return issue.assignee || issue.assignees.length > 0 ? "Evaluating" : "Waiting";
}

export function toQueueEntry(issue: RawIssue): QueueEntry | null {
  const modelId = extractModelId(issue.title);
  if (!modelId) return null;
  return {
    number: issue.number,
    url: issue.html_url,
    modelId,
    languageGroups: extractLanguageGroups(issue.body),
    status: issueStatus(issue),
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
  return raw
    .map(toQueueEntry)
    .filter((e): e is QueueEntry => e !== null)
    .sort((a, b) => {
      if (a.status !== b.status) return a.status === "Evaluating" ? -1 : 1;
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
