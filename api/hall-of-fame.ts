export const config = { runtime: "edge" };

declare const process: { env: Record<string, string | undefined> };

const REPO = "EuroEval/EuroEval";
const LABEL = "model evaluation request";
const TITLE_PREFIX = "[MODEL EVALUATION REQUEST]";
const EXCLUDE = new Set(["saattrupdan"]);
const MAX_PAGES = 10;
const PER_PAGE = 100;

interface RawAssignee {
  login: string;
  avatar_url: string;
}

interface RawIssue {
  title: string;
  body: string | null;
  assignee: RawAssignee | null;
  assignees: RawAssignee[];
}

interface EvaluatorCount {
  login: string;
  count: number;
  avatarUrl: string;
}

const MODEL_ID_BODY_RE = /(?:^|\n)#{1,6}\s*Model ID\s*\n+([^\n]+)/i;

function extractModelId(title: string, body: string | null): string | null {
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

function json(status: number, body: unknown, extra?: HeadersInit): Response {
  return new Response(typeof body === "string" ? body : JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json",
      "access-control-allow-origin": "*",
      "access-control-allow-methods": "GET, OPTIONS",
      "access-control-allow-headers": "content-type",
      "cache-control":
        "public, max-age=300, s-maxage=3600, stale-while-revalidate=86400",
      ...(extra ?? {}),
    },
  });
}

async function fetchPage(
  page: number,
  headers: Record<string, string>,
): Promise<RawIssue[]> {
  const ghUrl =
    `https://api.github.com/repos/${REPO}/issues` +
    `?state=closed&per_page=${PER_PAGE}&page=${page}` +
    `&labels=${encodeURIComponent(LABEL)}`;
  let lastErr: unknown = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const r = await fetch(ghUrl, { headers });
      if (r.ok) return (await r.json()) as RawIssue[];
      if (r.status >= 500 || r.status === 429) {
        lastErr = new Error(`GitHub ${r.status}`);
      } else {
        throw new Error(`GitHub ${r.status}: ${await r.text()}`);
      }
    } catch (e) {
      lastErr = e;
    }
    await new Promise((res) => setTimeout(res, 200 * (attempt + 1)));
  }
  throw lastErr instanceof Error ? lastErr : new Error("GitHub fetch failed");
}

export default async function handler(req: Request): Promise<Response> {
  if (req.method === "OPTIONS") return json(204, "");
  if (req.method !== "GET") return json(405, { error: "Method not allowed" });

  const token = process.env.GITHUB_TOKEN;
  const headers: Record<string, string> = {
    accept: "application/vnd.github+json",
    "x-github-api-version": "2022-11-28",
  };
  if (token) headers.authorization = `Bearer ${token}`;

  const counts = new Map<string, EvaluatorCount>();
  try {
    for (let page = 1; page <= MAX_PAGES; page++) {
      const chunk = await fetchPage(page, headers);
      for (const issue of chunk) {
        if (!extractModelId(issue.title, issue.body)) continue;
        const assignees =
          issue.assignees && issue.assignees.length > 0
            ? issue.assignees
            : issue.assignee
              ? [issue.assignee]
              : [];
        const seen = new Set<string>();
        for (const a of assignees) {
          if (EXCLUDE.has(a.login)) continue;
          if (seen.has(a.login)) continue;
          seen.add(a.login);
          const cur = counts.get(a.login);
          if (cur) {
            cur.count += 1;
          } else {
            counts.set(a.login, {
              login: a.login,
              count: 1,
              avatarUrl: a.avatar_url,
            });
          }
        }
      }
      if (chunk.length < PER_PAGE) break;
    }
  } catch (e) {
    return json(502, { error: (e as Error).message });
  }

  const result = Array.from(counts.values()).sort((a, b) => b.count - a.count);
  return json(200, result);
}
