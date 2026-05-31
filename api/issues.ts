import { detectGguf } from "../src/frontend/services/huggingface";
import { extractModelId } from "../src/frontend/services/github";

export const config = { runtime: "edge" };

declare const process: { env: Record<string, string | undefined> };

const REPO = "EuroEval/EuroEval";
const LABEL = "model evaluation request";
const GGUF_LABEL = "gguf";
const NOT_GGUF_LABEL = "not-gguf";

interface IssueLike {
  number: number;
  title?: string;
  body?: string | null;
  labels?: Array<{ name?: string } | string>;
}

function labelNames(issue: IssueLike): string[] {
  return (issue.labels ?? []).map((l) => (typeof l === "string" ? l : l.name ?? ""));
}

async function addLabel(
  number: number,
  label: string,
  token: string,
): Promise<boolean> {
  const r = await fetch(
    `https://api.github.com/repos/${REPO}/issues/${number}/labels`,
    {
      method: "POST",
      headers: {
        authorization: `Bearer ${token}`,
        accept: "application/vnd.github+json",
        "x-github-api-version": "2022-11-28",
        "content-type": "application/json",
      },
      body: JSON.stringify({ labels: [label] }),
    },
  );
  return r.ok;
}

// Tag any issue not yet carrying a gguf/not-gguf label with the result of an
// accurate, metadata-based Hub check. The label is the cache: each issue is
// looked up at most once, and the frontend hides the gguf-labelled ones. A
// transient Hub failure makes detectGguf return false (→ not-gguf), so we err
// towards showing the model — the same behaviour as before any check existed,
// and the queue processor still refuses to actually run a GGUF model.
async function labelGgufIssues(
  issues: IssueLike[],
  token: string,
): Promise<void> {
  await Promise.all(
    issues.map(async (issue) => {
      const names = labelNames(issue);
      if (names.includes(GGUF_LABEL) || names.includes(NOT_GGUF_LABEL)) return;
      const modelId = extractModelId(issue.title ?? "", issue.body ?? null);
      if (!modelId) return;
      const baseId = modelId.includes(":") ? modelId.split(":")[0] : modelId;
      const label = (await detectGguf(baseId)) ? GGUF_LABEL : NOT_GGUF_LABEL;
      if (await addLabel(issue.number, label, token)) {
        issue.labels = [...(issue.labels ?? []), { name: label }];
      }
    }),
  );
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
        "public, max-age=30, s-maxage=30, stale-while-revalidate=300",
      ...(extra ?? {}),
    },
  });
}

export default async function handler(req: Request): Promise<Response> {
  if (req.method === "OPTIONS") return json(204, "");
  if (req.method !== "GET") return json(405, { error: "Method not allowed" });

  const url = new URL(req.url);
  const state = url.searchParams.get("state") ?? "open";
  const page = url.searchParams.get("page") ?? "1";
  if (state !== "open" && state !== "closed") {
    return json(400, { error: "state must be 'open' or 'closed'." });
  }
  if (!/^\d+$/.test(page) || Number(page) < 1 || Number(page) > 50) {
    return json(400, { error: "page must be an integer between 1 and 50." });
  }

  const token = process.env.GITHUB_TOKEN;
  const headers: Record<string, string> = {
    accept: "application/vnd.github+json",
    "x-github-api-version": "2022-11-28",
  };
  if (token) headers.authorization = `Bearer ${token}`;

  const ghUrl =
    `https://api.github.com/repos/${REPO}/issues` +
    `?state=${state}&per_page=100&page=${page}` +
    `&labels=${encodeURIComponent(LABEL)}`;
  const r = await fetch(ghUrl, { headers });
  if (!r.ok) return json(r.status, await r.text());

  const issues = (await r.json()) as IssueLike[];
  // Only the open queue is filtered on GGUF status, so only backfill labels
  // there; labelling needs the write-scoped token submit-evaluation also uses.
  if (state === "open" && token) await labelGgufIssues(issues, token);
  return json(200, issues);
}
