import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";

export const config = { runtime: "edge" };

const REPO = "EuroEval/EuroEval";
const LABEL = "model evaluation request";
const TITLE_PREFIX = "[MODEL EVALUATION REQUEST]";

const LANGUAGE_GROUPS = [
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

const MODEL_ID_RE = /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/;

const ratelimit = new Ratelimit({
  redis: Redis.fromEnv(),
  limiter: Ratelimit.slidingWindow(5, "1 h"),
  analytics: false,
  prefix: "euroeval:submit",
});

function json(status: number, body: unknown, extra?: HeadersInit): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json",
      "access-control-allow-origin": "*",
      "access-control-allow-methods": "POST, OPTIONS",
      "access-control-allow-headers": "content-type",
      ...(extra ?? {}),
    },
  });
}

function buildIssueBody(modelId: string, groups: string[]): string {
  const checkboxes = LANGUAGE_GROUPS.map(
    (g) => `- [${groups.includes(g) ? "X" : " "}] ${g}`,
  ).join("\n");
  return `### Model ID\n\n${modelId}\n\n### Evaluation languages\n\n${checkboxes}\n`;
}

async function fetchEvaluatedGroups(
  reqUrl: string,
  modelId: string,
): Promise<string[]> {
  const url = new URL("/evaluated-models.json", reqUrl).toString();
  try {
    const r = await fetch(url);
    if (!r.ok) return [];
    const data = (await r.json()) as Record<string, string[]>;
    return data[modelId] ?? [];
  } catch {
    return [];
  }
}

async function huggingFaceModelExists(modelId: string): Promise<boolean> {
  const r = await fetch(
    `https://huggingface.co/api/models/${encodeURIComponent(modelId)}`,
    { method: "GET" },
  );
  if (r.status === 200) {
    const data = (await r.json()) as { private?: boolean; gated?: boolean | string };
    if (data.private) return false;
    if (data.gated && data.gated !== false) return false;
    return true;
  }
  return false;
}

async function findExistingOpenIssue(
  modelId: string,
  token: string,
): Promise<string | null> {
  const q = `repo:${REPO} is:issue is:open label:"${LABEL}" in:title "${TITLE_PREFIX} ${modelId}"`;
  const r = await fetch(
    `https://api.github.com/search/issues?q=${encodeURIComponent(q)}`,
    {
      headers: {
        authorization: `Bearer ${token}`,
        accept: "application/vnd.github+json",
        "x-github-api-version": "2022-11-28",
      },
    },
  );
  if (!r.ok) return null;
  const data = (await r.json()) as {
    items?: Array<{ html_url: string; title: string }>;
  };
  const exactTitle = `${TITLE_PREFIX} ${modelId}`;
  const hit = (data.items ?? []).find((i) => i.title === exactTitle);
  return hit?.html_url ?? null;
}

async function createIssue(
  modelId: string,
  groups: string[],
  token: string,
): Promise<{ url: string; number: number } | null> {
  const r = await fetch(`https://api.github.com/repos/${REPO}/issues`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${token}`,
      accept: "application/vnd.github+json",
      "x-github-api-version": "2022-11-28",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      title: `${TITLE_PREFIX} ${modelId}`,
      body: buildIssueBody(modelId, groups),
      labels: [LABEL],
    }),
  });
  if (!r.ok) return null;
  const data = (await r.json()) as { html_url: string; number: number };
  return { url: data.html_url, number: data.number };
}

export default async function handler(req: Request): Promise<Response> {
  if (req.method === "OPTIONS") return json(204, "");
  if (req.method !== "POST") return json(405, { error: "Method not allowed" });

  const token = process.env.GITHUB_TOKEN;
  if (!token) return json(500, { error: "Server is missing GITHUB_TOKEN" });

  const ip =
    req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ||
    req.headers.get("x-real-ip") ||
    "unknown";

  const { success, reset } = await ratelimit.limit(ip);
  if (!success) {
    const retryAfter = Math.max(1, Math.ceil((reset - Date.now()) / 1000));
    return json(
      429,
      { error: "Too many submissions. Try again later." },
      { "retry-after": String(retryAfter) },
    );
  }

  let payload: { model_id?: unknown; language_groups?: unknown };
  try {
    payload = (await req.json()) as typeof payload;
  } catch {
    return json(400, { error: "Invalid JSON" });
  }

  const modelId = payload.model_id;
  const groups = payload.language_groups;

  if (typeof modelId !== "string" || !MODEL_ID_RE.test(modelId)) {
    return json(400, {
      error: "model_id must look like 'org/name' and use only [A-Za-z0-9._-].",
    });
  }
  if (
    !Array.isArray(groups) ||
    groups.length === 0 ||
    !groups.every(
      (g): g is string =>
        typeof g === "string" &&
        (LANGUAGE_GROUPS as readonly string[]).includes(g),
    )
  ) {
    return json(400, {
      error: "language_groups must be a non-empty list of known language groups.",
    });
  }

  if (!(await huggingFaceModelExists(modelId))) {
    return json(422, {
      error:
        "Model not found on the Hugging Face Hub, or is private/gated. Public models only.",
    });
  }

  const existing = await findExistingOpenIssue(modelId, token);
  if (existing) {
    return json(409, {
      error: "This model is already in the evaluation queue.",
      url: existing,
    });
  }

  const alreadyEvaluated = await fetchEvaluatedGroups(req.url, modelId);
  const remaining = (groups as string[]).filter(
    (g) => !alreadyEvaluated.includes(g),
  );
  if (remaining.length === 0) {
    return json(409, {
      error:
        "This model has already been evaluated on every language group you" +
        " requested. See the leaderboards.",
    });
  }

  const created = await createIssue(modelId, remaining, token);
  if (!created) {
    return json(502, { error: "Failed to create the GitHub issue." });
  }
  return json(200, created);
}
