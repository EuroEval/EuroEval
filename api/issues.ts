export const config = { runtime: "edge" };

declare const process: { env: Record<string, string | undefined> };

const REPO = "EuroEval/EuroEval";
const LABEL = "model evaluation request";

function json(status: number, body: unknown, extra?: HeadersInit): Response {
  return new Response(typeof body === "string" ? body : JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json",
      "access-control-allow-origin": "*",
      "access-control-allow-methods": "GET, OPTIONS",
      "access-control-allow-headers": "content-type",
      "cache-control": "public, max-age=30",
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
  const body = await r.text();
  return json(r.status, body);
}
