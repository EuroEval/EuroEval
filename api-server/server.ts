import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { readFile } from "node:fs/promises";

export interface ApiServerOptions {
  allowedOrigins?: string[];
  evaluatedModels: Uint8Array;
}

const workerPaths = [
  "auth/start", "auth/poll", "auth/revoke", "canary-corpus", "claim",
  "coordinator-lock", "coordinator-release", "coordinator-renew", "finalise",
  "heartbeat", "promote", "promotion-lock", "promotion-reserve", "promotion-status",
  "release-info", "release", "result",
];
const routes = new Map([
  ["/api/issues", { file: "issues", method: "GET" }],
  ["/api/submit-evaluation", { file: "submit-evaluation", method: "POST" }],
  ["/api/hall-of-fame", { file: "hall-of-fame", method: "GET" }],
  ...workerPaths.map((name) => [
    `/api/worker/${name}`,
    { file: `worker/${name}`, method: name === "release-info" ? "GET" : "POST" },
  ] as const),
]);
const corsHeaders = new Set([
  "authorization", "content-type", "x-coordinator-secret", "x-promotion-secret",
]);
const strippedHeaders = new Set([
  "host", "connection", "transfer-encoding", "forwarded", "x-real-ip",
  "cf-connecting-ip", "true-client-ip",
]);

/** Adapt the existing Fetch handlers without exposing their helper modules. */
export function createApiServer(options: ApiServerOptions): Server {
  const origins = new Set(options.allowedOrigins ?? ["https://euroeval.com"]);
  for (const origin of origins) {
    const parsed = new URL(origin);
    if (!/^https?:$/.test(parsed.protocol) || parsed.origin !== origin) {
      throw new Error("API_ALLOWED_ORIGINS must contain HTTP(S) origins only");
    }
  }
  const server = createServer((request, response) => {
    void handle(request, response).catch(() => {
      if (response.destroyed) return;
      if (response.headersSent) response.destroy();
      else error(request, response, 500, "Internal server error");
    });
  });
  server.requestTimeout = 30_000;
  server.headersTimeout = 15_000;

  function reply(request: IncomingMessage, response: ServerResponse, result: Response, body?: Uint8Array) {
    const headers = new Headers(result.headers);
    for (const name of [...headers.keys()]) {
      if (name.startsWith("access-control-")) headers.delete(name);
    }
    const vary = new Set((headers.get("vary") ?? "").split(",").map((v) => v.trim()).filter(Boolean));
    vary.add("Origin");
    headers.set("vary", [...vary].join(", "));
    const origin = request.headers.origin;
    if (origin && origins.has(origin)) headers.set("access-control-allow-origin", origin);
    response.writeHead(result.status, Object.fromEntries(headers));
    response.end(request.method === "HEAD" || [204, 304].includes(result.status) ? undefined : body);
  }

  function error(request: IncomingMessage, response: ServerResponse, status: number, message: string, extra?: HeadersInit) {
    const result = Response.json({ error: message }, { status, headers: extra });
    reply(request, response, result, Buffer.from(JSON.stringify({ error: message })));
  }

  async function handle(request: IncomingMessage, response: ServerResponse) {
    const address = request.socket.localAddress;
    const port = request.socket.localPort;
    if (!address || !port) throw new Error("TCP listener required");
    const host = address.includes(":") ? `[${address}]` : address;
    const origin = `http://${host}:${port}`;
    const url = new URL(request.url ?? "/", origin);
    const asset = url.pathname === "/evaluated-models.json";
    const route = routes.get(url.pathname);
    if (!asset && !route) return error(request, response, 404, "Not found");
    const method = asset ? "GET" : route!.method;
    const allowedMethods = `${method}, OPTIONS`;
    if (request.headers.origin && !origins.has(request.headers.origin)) {
      return error(request, response, 403, "Origin not allowed");
    }
    if (request.method === "OPTIONS") {
      const requestedMethod = request.headers["access-control-request-method"];
      const requestedHeaders = String(request.headers["access-control-request-headers"] ?? "")
        .toLowerCase().split(",").map((v) => v.trim()).filter(Boolean);
      if ((requestedMethod && requestedMethod !== method) || requestedHeaders.some((v) => !corsHeaders.has(v))) {
        return error(request, response, 403, "Preflight not allowed");
      }
      response.setHeader("vary", "Origin, Access-Control-Request-Method, Access-Control-Request-Headers");
      if (request.headers.origin) response.setHeader("access-control-allow-origin", request.headers.origin);
      response.writeHead(204, {
        "access-control-allow-methods": allowedMethods,
        "access-control-allow-headers": [...corsHeaders].join(", "),
      });
      response.end();
      return;
    }
    if (request.method !== method) return error(request, response, 405, "Method not allowed", { allow: allowedMethods });
    if (asset) {
      return reply(request, response, new Response(null, {
        headers: { "content-type": "application/json", "cache-control": "public, max-age=300" },
      }), options.evaluatedModels);
    }
    // The handlers retain their smaller contract-specific limits. Bound buffering
    // here as well, including requests with no Content-Length header.
    const limit = 2 * 1024 * 1024;
    if (Number(request.headers["content-length"] ?? 0) > limit) {
      request.resume();
      return error(request, response, 413, "Request body is too large");
    }
    const body = await new Promise<Buffer | null>((resolve, reject) => {
      const chunks: Buffer[] = [];
      let size = 0;
      const cleanup = () => {
        request.removeListener("data", data);
        request.removeListener("end", end);
        request.removeListener("error", failure);
        request.removeListener("aborted", aborted);
      };
      const failure = (cause: Error) => { cleanup(); reject(cause); };
      const aborted = () => failure(new Error("Request aborted"));
      const end = () => { cleanup(); resolve(Buffer.concat(chunks)); };
      const data = (chunk: Buffer) => {
        size += chunk.length;
        if (size > limit) {
          cleanup();
          request.resume();
          resolve(null);
        } else chunks.push(chunk);
      };
      request.on("data", data);
      request.once("end", end);
      request.once("error", failure);
      request.once("aborted", aborted);
    });
    if (body === null) return error(request, response, 413, "Request body is too large");
    if (!request.complete || response.destroyed) return;
    const headers = new Headers();
    for (const [name, value] of Object.entries(request.headers)) {
      if (value === undefined || strippedHeaders.has(name) || name.startsWith("x-forwarded-") || name.startsWith("x-vercel-")) continue;
      headers.set(name, Array.isArray(value) ? value.join(", ") : value);
    }
    headers.set("x-forwarded-for", request.socket.remoteAddress ?? "unknown");
    const abort = new AbortController();
    const disconnect = () => { if (!response.writableEnded) abort.abort(); };
    response.once("close", disconnect);
    try {
      const module = await import(new URL(`../api/${route!.file}.js`, import.meta.url).href);
      const handler = module.default ?? module.fetch;
      const input = new Request(`${origin}${url.pathname}${url.search}`, {
        method, headers, signal: abort.signal,
        ...(method === "POST" ? { body: new Uint8Array(body).buffer } : {}),
      });
      const result: Response = await handler(input);
      const output = new Uint8Array(await result.arrayBuffer());
      if (!response.destroyed) reply(request, response, result, output);
    } finally {
      response.removeListener("close", disconnect);
    }
  }
  return server;
}

/** Load the generated submission index before accepting traffic. */
export async function loadEvaluatedModels(): Promise<Uint8Array> {
  const content = await readFile(new URL("../evaluated-models.json", import.meta.url));
  const parsed: unknown = JSON.parse(content.toString());
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed) ||
      Object.values(parsed).some((groups) => !Array.isArray(groups) || groups.some((group) => typeof group !== "string"))) {
    throw new Error("Invalid evaluated-model index");
  }
  return content;
}
