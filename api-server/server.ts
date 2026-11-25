import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import { readFile } from 'node:fs/promises';

export type ApiServerOptions = { allowedOrigins?: string[]; evaluatedModelsPath?: string; evaluatedModels?: Uint8Array };
const routes = ['/api/issues','/api/submit-evaluation','/api/hall-of-fame', ...['auth/start','auth/poll','auth/revoke','canary-corpus','claim','coordinator-lock','coordinator-release','coordinator-renew','finalise','heartbeat','promote','promotion-lock','promotion-reserve','promotion-status','release-info','release','result'].map(route => `/api/worker/${route}`)];
const workerRoutes = new Set(routes.filter(route => route.startsWith('/api/worker/')));
const maxBody = (route: string) => route === '/api/worker/result' ? 2 * 1024 * 1024 : route.startsWith('/api/worker/') ? 64 * 1024 : 128 * 1024;
function send(res: ServerResponse, status: number, body: string, headers: Record<string,string> = {}) { res.writeHead(status, { 'Content-Type':'application/json; charset=utf-8', ...headers }); res.end(body); }

export function createApiServer(options: ApiServerOptions = {}): Server {
  const allowed = new Set(options.allowedOrigins ?? ['https://euroeval.com']);
  return createServer(async (req: IncomingMessage, res: ServerResponse) => {
    res.setHeader('Vary', 'Origin');
    const origin = req.headers.origin;
    if (origin && allowed.has(origin)) res.setHeader('Access-Control-Allow-Origin', origin);
    const route = new URL(req.url ?? '/', 'http://localhost').pathname;
    const methods = workerRoutes.has(route) ? 'GET, POST, OPTIONS' : route === '/api/hall-of-fame' ? 'GET, OPTIONS' : 'GET, POST, OPTIONS';
    if (req.method === 'OPTIONS') { res.writeHead(204, { 'Access-Control-Allow-Methods': methods, 'Access-Control-Allow-Headers': 'Authorization, Content-Type, X-Worker-Token, X-Protocol-Version' }); res.end(); return; }
    if (!routes.includes(route) && route !== '/api/evaluated-models.json') { send(res,404,'{"error":"Not found"}'); return; }
    if (origin && !allowed.has(origin)) { send(res,403,'{"error":"Origin not allowed"}'); return; }
    if (route === '/api/evaluated-models.json') {
      try { const data = options.evaluatedModels ?? (options.evaluatedModelsPath ? await readFile(options.evaluatedModelsPath) : undefined); if (!data) return send(res,404,'{"error":"Not found"}'); res.writeHead(200, {'Content-Type':'application/json; charset=utf-8','Cache-Control':'public, max-age=300'}); res.end(data); } catch { send(res,404,'{"error":"Not found"}'); } return;
    }
    const permitted = methods.split(', ').filter(method => method !== 'OPTIONS');
    if (!permitted.includes(req.method ?? '')) { res.setHeader('Allow', methods); send(res,405,'{"error":"Method not allowed"}'); return; }
    const limit = maxBody(route); let size = 0; const chunks: Buffer[] = [];
    try {
      for await (const chunk of req) { size += chunk.length; if (size > limit) { req.resume(); send(res,413,'{"error":"Request too large"}'); return; } chunks.push(Buffer.from(chunk)); }
      const file = path.resolve('dist-api', route.replace(/^\/api\//,'api/') + '.js');
      const mod = await import(pathToFileURL(file).href); const handler = mod.default ?? mod.fetch;
      if (!handler) return send(res,404,'{"error":"Not found"}');
      const headers = new Headers(); for (const [key,value] of Object.entries(req.headers)) if (value !== undefined && key !== 'x-forwarded-for' && key !== 'x-real-ip') headers.set(key, Array.isArray(value) ? value.join(', ') : value);
      headers.set('x-forwarded-for', req.socket.remoteAddress ?? '');
      const request = new Request(`http://localhost${req.url}`, { method:req.method, headers, body: chunks.length ? Buffer.concat(chunks) : undefined, duplex:'half' } as RequestInit);
      const response: Response = await handler(request); const output = Buffer.from(await response.arrayBuffer());
      const responseHeaders = Object.fromEntries(response.headers);
      res.writeHead(response.status, { ...Object.fromEntries(Object.entries(res.getHeaders()).map(([key, value]) => [key, String(value)])), ...responseHeaders });
      res.end(response.status === 204 || response.status === 304 ? undefined : output);
    } catch (error) { if (!res.headersSent) send(res, error instanceof SyntaxError ? 400 : 500, error instanceof SyntaxError ? '{"error":"Bad request"}' : '{"error":"Internal server error"}'); }
  });
}
