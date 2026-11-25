import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const routes = ['/api/issues', '/api/submit-evaluation', '/api/hall-of-fame', ...['auth/start','auth/poll','auth/revoke','canary-corpus','claim','coordinator-lock','coordinator-release','coordinator-renew','finalise','heartbeat','promote','promotion-lock','promotion-reserve','promotion-status','release-info','release','result'].map(x => `/api/worker/${x}`)];
const origins = new Set((process.env.API_ALLOWED_ORIGINS ?? 'https://euroeval.com').split(',').map(x => x.trim()).filter(Boolean));
async function serve(req: IncomingMessage, res: ServerResponse) {
  const origin = req.headers.origin; res.setHeader('Vary', 'Origin');
  if (origin && origins.has(origin)) res.setHeader('Access-Control-Allow-Origin', origin);
  if (req.method === 'OPTIONS') { res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, PATCH, DELETE, OPTIONS'); res.setHeader('Access-Control-Allow-Headers', 'Authorization, Content-Type, X-Worker-Token, X-Protocol-Version'); res.statusCode = 204; res.end(); return; }
  const route = new URL(req.url ?? '/', 'http://localhost').pathname;
  if (!routes.includes(route)) { res.statusCode = 404; res.end('Not found'); return; }
  try {
    const file = path.resolve('dist-api/api', route.replace(/^\/api\//, '') + '.js');
    const mod = await import(pathToFileURL(file).href); const handler = mod.default ?? mod.fetch;
    if (!handler) { res.statusCode = 404; res.end('Not found'); return; }
    const chunks: Buffer[] = []; for await (const chunk of req) chunks.push(Buffer.from(chunk));
    const method = req.method ?? 'GET';
    const request = new Request(`http://${req.headers.host ?? 'localhost'}${req.url}`, { method, headers: req.headers as HeadersInit, body: ['GET','HEAD'].includes(method) || !chunks.length ? undefined : Buffer.concat(chunks), duplex: 'half' } as RequestInit);
    const response: Response = await handler(request); res.statusCode = response.status; response.headers.forEach((v,k) => res.setHeader(k,v)); res.end(Buffer.from(await response.arrayBuffer()));
  } catch { if (!res.headersSent) { res.statusCode = 500; res.end('Internal server error'); } }
}
const server = createServer((req,res) => void serve(req,res));
server.listen(Number(process.env.PORT ?? 3000), process.env.HOST ?? '127.0.0.1');
