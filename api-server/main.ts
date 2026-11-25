import { createApiServer } from './server.js';

const port = Number(process.env.PORT ?? 3000);
const server = createApiServer({ allowedOrigins: (process.env.API_ALLOWED_ORIGINS ?? 'https://euroeval.com').split(',').map(value => value.trim()).filter(Boolean) });
server.listen(port, '127.0.0.1');
for (const signal of ['SIGINT', 'SIGTERM'] as const) process.on(signal, () => server.close(() => process.exit(0)));
