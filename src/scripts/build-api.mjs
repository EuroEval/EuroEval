import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const result = spawnSync(path.join(root, 'node_modules/.bin/tsc'), ['-p', 'tsconfig.api.json'], { cwd: root, stdio: 'inherit' });
if (result.status !== 0) process.exit(result.status ?? 1);
