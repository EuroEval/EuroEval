import { spawnSync } from 'node:child_process';
import { promises as fs } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const result = spawnSync(path.join(root, 'node_modules/.bin/tsc'), ['-p', 'tsconfig.api.json'], { cwd: root, stdio: 'inherit' });
if (result.status !== 0) process.exit(result.status ?? 1);
const csvDir = path.join(root, 'src/frontend/csv');
const stems = ['albanian', 'belarusian', 'bosnian', 'bulgarian', 'catalan', 'croatian', 'czech', 'danish', 'dutch', 'english', 'estonian', 'faroese', 'finnish', 'french', 'german', 'greek', 'luxembourgish', 'hungarian', 'icelandic', 'italian', 'latvian', 'lithuanian', 'norwegian', 'polish', 'portuguese', 'romanian', 'serbian', 'slovak', 'slovene', 'spanish', 'swedish', 'ukrainian'];
const expectedData = stems.flatMap((stem) => [`${stem}_generative_simplified.csv`, `${stem}_all_models_simplified.csv`]);
await fs.access(csvDir).catch(() => {
  throw new Error(`Leaderboard CSV directory is required: ${csvDir}`);
});
for (const file of expectedData) {
  await fs.access(path.join(csvDir, file)).catch(() => {
    throw new Error(`Required leaderboard CSV is missing: ${path.join(csvDir, file)}`);
  });
}
const { buildEvaluatedModelsIndex } = await import('./build-seo-files.mjs');
const evaluatedIndex = await buildEvaluatedModelsIndex(csvDir);
const outputDir = path.join(root, 'dist-api');
await fs.mkdir(outputDir, { recursive: true });
await fs.writeFile(path.join(outputDir, 'evaluated-models.json'), JSON.stringify(evaluatedIndex), 'utf-8');
