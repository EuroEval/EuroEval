import assert from "node:assert/strict";
import {
  cp,
  mkdir,
  mkdtemp,
  realpath,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const REPO_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const NODE_MODULES = path.join(REPO_ROOT, "node_modules");
const CSV_CONTENT = "model,score\nshared-model,0.75\n";

async function createFixture({ csvTarget, createCsv = true } = {}) {
  const root = await mkdtemp(path.join(os.tmpdir(), "euroeval-vite-csv-"));
  await mkdir(path.join(root, "src/scripts"), { recursive: true });
  await mkdir(path.join(root, "src/frontend"), { recursive: true });
  await cp(
    path.join(REPO_ROOT, "vite.config.js"),
    path.join(root, "vite.config.js"),
  );
  await cp(
    path.join(REPO_ROOT, "pyproject.toml"),
    path.join(root, "pyproject.toml"),
  );
  await cp(
    path.join(REPO_ROOT, "src/scripts/build-seo-files.mjs"),
    path.join(root, "src/scripts/build-seo-files.mjs"),
  );
  await cp(
    path.join(REPO_ROOT, "src/scripts/build-api-reference.mjs"),
    path.join(root, "src/scripts/build-api-reference.mjs"),
  );
  await symlink(NODE_MODULES, path.join(root, "node_modules"), "dir");
  await writeFile(path.join(root, "package.json"), '{"type":"module"}\n');
  await writeFile(path.join(root, "src/frontend/config.yaml"), "title: Test\n");
  await writeFile(path.join(root, "src/scripts/build_api_reference.py"), "pass\n");
  if (createCsv) {
    const csvDir = path.join(root, "src/frontend/csv");
    if (csvTarget) await symlink(csvTarget, csvDir, "dir");
    else await mkdir(csvDir);
  }
  return root;
}

async function withServer(root, callback) {
  const { createServer } = await import("vite");
  const server = await createServer({
    configFile: path.join(root, "vite.config.js"),
    root,
    cacheDir: path.join(root, ".vite-cache"),
    server: { host: "127.0.0.1", port: 0 },
    logLevel: "silent",
  });
  try {
    await server.listen();
    const address = server.httpServer.address();
    assert.ok(address && typeof address !== "string");
    await callback(`http://127.0.0.1:${address.port}`);
  } finally {
    await server.close();
  }
}

function fileUrl(filePath) {
  return `/@fs${filePath.split(path.sep).join("/")}`;
}

test("Vite transforms shared CSV imports without broadening filesystem access", async () => {
  const fixture = await mkdtemp(path.join(os.tmpdir(), "euroeval-csv-target-"));
  const csvTarget = path.join(fixture, "csv");
  const outsideFile = path.join(fixture, "unrelated.txt");
  await mkdir(csvTarget);
  await writeFile(
    path.join(csvTarget, "european_understanding.csv"),
    CSV_CONTENT,
  );
  await writeFile(outsideFile, "not allowed\n");
  const root = await createFixture({ csvTarget });
  await writeFile(
    path.join(root, "src/frontend/csv/local.csv"),
    CSV_CONTENT,
  );

  try {
    await withServer(root, async (baseUrl) => {
      const csvPath = path.join(csvTarget, "european_understanding.csv");
      const imported = await fetch(
        `${baseUrl}${fileUrl(await realpath(csvPath))}?import&raw&cacheBust=1`,
        { headers: { "Sec-Fetch-Dest": "script" } },
      );
      assert.equal(imported.status, 200);
      assert.match(imported.headers.get("content-type"), /javascript/);
      const moduleSource = await imported.text();
      assert.match(moduleSource, /export default/);
      assert.match(moduleSource, /shared-model,0\.75/);

      const localCsv = await fetch(
        `${baseUrl}/src/frontend/csv/local.csv?import&raw`,
      );
      assert.equal(localCsv.status, 200);
      assert.match(localCsv.headers.get("content-type"), /javascript/);
      assert.match(await localCsv.text(), /shared-model,0\.75/);

      const download = await fetch(
        `${baseUrl}/leaderboards-csv/european_understanding.csv`,
      );
      assert.equal(download.status, 200);
      assert.match(download.headers.get("content-type"), /text\/csv/);
      assert.equal(await download.text(), CSV_CONTENT);

      const blocked = await fetch(
        `${baseUrl}${fileUrl(await realpath(outsideFile))}?import&raw`,
      );
      assert.equal(blocked.status, 403);
    });
  } finally {
    await rm(root, { recursive: true, force: true });
    await rm(fixture, { recursive: true, force: true });
  }
});

test("Vite config loads when the generated CSV directory is absent", async () => {
  const root = await createFixture({ createCsv: false });
  try {
    await withServer(root, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/@vite/client`);
      assert.equal(response.status, 200);
    });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
