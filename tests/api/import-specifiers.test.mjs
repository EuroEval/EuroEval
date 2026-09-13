import test from "node:test";
import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import { statSync } from "node:fs";
import path from "node:path";
import { builtinModules } from "node:module";
import * as ts from "typescript";

async function collectApiTypeScriptFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const filePath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...await collectApiTypeScriptFiles(filePath));
      continue;
    }
    if (entry.isFile() && filePath.endsWith(".ts")) {
      files.push(filePath);
    }
  }
  return files;
}

function isRelativeSpecifier(specifier) {
  return specifier.startsWith("./") || specifier.startsWith("../");
}

test("api TypeScript files use explicit relative JavaScript specifiers", async () => {
  const apiDir = path.join(process.cwd(), "api");
  const files = await collectApiTypeScriptFiles(apiDir);
  const bad = [];

  for (const file of files) {
    const source = await readFile(file, "utf8");
    const sourceFile = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);

    const visit = (node) => {
      if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) {
        const specifier = node.moduleSpecifier?.text;
        if (specifier && isRelativeSpecifier(specifier) && !specifier.endsWith(".js")) {
          bad.push(`${path.relative(process.cwd(), file)} -> ${specifier}`);
        }
        if (node.attributes?.elements?.length) {
          bad.push(`${path.relative(process.cwd(), file)} -> import attributes`);
        }
      }
      ts.forEachChild(node, visit);
    };

    visit(sourceFile);
  }

  assert.equal(
    bad.length,
    0,
    `Found unsupported relative import/export specifiers or attributes:\n${bad.join("\n")}`,
  );
});

function moduleSpecifiers(file, source) {
  const sourceFile = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const specifiers = [];
  const visit = (node) => {
    if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) {
      if (node.moduleSpecifier) specifiers.push(node.moduleSpecifier.text);
    }
    if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword) {
      const [argument] = node.arguments;
      if (argument && ts.isStringLiteral(argument)) specifiers.push(argument.text);
    }
    ts.forEachChild(node, visit);
  };
  visit(sourceFile);
  return specifiers;
}

function resolveLocalModule(file, specifier) {
  const withoutJavaScript = specifier.endsWith(".js") ? specifier.slice(0, -3) : specifier;
  const base = path.resolve(path.dirname(file), withoutJavaScript);
  const candidates = [`${base}.ts`, path.join(base, "index.ts")];
  return candidates.find((candidate) => {
    try { return statSync(candidate).isFile(); } catch { return false; }
  });
}

async function sourceGraph(entry) {
  const modules = new Set();
  const external = new Set();
  const visit = async (file) => {
    if (modules.has(file)) return;
    modules.add(file);
    const source = await readFile(file, "utf8");
    for (const specifier of moduleSpecifiers(file, source)) {
      if (!isRelativeSpecifier(specifier)) {
        external.add(specifier);
        continue;
      }
      const resolved = resolveLocalModule(file, specifier);
      assert.ok(resolved, `Could not resolve ${specifier} imported by ${file}`);
      await visit(resolved);
    }
  };
  await visit(entry);
  return { modules, external };
}

function endpointFiles(files) {
  return files.filter((file) => {
    const relative = path.relative(path.join(process.cwd(), "api"), file);
    // Vercel treats underscore-prefixed files and directories as private modules.
    return relative.split(path.sep).every((part) => !part.startsWith("_"));
  });
}

async function runtimeDeclaration(file, seen = new Set()) {
  if (seen.has(file)) return null;
  seen.add(file);
  const source = await readFile(file, "utf8");
  const runtime = source.match(/export\s+const\s+config\s*=\s*\{\s*runtime\s*:\s*["']([^"']+)["']/)?.[1];
  if (runtime) return runtime;
  const reexport = source.match(/export\s*\{\s*config\s*\}\s*from\s*["']([^"']+)["']/)?.[1];
  if (!reexport) return null;
  const resolved = resolveLocalModule(file, reexport);
  assert.ok(resolved, `Could not resolve runtime config imported by ${file}`);
  return runtimeDeclaration(resolved, seen);
}

test("Edge endpoint graphs cannot reach Hugging Face Hub upload code", async () => {
  const apiDir = path.join(process.cwd(), "api");
  const files = endpointFiles(await collectApiTypeScriptFiles(apiDir));
  const nodeEndpoints = new Set([
    path.join(apiDir, "worker", "result.ts"),
    path.join(apiDir, "worker", "finalise.ts"),
  ]);
  const eee = path.join(apiDir, "worker", "_lib", "eee.ts");
  const eeeConsumers = [];

  for (const file of files) {
    const graph = await sourceGraph(file);
    if (graph.modules.has(eee)) eeeConsumers.push(file);
    if (!nodeEndpoints.has(file)) {
      assert.equal(
        [...graph.external].some((specifier) => specifier === "@huggingface/hub" || specifier.startsWith("@huggingface/hub/")),
        false,
        `${path.relative(process.cwd(), file)} reaches @huggingface/hub`,
      );
      assert.deepEqual(
        [...graph.external].filter((specifier) => builtinModules.includes(specifier) || specifier.startsWith("node:")),
        [],
        `${path.relative(process.cwd(), file)} reaches a Node builtin`,
      );
    }
  }

  assert.deepEqual(eeeConsumers.sort(), [...nodeEndpoints].sort(), "Only result and finalise may reach eee.ts");
});

test("API endpoint runtime declarations match their module graphs", async () => {
  const apiDir = path.join(process.cwd(), "api");
  const files = endpointFiles(await collectApiTypeScriptFiles(apiDir));
  const nodeEndpoints = new Set([
    path.join(apiDir, "worker", "result.ts"),
    path.join(apiDir, "worker", "finalise.ts"),
  ]);

  for (const file of files) {
    const runtime = await runtimeDeclaration(file);
    if (nodeEndpoints.has(file)) {
      assert.equal(runtime, "nodejs", `${path.relative(process.cwd(), file)} must use Node`);
    } else {
      assert.equal(runtime, "edge", `${path.relative(process.cwd(), file)} must declare Edge runtime`);
    }
  }
});
