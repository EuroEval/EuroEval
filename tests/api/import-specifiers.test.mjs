import test from "node:test";
import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
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

test("api TypeScript files have no relative .ts import/export specifiers", async () => {
  const apiDir = path.join(process.cwd(), "api");
  const files = await collectApiTypeScriptFiles(apiDir);
  const bad = [];

  for (const file of files) {
    const source = await readFile(file, "utf8");
    const sourceFile = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);

    const visit = (node) => {
      if (ts.isImportDeclaration(node)) {
        const specifier = node.moduleSpecifier?.text;
        if (specifier && isRelativeSpecifier(specifier) && specifier.endsWith(".ts")) {
          bad.push(`${path.relative(process.cwd(), file)} -> ${specifier}`);
        }
      }

      if (ts.isExportDeclaration(node) && node.moduleSpecifier) {
        const specifier = node.moduleSpecifier.text;
        if (specifier && isRelativeSpecifier(specifier) && specifier.endsWith(".ts")) {
          bad.push(`${path.relative(process.cwd(), file)} -> ${specifier}`);
        }
      }

      ts.forEachChild(node, visit);
    };

    visit(sourceFile);
  }

  assert.equal(
    bad.length,
    0,
    `Found relative TypeScript import/export specifiers with a .ts extension:\n${bad.join("\n")}`,
  );
});
