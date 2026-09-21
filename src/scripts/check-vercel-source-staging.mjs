#!/usr/bin/env node
/** Check that Vercel Git builds receive every build-time source input. */

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import ignore from "ignore";

const REPO_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const IGNORE_FILE = ".vercelignore";
const FRONTEND_ROOT = "src/frontend/";
const GENERATED_CSV_ROOT = "src/frontend/csv/";
const API_ROOT = "api/";
const PYTHON_SOURCE_ROOT = "src/euroeval/";

export const REQUIRED_FILES = new Set([
  "index.html",
  "vite.config.js",
  "package.json",
  "package-lock.json",
  "tsconfig.json",
  "tsconfig.node.json",
  "vercel.json",
  "pyproject.toml",
  "src/scripts/build-seo-files.mjs",
  "src/scripts/build-api-reference.mjs",
  "src/scripts/build_api_reference.py",
  "src/frontend/App.vue",
  "src/frontend/config.yaml",
  "src/frontend/main.ts",
  "src/frontend/md/about.md",
  "src/euroeval/__init__.py",
]);

function trackedPaths(repoRoot) {
  const output = execFileSync("git", ["ls-files", "-z"], {
    cwd: repoRoot,
    encoding: "utf8",
  });
  return new Set(output.split("\0").filter(Boolean));
}

export function buildInputPaths(paths, requiredFiles = REQUIRED_FILES) {
  const buildInputs = new Set(requiredFiles);
  for (const filePath of paths) {
    if (filePath.startsWith(FRONTEND_ROOT) && !filePath.startsWith(GENERATED_CSV_ROOT)) {
      buildInputs.add(filePath);
    }
    // Vercel deploys each tracked API function from the repository checkout.
    // Derive this list so new routes and shared modules cannot be omitted here.
    if (filePath.startsWith(API_ROOT)) {
      buildInputs.add(filePath);
    }
    if (filePath.startsWith(PYTHON_SOURCE_ROOT) && filePath.endsWith(".py")) {
      buildInputs.add(filePath);
    }
  }
  return buildInputs;
}

function parentPaths(filePath) {
  const parents = [];
  let parent = path.posix.dirname(filePath);
  while (parent !== ".") {
    parents.push(parent);
    parent = path.posix.dirname(parent);
  }
  return parents;
}

export function validateIgnoreText({
  tracked,
  ignoreText,
  requiredFiles = REQUIRED_FILES,
}) {
  // `ignore` implements the same ordered gitignore/negation matching Vercel uses.
  // It is deliberately case-sensitive, matching Vercel's Linux build environment.
  const spec = ignore({ ignoreCase: false }).add(ignoreText.split(/\r?\n/));
  const paths = new Set(tracked);
  const inputs = buildInputPaths(paths, requiredFiles);
  const errors = [];

  for (const filePath of [...requiredFiles].sort()) {
    if (!paths.has(filePath)) {
      errors.push(`required build input is not tracked: ${filePath}`);
    }
  }

  for (const filePath of [...inputs].sort()) {
    if (spec.ignores(filePath)) {
      errors.push(`build input is ignored by ${IGNORE_FILE}: ${filePath}`);
    }
    for (const parent of parentPaths(filePath)) {
      if (spec.ignores(`${parent}/`)) {
        errors.push(
          `parent directory of build input is ignored by ${IGNORE_FILE}: ${parent}/ (needed by ${filePath})`,
        );
      }
    }
  }
  return errors;
}

export function validateSourceStaging(repoRoot = REPO_ROOT) {
  const ignorePath = path.join(repoRoot, IGNORE_FILE);
  if (!fs.existsSync(ignorePath)) return [`missing ${IGNORE_FILE}`];

  const tracked = trackedPaths(repoRoot);
  return validateIgnoreText({
    tracked,
    ignoreText: fs.readFileSync(ignorePath, "utf8"),
  });
}

export function main(repoRoot = REPO_ROOT) {
  const errors = validateSourceStaging(path.resolve(repoRoot));
  if (errors.length > 0) {
    for (const error of errors) console.error(`ERROR: ${error}`);
    return 1;
  }
  console.info("Vercel source staging includes all build-time inputs");
  return 0;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  process.exitCode = main(process.argv[2] ?? REPO_ROOT);
}
