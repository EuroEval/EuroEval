import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { resolve } from "./loader-extensionless-ts.mjs";

const apiRoot = path.resolve(process.cwd(), "api");
const parent = pathToFileURL(path.join(apiRoot, "worker", "claim.ts")).href;

function nextResolve(specifier, context) {
  return { url: specifier, parent: context.parentURL };
}

test("loader bridges only existing relative JavaScript imports in api TypeScript", async () => {
  const result = await resolve("./_lib.js", { parentURL: parent }, nextResolve);
  assert.equal(result.url, pathToFileURL(path.join(apiRoot, "worker", "_lib.ts")).href);
});

test("loader does not mask unrelated, outside, or missing imports", async () => {
  const probes = [
    "./scope-policy.json",
    "./does-not-exist.js",
    "../../src/frontend/services/github.js",
    "node:fs",
    pathToFileURL(path.join(apiRoot, "worker", "claim.ts")).href,
  ];
  for (const specifier of probes) {
    const result = await resolve(specifier, { parentURL: parent }, nextResolve);
    assert.equal(result.url, specifier, `loader rewrote ${specifier}`);
  }
});
