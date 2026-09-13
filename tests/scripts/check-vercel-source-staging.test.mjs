import assert from "node:assert/strict";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  validateIgnoreText,
  validateSourceStaging,
} from "../../src/scripts/check-vercel-source-staging.mjs";

const REPO_ROOT = fileURLToPath(new URL("../../", import.meta.url));

function errorsFor(ignoreText, tracked = ["src/frontend/App.vue"]) {
  return validateIgnoreText({
    tracked,
    ignoreText,
    requiredFiles: new Set(["src/frontend/App.vue"]),
  });
}

test("the repository keeps all tracked build inputs available", () => {
  assert.deepEqual(validateSourceStaging(REPO_ROOT), []);
});

test("an excluded parent directory hides a build input", () => {
  assert.match(
    errorsFor("src/frontend/\n").join("\n"),
    /parent directory of build input.*src\/frontend\//,
  );
});

test("a child-only negation cannot resurrect an excluded parent", () => {
  const errors = errorsFor("src/frontend/\n!src/frontend/App.vue\n");
  assert.match(
    errors.join("\n"),
    /build input is ignored.*src\/frontend\/App\.vue/,
  );
  assert.match(
    errors.join("\n"),
    /parent directory of build input.*src\/frontend\//,
  );
});

test("reinclusion of the parent directory keeps the input available", () => {
  assert.deepEqual(
    errorsFor("src/frontend/\n!src/frontend/\n"),
    [],
  );
});

test("reinclusion of a leaf after a child glob keeps that leaf available", () => {
  assert.deepEqual(
    errorsFor("src/frontend/*\n!src/frontend/App.vue\n"),
    [],
  );
});

test("later patterns override earlier negations", () => {
  assert.notDeepEqual(
    errorsFor("src/frontend/*\n!src/frontend/App.vue\nsrc/frontend/App.vue\n"),
    [],
  );
  assert.deepEqual(
    errorsFor("src/frontend/*\nsrc/frontend/App.vue\n!src/frontend/App.vue\n"),
    [],
  );
});

test("matching is case-sensitive like Vercel on Linux", () => {
  assert.deepEqual(errorsFor("SRC/FRONTEND/\n"), []);
  assert.notDeepEqual(errorsFor("src/frontend/\n"), []);
});
