import { test } from "node:test";
import assert from "node:assert/strict";
import handler from "../../../api/worker/release-info.ts";

const digest = `sha256:${"a".repeat(64)}`;
function configured() {
  process.env.VOLUNTEER_WORKER_IMAGE_DIGEST = digest;
  process.env.VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64 = digest;
  process.env.VOLUNTEER_WORKER_VERSION = "1.2.3";
  process.env.EUROEVAL_VERSION = "2.0.0";
}
test("release-info normalizes architecture and disables caching", () => {
  configured();
  const response = handler(new Request("https://example.test/api/worker/release-info?architecture=x86_64"));
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("cache-control"), "no-store");
});
test("release-info rejects invalid architectures and methods", () => {
  configured();
  assert.equal(handler(new Request("https://example.test/?architecture=wasm")).status, 503);
  assert.equal(handler(new Request("https://example.test/", { method: "POST" })).status, 405);
});
test("release-info reports missing configuration", () => {
  delete process.env.VOLUNTEER_WORKER_IMAGE_DIGEST;
  delete process.env.VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64;
  const response = handler(new Request("https://example.test/?architecture=amd64"));
  assert.equal(response.status, 503);
});
