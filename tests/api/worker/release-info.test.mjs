import { test } from "node:test";
import assert from "node:assert/strict";
import handler from "../../../api/worker/release-info.ts";

const digest = `sha256:${"a".repeat(64)}`;
const armDigest = `sha256:${"b".repeat(64)}`;

async function configured(run) {
  const originalEnv = { ...process.env };
  Object.assign(process.env, { VOLUNTEER_WORKER_IMAGE_DIGEST: digest,
    VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64: armDigest, VOLUNTEER_WORKER_VERSION: "1.2.3",
    EUROEVAL_VERSION: "2.0.0.dev" });
  try { await run(); } finally { process.env = originalEnv; }
}

for (const [alias, architecture, selectedDigest] of [
  ["amd64", "amd64", digest], ["x86_64", "amd64", digest],
  ["arm64", "arm64", armDigest], ["aarch64", "arm64", armDigest],
]) {
  test(`release-info selects immutable ${alias} release and disables caching`, () => configured(async () => {
    const response = handler(new Request(`https://example.test/api/worker/release-info?architecture=${alias}`));
    assert.equal(response.status, 200);
    assert.equal(response.headers.get("cache-control"), "no-store");
    assert.deepEqual(await response.json(), {
      protocol_version: "volunteer-worker/v1", launcher_protocol_version: "volunteer-launcher/v1",
      architecture, image_digest: selectedDigest,
      image_reference: `ghcr.io/euroeval/euroeval-worker@${selectedDigest}`,
      image_repository: "ghcr.io/euroeval/euroeval-worker", worker_version: "1.2.3",
      euroeval_version: "2.0.0.dev0",
    });
  }));
}

test("release-info allows ARM to use the approved index digest", () => configured(async () => {
  delete process.env.VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64;
  const response = handler(new Request("https://example.test/?architecture=arm64"));
  assert.equal(response.status, 200);
  assert.equal((await response.json()).image_digest, digest);
}));

test("release-info rejects missing and unknown architectures without caching", () => configured(async () => {
  for (const query of ["", "?architecture=wasm"]) {
    const response = handler(new Request(`https://example.test/${query}`));
    assert.equal(response.status, 400);
    assert.equal(response.headers.get("cache-control"), "no-store");
  }
  const response = handler(new Request("https://example.test/", { method: "POST" }));
  assert.equal(response.status, 405);
  assert.equal(response.headers.get("allow"), "GET");
}));

for (const [key, value] of [
  ["VOLUNTEER_WORKER_IMAGE_DIGEST", undefined], ["VOLUNTEER_WORKER_IMAGE_DIGEST", "sha256:image"],
  ["VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64", "sha256:image"],
  ["VOLUNTEER_WORKER_VERSION", undefined], ["VOLUNTEER_WORKER_VERSION", "latest"],
  ["EUROEVAL_VERSION", undefined], ["EUROEVAL_VERSION", "not a version"],
]) {
  test(`release-info fails closed for ${key}=${value}`, () => configured(async () => {
    if (value === undefined) delete process.env[key]; else process.env[key] = value;
    const architecture = key.endsWith("ARM64") ? "arm64" : "amd64";
    const response = handler(new Request(`https://example.test/?architecture=${architecture}`));
    assert.equal(response.status, 503);
    assert.equal(response.headers.get("cache-control"), "no-store");
    assert.equal((await response.json()).protocol_version, "volunteer-worker/v1");
  }));
}
