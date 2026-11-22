import test from "node:test";
import assert from "node:assert/strict";
import claim from "../../../api/worker/claim.ts";
import { sha256 } from "../../../api/worker/_lib.ts";

const amdDigest = `sha256:${"a".repeat(64)}`;
const armDigest = `sha256:${"b".repeat(64)}`;
const hardware = { architecture: "amd64", free_disk_bytes: 1_000_000, gpu_memory_utilisation: 0.8,
  selected_gpu_index: 0, selected_gpu_uuid: "gpu", gpus: [{ index: 0, name: "GPU", uuid: "gpu",
    free_memory_bytes: 1_000_000, total_memory_bytes: 1_000_000 }] };

const scenarios = [
  { name: "stale digest", body: { image_digest: armDigest }, status: 426 },
  { name: "missing digest", body: { image_digest: undefined }, status: 426 },
  { name: "malformed digest", body: { image_digest: "sha256:image" }, status: 426 },
  { name: "missing EuroEval version", body: { euroeval_version: undefined }, status: 426 },
  { name: "stale EuroEval version", body: { euroeval_version: "0.9" }, status: 426 },
  { name: "invalid EuroEval version", body: { euroeval_version: "latest" }, status: 426 },
  { name: "missing worker version", body: { worker_version: undefined }, status: 426 },
  { name: "stale worker version", body: { worker_version: "0.9" }, status: 426 },
  { name: "development number differs", body: { euroeval_version: "1.0.0.dev1" }, status: 426 },
  { name: "release is not development", body: { euroeval_version: "1.0.0" }, status: 426 },
  { name: "unknown managed architecture", architecture: "riscv64", status: 426 },
  { name: "ARM cannot claim with AMD digest", architecture: "arm64", status: 426 },
  { name: "bad configured digest", env: { VOLUNTEER_WORKER_IMAGE_DIGEST: "sha256:image" }, status: 503 },
  { name: "bad configured version", env: { EUROEVAL_VERSION: "latest" }, status: 503 },
  { name: "missing configured version", env: { EUROEVAL_VERSION: "" }, status: 503 },
  { name: "managed AMD", status: 200 },
  { name: "managed x86 alias", architecture: "x86_64", status: 200 },
  { name: "managed ARM", architecture: "arm64", body: { image_digest: armDigest }, status: 200 },
  { name: "managed ARM alias", architecture: "aarch64", body: { image_digest: armDigest }, status: 200 },
  { name: "ARM index fallback", architecture: "arm64", env: { VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64: "" }, status: 200 },
  { name: "canonical PEP440 spellings", body: { euroeval_version: "v01.0.0-dev", worker_version: "1.0.0.0" }, status: 200 },
  { name: "canonical prerelease spelling", body: { worker_version: "1.0preview" },
    env: { VOLUNTEER_WORKER_VERSION: "1.0.0rc0" }, status: 200 },
  { name: "canonical postrelease spelling", body: { worker_version: "1.0-1" },
    env: { VOLUNTEER_WORKER_VERSION: "1.0.0.post1" }, status: 200 },
  { name: "canonical epoch and local spelling", body: { worker_version: "01!1.0+GPU-01" },
    env: { VOLUNTEER_WORKER_VERSION: "1!1.0.0+gpu.1" }, status: 200 },
  { name: "legacy unknown architecture and unvalidated digest", architecture: "riscv64",
    body: { capabilities: undefined, image_digest: undefined, euroeval_version: undefined },
    env: { VOLUNTEER_WORKER_IMAGE_DIGEST: "sha256:image" }, status: 200 },
  { name: "legacy worker mismatch stays 422", body: { capabilities: undefined, worker_version: "0.9" }, status: 422 },
];

for (const scenario of scenarios) {
  test(`claim handshake: ${scenario.name}`, async () => {
    const originalFetch = globalThis.fetch;
    const originalEnv = { ...process.env };
    Object.assign(process.env, { VOLUNTEER_MARKER_SECRET: "marker-secret",
      VOLUNTEER_WORKER_IMAGE_DIGEST: amdDigest, VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64: armDigest,
      EUROEVAL_VERSION: "1.0.0.dev", VOLUNTEER_WORKER_VERSION: "1.0.0",
      UPSTASH_REDIS_REST_URL: "https://redis.test", UPSTASH_REDIS_REST_TOKEN: "redis-token",
      GITHUB_TOKEN: "github-token", VOLUNTEER_SCOPE_POLICY_JSON: JSON.stringify({
        policy_version: "test-policy", policies: [{ euroeval_version: "1.0.0.dev0", model_type: "encoder",
          language: "da", language_group: "Danish", allowed_identity_suffix_sets: [["test"]],
          task_groups: ["classification"], warnings: [] }],
      }), ...scenario.env });
    const credentialKey = `euroeval:worker:credential:${await sha256("credential")}`;
    const externalCalls = [];
    const redisCommands = [];
    globalThis.fetch = async (input, init = {}) => {
      const url = String(input);
      if (url === "https://redis.test") {
        const command = JSON.parse(init.body);
        redisCommands.push(command);
        if (command[0] === "GET" && command[1] === credentialKey) {
          return Response.json({ result: JSON.stringify({ contributor: "alice" }) });
        }
        return Response.json({ result: command[0] === "SET" ? "OK" : 1 });
      }
      externalCalls.push([url, init.method || "GET"]);
      if (url.includes("/issues?")) return Response.json([]);
      throw new Error(`Unexpected request: ${url}`);
    };
    try {
      const response = await claim(new Request("https://euroeval.test/api/worker/claim", {
        method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: "volunteer-worker/v1", worker_version: "1.0.0",
          capabilities: ["managed_image_v1"], image_digest: amdDigest, euroeval_version: "1.0.0.dev0",
          hardware: { ...hardware, architecture: scenario.architecture || "amd64" }, ...scenario.body }),
      }));
      assert.equal(response.status, scenario.status, await response.clone().text());
      const body = await response.json();
      if (scenario.status === 426) {
        assert.equal(body.code, "worker_update_required");
        assert.equal(body.protocol_version, "volunteer-worker/v1");
      }
      if (scenario.status === 200) {
        assert.equal(body.status, "no_work");
        assert.ok(externalCalls.length > 0);
      } else {
        // No queue discovery, model lookup, canary reservation, assignment or lease writes.
        assert.deepEqual(externalCalls, []);
        assert.ok(redisCommands.every((command) => command[1] === credentialKey ||
          String(command[1]).startsWith("euroeval:worker:limit:claim:")));
      }
    } finally {
      globalThis.fetch = originalFetch;
      process.env = originalEnv;
    }
  });
}
