import test from "node:test";
import assert from "node:assert/strict";
import claim from "../../../api/worker/claim.ts";
import { sha256 } from "../../../api/worker/_lib.ts";

const hardware = { architecture: "amd64", free_disk_bytes: 0, gpu_memory_utilisation: 0.8,
  selected_gpu_index: 0, selected_gpu_uuid: "gpu", gpus: [{ index: 0, name: "GPU", uuid: "gpu",
    free_memory_bytes: 1_000_000, total_memory_bytes: 1_000_000 }] };

test("claim affinity ignores unsigned contributor history and preserves ordinary queue priority", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  Object.assign(process.env, { VOLUNTEER_MARKER_SECRET: "marker-secret",
    VOLUNTEER_WORKER_IMAGE_DIGEST: "sha256:image", EUROEVAL_VERSION: "1.0.0",
    VOLUNTEER_WORKER_VERSION: "1.0.0", UPSTASH_REDIS_REST_URL: "https://redis.test",
    UPSTASH_REDIS_REST_TOKEN: "redis-token", GITHUB_TOKEN: "github-token",
    VOLUNTEER_SCOPE_POLICY_JSON: JSON.stringify({ policy_version: "test-policy", policies: [
      { euroeval_version: "1.0.0", model_type: "encoder", language: "da", language_group: "Danish",
        allowed_identity_suffix_sets: [["test"]], task_groups: ["classification"], warnings: [] },
    ] }) });
  const credentialKey = `euroeval:worker:credential:${await sha256("credential")}`;
  const issues = [1, 2].map((number) => ({ number,
    title: `[MODEL EVALUATION REQUEST] org/model-${number}`,
    body: `### Model ID\n\norg/model-${number}\n\n- [x] Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)`,
    state: "open", assignees: [] }));
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input);
    if (url === "https://redis.test") {
      const command = JSON.parse(init.body);
      if (command[0] === "GET") return Response.json({ result: command[1] === credentialKey
        ? JSON.stringify({ contributor: "alice" }) : null });
      return Response.json({ result: command[0] === "SET" ? "OK" : 1 });
    }
    if (url.includes("/repos/EuroEval/EuroEval/issues?")) {
      const page = Number(new URL(url).searchParams.get("page"));
      return Response.json(page === 1 ? issues : []);
    }
    if (url.includes("huggingface.co/api/models/")) {
      const id = url.split("/api/models/")[1].split("?")[0];
      return Response.json({ id, private: false, gated: false, siblings: [
        { rfilename: "model.safetensors", size: 100 }, { rfilename: "config.json", size: 10 },
      ] });
    }
    throw new Error(`Unexpected request: ${init.method || "GET"} ${url}`);
  };
  try {
    const response = await claim(new Request("https://euroeval.test/api/worker/claim", {
      method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: "volunteer-worker/v1", worker_version: "1.0.0", hardware }),
    }));
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), { protocol_version: "volunteer-worker/v1", status: "no_work" });
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
  }
});
