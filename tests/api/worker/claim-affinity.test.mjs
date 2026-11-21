import test from "node:test";
import assert from "node:assert/strict";
import claim from "../../../api/worker/claim.ts";
import { sha256, signVolunteerMarker, renderVolunteerMarker } from "../../../api/worker/_lib.ts";

const hardware = { architecture: "amd64", free_disk_bytes: 0, gpu_memory_utilisation: 0.8,
  selected_gpu_index: 0, selected_gpu_uuid: "gpu", gpus: [{ index: 0, name: "GPU", uuid: "gpu",
    free_memory_bytes: 1_000_000, total_memory_bytes: 1_000_000 }] };

for (const scenario of ["recent own work", "completed", "foreign", "unsigned", "forged", "no history"]) {
  test(`claim affinity preserves eligibility and orders ${scenario}`, async () => {
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
    try {
      const credentialKey = `euroeval:worker:credential:${await sha256("credential")}`;
      const issues = [1, 2, 3].map((number) => ({ number,
        title: `[MODEL EVALUATION REQUEST] org/model-${number}`,
        body: `### Model ID\n\norg/model-${number}\n\n- [x] Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)`,
        state: "open", assignees: [] }));
      for (const issue of issues.slice(1)) {
        if (scenario === "no history" || (issue.number === 2 && scenario !== "recent own work")) continue;
        const languages = scenario === "completed" ? ["da", "fo", "is", "no", "sv"] : ["sv"];
        const marker = await signVolunteerMarker(issue.number, {
          protocol_version: "volunteer-worker/v1", coordinator: "coordinator", submission: "submitted",
          leases: [], submissions: languages.map((language, index) => ({
            submission_id: `done-${language}`, language, manifest_path: `manifests/${language}.json`,
            submitted_at: issue.number === 3 ? "2026-11-20T18:00:00Z" : "2026-11-19T18:00:00Z",
            verified_contributor: scenario === "foreign" ? "bob" : "alice", result_count: 1,
            status: index === 0 ? "accepted" : "submitted", ownership_mode: "public",
          })),
        });
        if (scenario === "unsigned") delete marker.signature;
        if (scenario === "forged") marker.signature = "forged";
        issue.body += `\n${renderVolunteerMarker(marker)}`;
      }
      const inspectedModels = [];
      globalThis.fetch = async (input, init = {}) => {
        const url = String(input);
        if (url === "https://redis.test") {
          const command = JSON.parse(init.body);
          if (command[0] === "GET") return Response.json({ result: command[1] === credentialKey
            ? JSON.stringify({ contributor: "alice" }) : null });
          return Response.json({ result: command[0] === "SET" ? "OK" : 1 });
        }
        if (url.includes("/repos/EuroEval/EuroEval/issues?")) {
          return Response.json(Number(new URL(url).searchParams.get("page")) === 1 ? issues : []);
        }
        if (url.includes("huggingface.co/api/models/")) {
          const id = url.split("/api/models/")[1].split("?")[0];
          inspectedModels.push(id);
          return Response.json({ id, private: false, gated: false, siblings: [
            { rfilename: "model.safetensors", size: 100 }, { rfilename: "config.json", size: 10 },
          ] });
        }
        throw new Error(`Unexpected request: ${init.method || "GET"} ${url}`);
      };
      const response = await claim(new Request("https://euroeval.test/api/worker/claim", {
        method: "POST", headers: { authorization: "Bearer credential", "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: "volunteer-worker/v1", worker_version: "1.0.0", hardware }),
      }));
      assert.equal(response.status, 200, await response.clone().text());
      assert.deepEqual(await response.json(), { protocol_version: "volunteer-worker/v1", status: "no_work" });
      assert.deepEqual(inspectedModels, scenario === "recent own work"
        ? ["org/model-3", "org/model-2", "org/model-1"]
        : ["org/model-1", "org/model-2", "org/model-3"]);
    } finally {
      globalThis.fetch = originalFetch;
      process.env = originalEnv;
    }
  });
}
