import { uploadFile } from "@huggingface/hub";
import { fetchWithRetry } from "./http.js";
import { storedCanaryPaths } from "./canary.js";
import { BrokerError, ConfigurationError, optionalEnv } from "./protocol.js";

let evidenceBucket: { id: string; private: boolean; checkedAt: number } | null = null;

export async function uploadCanaryEvidence(
  identity: string,
  digest: string,
  content: string,
): Promise<"uploaded" | "duplicate"> {
  const bucket = optionalEnv("HF_CANARY_EVIDENCE_BUCKET", "");
  const token = optionalEnv("HF_TOKEN", "");
  if (!/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(bucket) || !token) {
    throw new ConfigurationError("HF_CANARY_EVIDENCE_BUCKET and HF_TOKEN are required.");
  }
  if (!evidenceBucket || evidenceBucket.id !== bucket || Date.now() - evidenceBucket.checkedAt > 60_000) {
    const info = await fetchWithRetry(`https://huggingface.co/api/buckets/${bucket}`, {
      headers: { accept: "application/json", authorization: `Bearer ${token}` },
    });
    if (!info.ok) throw new BrokerError(502, `Canary evidence bucket lookup failed with HTTP ${info.status}.`);
    const metadata = await info.json() as { private?: boolean };
    evidenceBucket = { id: bucket, private: metadata.private === true, checkedAt: Date.now() };
  }
  if (!evidenceBucket.private) throw new ConfigurationError("Canary evidence bucket must be private.");
  const path = `v1/${identity}/${digest}.json`;
  const existing = await storedCanaryPaths(identity);
  if (existing.includes(path)) return "duplicate";
  if (existing.length) {
    throw new BrokerError(409, "Different canary evidence already exists; manual review is required.");
  }
  try {
    await uploadFile({
      repo: `buckets/${bucket}`,
      file: { path, content: new Blob([content], { type: "application/json" }) },
      accessToken: token,
    });
    return "uploaded";
  } catch (error) {
    throw new BrokerError(502, `Canary evidence upload failed: ${error instanceof Error ? error.message : "unknown error"}`);
  }
}
