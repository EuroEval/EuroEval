import { ConfigurationError, env, optionalEnv } from "./protocol.js";

export const IMAGE_REPOSITORY = "ghcr.io/euroeval/euroeval-worker";
export function normalizeArchitecture(value: string): "amd64" | "arm64" | null {
  if (value === "amd64" || value === "x86_64") return "amd64";
  if (value === "arm64" || value === "aarch64") return "arm64";
  return null;
}

export function approvedWorkerRelease(architecture: string) {
  const normalized = normalizeArchitecture(architecture);
  if (!normalized) throw new ConfigurationError("Unsupported worker architecture.");
  const armDigest = normalized === "arm64" ? optionalEnv("VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64", "") : "";
  const digest = armDigest || env("VOLUNTEER_WORKER_IMAGE_DIGEST");
  if (!/^sha256:[a-f0-9]{64}$/.test(digest)) throw new ConfigurationError("Invalid approved worker image digest.");
  const workerVersion = env("VOLUNTEER_WORKER_VERSION");
  const euroevalVersion = env("EUROEVAL_VERSION").replace(/\.dev$/, ".dev0");
  if (!/^\S+$/.test(workerVersion) || !/^\S+$/.test(euroevalVersion)) throw new ConfigurationError("Invalid worker release version.");
  return { architecture: normalized, image_digest: digest, image_reference: `${IMAGE_REPOSITORY}@${digest}`, worker_version: workerVersion, euroeval_version: euroevalVersion };
}
