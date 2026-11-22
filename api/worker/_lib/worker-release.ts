import { ConfigurationError, env, optionalEnv } from "./protocol.js";

export const IMAGE_REPOSITORY = "ghcr.io/euroeval/euroeval-worker";
export function normalizeArchitecture(value: string): "amd64" | "arm64" | null {
  if (value === "amd64" || value === "x86_64") return "amd64";
  if (value === "arm64" || value === "aarch64") return "arm64";
  return null;
}

/** Compare PEP 440 versions without confusing spelling changes with updates. */
export function canonicalWorkerVersion(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const match = value.trim().toLowerCase().match(/^v?(?:(\d+)!)?(\d+(?:\.\d+)*)(?:[-_.]?(a|b|c|rc|alpha|beta|pre|preview)[-_.]?(\d*))?(?:(?:-(\d+))|(?:[-_.]?(post|rev|r)[-_.]?(\d*)))?(?:[-_.]?dev[-_.]?(\d*))?(?:\+([a-z0-9]+(?:[-_.][a-z0-9]+)*))?$/);
  if (!match) return null;
  const integer = (part: string) => part.replace(/^0+(?=\d)/, "");
  const release = match[2].split(".").map(integer);
  while (release.length > 1 && release[release.length - 1] === "0") release.pop();
  const aliases: Record<string, string> = { alpha: "a", beta: "b", c: "rc", pre: "rc", preview: "rc" };
  const pre = match[3] ? `${aliases[match[3]] || match[3]}${integer(match[4] || "0")}` : "";
  const post = match[5] !== undefined || match[6] ? `.post${integer(match[5] || match[7] || "0")}` : "";
  const dev = match[8] !== undefined ? `.dev${integer(match[8] || "0")}` : "";
  const local = match[9] ? `+${match[9].split(/[-_.]/).map((part) => /^\d+$/.test(part) ? integer(part) : part).join(".")}` : "";
  return `${integer(match[1] || "0")}!${release.join(".")}${pre}${post}${dev}${local}`;
}

export function approvedWorkerRelease(architecture: string) {
  const normalized = normalizeArchitecture(architecture);
  if (!normalized) throw new ConfigurationError("Unsupported worker architecture.");
  const armDigest = normalized === "arm64" ? optionalEnv("VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64", "") : "";
  const digest = armDigest || env("VOLUNTEER_WORKER_IMAGE_DIGEST");
  if (!/^sha256:[a-f0-9]{64}$/.test(digest)) throw new ConfigurationError("Invalid approved worker image digest.");
  const workerVersion = env("VOLUNTEER_WORKER_VERSION");
  const euroevalVersion = env("EUROEVAL_VERSION").replace(/\.dev$/, ".dev0");
  if (!canonicalWorkerVersion(workerVersion) || !canonicalWorkerVersion(euroevalVersion)) throw new ConfigurationError("Invalid worker release version.");
  return { architecture: normalized, image_digest: digest, image_reference: `${IMAGE_REPOSITORY}@${digest}`, worker_version: workerVersion, euroeval_version: euroevalVersion };
}
