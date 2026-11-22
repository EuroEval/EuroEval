import { BrokerError, ConfigurationError, brokerErrorBody, json } from "./_lib.js";
import { approvedWorkerRelease, IMAGE_REPOSITORY, normalizeArchitecture } from "./_lib/worker-release.js";
export const config = { runtime: "edge" };
export default function handler(req: Request): Response {
  if (req.method !== "GET") return json(405, { error: "Method not allowed" }, { allow: "GET", "cache-control": "no-store" });
  try {
    const architecture = new URL(req.url).searchParams.get("architecture");
    if (!architecture) throw new BrokerError(400, "architecture is required.");
    if (!normalizeArchitecture(architecture)) throw new BrokerError(400, "Unsupported worker architecture.");
    const release = approvedWorkerRelease(architecture);
    return json(200, { protocol_version: "volunteer-worker/v1", launcher_protocol_version: "volunteer-launcher/v1", ...release, image_repository: IMAGE_REPOSITORY }, { "cache-control": "no-store" });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 500;
    return json(status, brokerErrorBody(error, "Unable to resolve worker release."), { "cache-control": "no-store" });
  }
}
