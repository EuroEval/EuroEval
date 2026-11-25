import { optionalEnv } from "./protocol.js";

export const DEFAULT_PUBLICATION_BUCKET = "EuroEval/results";

export function canonicalPublicationBucket(): string {
  return optionalEnv("HF_RESULTS_BUCKET", DEFAULT_PUBLICATION_BUCKET);
}
