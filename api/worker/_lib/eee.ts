import { uploadFile } from "@huggingface/hub";
import { BrokerError, ConfigurationError, GROUPS, env } from "./protocol.js";
import { fetchWithRetry } from "./http.js";
import { languageGroup } from "./model.js";
import {
  CANARY_RESULT_DATASET,
  validateCanaryEvidence,
} from "./canary.js";

export function validateRecord(record: unknown, expected: { modelId: string; revision: string; language: string; euroevalVersion?: string; modelType?: "encoder" | "generative"; requireRevision?: boolean }): { identity: string; failed: number; warnings: string[] } {
  if (!record || typeof record !== "object" || Array.isArray(record)) throw new BrokerError(422, "record must be an EEE JSON object.");
  const value = record as Record<string, any>;
  const modelInfo = value.model_info as Record<string, any> | undefined;
  const library = value.eval_library as Record<string, any> | undefined;
  if (!["0.2.1", "0.3.0"].includes(value.schema_version) || !modelInfo || !library || !Array.isArray(value.evaluation_results)) throw new BrokerError(422, "record is not a supported EEE record.");
  const aliases = [modelInfo.id, modelInfo.name, modelInfo.aliases, (modelInfo.additional_details as Record<string, unknown> | undefined)?.aliases]
    .flatMap((item) => Array.isArray(item) ? item : [item]).filter((item): item is string => typeof item === "string");
  if (!aliases.some((item) => item === expected.modelId)) throw new BrokerError(422, `record model identity does not match ${expected.modelId}.`);
  const missingRevision = modelInfo.revision === undefined;
  if (missingRevision && expected.requireRevision !== false ||
      !missingRevision && modelInfo.revision !== expected.revision) {
    throw new BrokerError(422, "record model_info.revision is missing or does not match the lease.");
  }
  if (library.name !== "euroeval" || typeof library.version !== "string" || !library.version || expected.euroevalVersion && library.version.replace(/\.dev\d*$/, "") !== expected.euroevalVersion.replace(/\.dev\d*$/, "")) throw new BrokerError(422, "record has an invalid eval_library name or version.");
  const details = library.additional_details as Record<string, any> | undefined;
  if (!details || typeof details.dataset !== "string" || !details.dataset || typeof details.task !== "string" || !details.task) throw new BrokerError(422, "record has no canonical dataset/task identity.");
  if (details.dataset === CANARY_RESULT_DATASET || details.dataset.startsWith(`${CANARY_RESULT_DATASET}-`)) {
    if (details.task !== "contamination-detection") throw new BrokerError(422, "canary result has an invalid task.");
    const evidence = parseCanaryEvidence(details.contamination_canary_evidence);
    validateCanaryEvidence(evidence, {
      modelId: expected.modelId,
      revision: expected.revision,
    });
    const canary = evidence as Record<string, unknown>;
    if (
      expected.modelType === "encoder" &&
      (canary.status !== "not_applicable" || canary.reason !== "encoder")
    ) {
      throw new BrokerError(422, "encoder canary evidence must be not_applicable/encoder.");
    }
    if (expected.modelType === "generative" && canary.reason === "encoder") {
      throw new BrokerError(422, "generative canary evidence cannot use the encoder status.");
    }
  }
  let raw: unknown;
  try { raw = typeof details.raw_results === "string" ? JSON.parse(details.raw_results) : details.raw_results; } catch { throw new BrokerError(422, "additional_details.raw_results is not valid JSON."); }
  if (!Array.isArray(raw)) throw new BrokerError(422, "additional_details.raw_results is not valid JSON.");
  const failureEntries = raw.flatMap((iteration) => {
    const iterationCounts: number[] = [];
    const collectFailures = (item: unknown): void => {
      if (!item || typeof item !== "object") return;
      if (Array.isArray(item)) {
        for (const nested of item) collectFailures(nested);
        return;
      }
      for (const [key, nested] of Object.entries(item)) {
        if (key === "failed_instances") {
          let failures = nested;
          if (typeof failures === "string") {
            try { failures = JSON.parse(failures); } catch { /* validate below */ }
          }
          const count = typeof failures === "number" ? failures :
            Array.isArray(failures) ? failures.length : NaN;
          if (!Number.isSafeInteger(count) || count < 0) {
            throw new BrokerError(422, "failed instance count is invalid.");
          }
          iterationCounts.push(count);
        } else {
          collectFailures(nested);
        }
      }
    };
    collectFailures(iteration);
    return iterationCounts.length ? [Math.max(...iterationCounts)] : [];
  });
  const hasRawFailures = failureEntries.length > 0;
  const rawFailureCount = failureEntries.reduce((total, count) => total + count, 0);
  const claims = [
    value.num_failed_instances,
    value.failed_instances,
    details.num_failed_instances,
    details.failed_instances,
    ...value.evaluation_results.map((item: Record<string, any>) =>
      item?.score_details?.details?.num_failed_instances),
  ].filter((item) => item !== undefined);
  const normaliseCount = (item: unknown): number => {
    const count = typeof item === "number" ? item : Array.isArray(item) ? item.length :
      typeof item === "string" && item.trim() ? Number(item) : NaN;
    if (!Number.isSafeInteger(count) || count < 0) throw new BrokerError(422, "failed instance count is invalid.");
    return count;
  };
  const metricCount = claims.length ? normaliseCount(claims[0]) : undefined;
  const failed = hasRawFailures ? rawFailureCount :
    normaliseCount(details.num_failed_instances ?? metricCount ?? 0);
  if (claims.some((item) => normaliseCount(item) !== failed)) {
    throw new BrokerError(422, "failed instance count is inconsistent with raw results.");
  }
  const warnings: string[] = failed ? [`${failed} failed instances`] : [];
  if (missingRevision) warnings.push("Legacy record is missing model_info.revision");
  const group = languageGroup(expected.language);
  let languages: unknown;
  try { languages = typeof details.languages === "string" ? JSON.parse(details.languages) : details.languages; } catch { throw new BrokerError(422, "additional_details.languages is not valid JSON."); }
  if (!Array.isArray(languages) || !languages.length || !languages.every((item) => typeof item === "string" && group !== null && GROUPS[group].includes(item))) throw new BrokerError(422, "record languages must be a non-empty subset of the leased language group.");
  const recordLanguage = details.language ?? value.language;
  if (recordLanguage !== undefined && (typeof recordLanguage !== "string" || !languages.includes(recordLanguage))) throw new BrokerError(422, "record language is inconsistent with languages.");
  const normaliseBool = (item: unknown): boolean | null | undefined => item === null || item === undefined ? null : item === true || item === "true" ? true : item === false || item === "false" ? false : undefined;
  const split = normaliseBool(details.validation_split); const shot = normaliseBool(details.few_shot);
  if (split === undefined || shot === undefined) throw new BrokerError(422, "validation_split and few_shot must be boolean or null.");
  for (const result of value.evaluation_results) {
    if (!result || typeof result !== "object") throw new BrokerError(422, "evaluation_results contains an invalid item.");
    const evaluation = result as Record<string, any>; const scoreDetails = evaluation.score_details as Record<string, any>;
    const evaluationName = evaluation.evaluation_name ?? evaluation.metric_name ?? evaluation.name;
    if (typeof evaluationName !== "string" || !evaluationName || !scoreDetails || typeof scoreDetails !== "object") throw new BrokerError(422, "every evaluation result needs a metric and score details.");
    const source = evaluation.source_data as Record<string, any> | undefined;
    const sourceDataset = source?.dataset_name ?? source?.dataset ?? source?.name;
    const globalSpeed = /(^|[_ -])speed($|[_ -])/i.test(evaluationName) && (!sourceDataset || /global.*speed|^global$/i.test(sourceDataset));
    if (!source || !globalSpeed && sourceDataset !== details.dataset) throw new BrokerError(422, "evaluation source_data is inconsistent with the record dataset.");
    const metric = evaluation.metric_config as Record<string, any> | undefined;
    if (!metric || typeof metric.lower_is_better !== "boolean" || metric.score_type !== undefined && typeof metric.score_type !== "string") throw new BrokerError(422, "every evaluation result needs a valid metric config.");
    for (const bound of [metric.min_score, metric.max_score]) if (bound !== undefined && bound !== null && (typeof bound === "number" && !Number.isFinite(bound) || typeof bound !== "number" && bound !== "Infinity" && bound !== "-Infinity")) throw new BrokerError(422, "metric bounds must be finite numbers or explicit infinities.");
    const score = scoreDetails.score;
    if (typeof score !== "number" || !Number.isFinite(score)) throw new BrokerError(422, "every evaluation score must be finite.");
    if (typeof metric.min_score === "number" && score < metric.min_score || typeof metric.max_score === "number" && score > metric.max_score) warnings.push(`${evaluationName}: score ${score} is outside declared metric bounds`);
    const uncertainty = scoreDetails.uncertainty as Record<string, any> | undefined;
    const interval = uncertainty?.confidence_interval as Record<string, any> | undefined;
    if (uncertainty && typeof uncertainty !== "object") throw new BrokerError(422, "evaluation uncertainty is invalid.");
    if (uncertainty && Object.values(uncertainty).some((item) => typeof item === "number" && !Number.isFinite(item))) throw new BrokerError(422, "evaluation uncertainty is invalid.");
    if (interval && (typeof interval.lower !== "number" || typeof interval.upper !== "number" || !Number.isFinite(interval.lower) || !Number.isFinite(interval.upper) || interval.lower > interval.upper || score < interval.lower || score > interval.upper)) throw new BrokerError(422, "evaluation uncertainty is invalid.");
  }
  if (!value.evaluation_results.length) throw new BrokerError(422, "record contains no results.");
  return { identity: JSON.stringify([expected.modelId, details.dataset, split, shot]), failed, warnings };
}

function parseCanaryEvidence(value: unknown): unknown {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    throw new BrokerError(422, "contamination_canary_evidence is not valid JSON.");
  }
}

let stagingMetadata: { bucket: string; private: boolean; checkedAt: number } | null = null;
export async function uploadStaging(path: string, content: string): Promise<void> {
  const bucket = env("HF_STAGING_BUCKET"); const token = env("HF_TOKEN");
  if (!/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(bucket)) throw new ConfigurationError("HF_STAGING_BUCKET must be a namespace/bucket id.");
  if (!stagingMetadata || stagingMetadata.bucket !== bucket || Date.now() - stagingMetadata.checkedAt > 60_000) {
    const info = await fetchWithRetry(`https://huggingface.co/api/buckets/${bucket}`, { headers: { accept: "application/json", authorization: `Bearer ${token}` } });
    if (!info.ok) throw new BrokerError(502, `Hugging Face bucket lookup failed with HTTP ${info.status}.`);
    const metadata = await info.json() as { private?: boolean };
    stagingMetadata = { bucket, private: metadata.private === true, checkedAt: Date.now() };
  }
  if (!stagingMetadata.private) throw new ConfigurationError("HF_STAGING_BUCKET is not private; refusing to upload worker results.");
  try {
    await uploadFile({ repo: `buckets/${bucket}`, file: { path, content: new Blob([content], { type: "application/json" }) }, accessToken: token });
  } catch (error) { throw new BrokerError(502, `Hugging Face bucket upload failed: ${error instanceof Error ? error.message : "unknown error"}`); }
}
