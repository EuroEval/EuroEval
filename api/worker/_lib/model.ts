import generatedScopePolicy from "../scope-policy.json" with { type: "json" };
declare const process: { env: Record<string, string | undefined> };
import { BrokerError, ConfigurationError, GROUPS } from "./protocol.ts";
import type { ModelMetadataEvidence } from "./protocol.ts";
import { fetchWithRetry } from "./http.ts";

export type ModelType = "encoder" | "generative";

export interface ResolvedModel {
  id: string;
  revision: string;
  config: Record<string, unknown>;
  weight_bytes: number;
  repo_bytes: number;
  model_type: ModelType;
  model_metadata: ModelMetadataEvidence;
}

export async function resolveModel(modelId: string): Promise<ResolvedModel> {
  const encoded = modelId.split("/").map(encodeURIComponent).join("/");
  const infoResponse = await fetchWithRetry(`https://huggingface.co/api/models/${encoded}?blobs=true`);
  if (!infoResponse.ok) throw new BrokerError(422, `Hugging Face model ${modelId} could not be resolved.`);
  const info = await infoResponse.json() as {
    id?: string; sha?: string; gated?: boolean | string; private?: boolean;
    pipeline_tag?: unknown;
    siblings?: Array<{ rfilename?: string; size?: number }>;
  };
  if (typeof info.id !== "string" || info.id.toLowerCase() !== modelId.toLowerCase()) {
    throw new BrokerError(422, "Hugging Face returned incomplete model metadata.");
  }
  if (typeof info.private !== "boolean" ||
      typeof info.gated !== "boolean" && info.gated !== "false") {
    throw new BrokerError(422, "Hugging Face omitted public access metadata.");
  }
  const gated = info.gated === true;
  if (info.private || gated) {
    throw new BrokerError(422, "The model must be public and ungated.");
  }
  if (typeof info.pipeline_tag !== "string" || !info.pipeline_tag.trim()) {
    throw new BrokerError(422, "Hugging Face omitted an unambiguous pipeline tag.");
  }
  if (!info.sha || !/^[0-9a-f]{40}$/.test(info.sha)) {
    throw new BrokerError(422, "Hugging Face did not provide an immutable model revision.");
  }
  if (!Array.isArray(info.siblings) || info.siblings.some((file) =>
      !file || typeof file.rfilename !== "string" || !file.rfilename)) {
    throw new BrokerError(422, "Hugging Face response omitted repository files.");
  }
  const files = info.siblings;
  const weights = files.filter((file) => file.rfilename!.toLowerCase().endsWith(".safetensors"));
  const unsafe = files.some((file) => /\.(bin|pt|pth|ckpt|gguf|onnx|h5|msgpack)$/i.test(file.rfilename || ""));
  if (!weights.length || unsafe || files.some((file) => file.rfilename?.toLowerCase().endsWith(".py"))) {
    throw new BrokerError(422, "The model must use safetensors only and contain no custom repository Python.");
  }
  const configResponse = await fetchWithRetry(`https://huggingface.co/${encoded}/raw/${encodeURIComponent(info.sha)}/config.json`);
  if (!configResponse.ok) throw new BrokerError(422, "The model config could not be resolved at its immutable revision.");
  const config = await configResponse.json() as Record<string, unknown>;
  if (config.auto_map || config.custom_code || config.trust_remote_code) {
    throw new BrokerError(422, "Models with auto_map or custom code are not supported.");
  }
  const weightBytes = weights.reduce((total, file) => total +
    (typeof file.size === "number" && Number.isSafeInteger(file.size) ? file.size : 0), 0);
  const repoBytes = files.reduce((total, file) => total +
    (typeof file.size === "number" && Number.isSafeInteger(file.size) && file.size >= 0 ? file.size : 0), 0);
  if (!weightBytes || !repoBytes || files.some((file) => typeof file.size !== "number" ||
      !Number.isSafeInteger(file.size) || file.size < 0)) {
    throw new BrokerError(422, "Hugging Face did not provide complete model file sizes for a fit estimate.");
  }
  if (typeof config.model_type !== "string" || !config.model_type.trim()) {
    throw new BrokerError(422, "The model config omitted model_type.");
  }
  const architectures = config.architectures;
  if (!Array.isArray(architectures) || !architectures.length ||
      architectures.some((item) => typeof item !== "string" || !item)) {
    throw new BrokerError(422, "The model config has no unambiguous architectures.");
  }
  const encoderDecoder = config.is_encoder_decoder;
  if (encoderDecoder !== undefined && typeof encoderDecoder !== "boolean") {
    throw new BrokerError(422, "The model config has an invalid encoder-decoder flag.");
  }
  const modelType = modelTypeFor(info.pipeline_tag, encoderDecoder);
  if (!modelType) throw new BrokerError(422, "The model has contradictory capability metadata.");
  const modelMetadata = {
    pipeline_tag: info.pipeline_tag,
    architectures: [...architectures] as string[],
    model_type: modelType,
    is_encoder_decoder: encoderDecoder === undefined ? null : encoderDecoder,
  } satisfies ModelMetadataEvidence;
  return { id: modelId, revision: info.sha, config, weight_bytes: weightBytes,
    repo_bytes: repoBytes, model_type: modelType, model_metadata: modelMetadata };
}

export function selectedGpu(hardware: Record<string, unknown>): Record<string, unknown> | null {
  if (!Array.isArray(hardware.gpus) || !Number.isSafeInteger(hardware.selected_gpu_index) ||
      (hardware.selected_gpu_index as number) < 0 || typeof hardware.selected_gpu_uuid !== "string" ||
      !(hardware.selected_gpu_uuid as string)) return null;
  const index = hardware.selected_gpu_index as number;
  const hasExplicitIndexes = hardware.gpus.some((gpu) =>
    !!gpu && typeof gpu === "object" && (gpu as Record<string, unknown>).index !== undefined);
  const gpu = hasExplicitIndexes
    ? hardware.gpus.find((item) => !!item && typeof item === "object" &&
      (item as Record<string, unknown>).index === index)
    : hardware.gpus[index];
  if (!gpu || typeof gpu !== "object") return null;
  const item = gpu as Record<string, unknown>;
  return item.uuid === hardware.selected_gpu_uuid ? item : null;
}

export function fitsGpu(model: ResolvedModel, hardware: Record<string, unknown>,
  selected?: Record<string, unknown> | null): boolean {
  if (typeof hardware.free_disk_bytes !== "number" || !Number.isFinite(hardware.free_disk_bytes) ||
      hardware.free_disk_bytes < model.repo_bytes) return false;
  const utilisation = hardware.gpu_memory_utilisation === undefined ? 1 : hardware.gpu_memory_utilisation;
  if (typeof utilisation !== "number" || !Number.isFinite(utilisation) || utilisation <= 0 || utilisation > 1) return false;
  if (!Array.isArray(hardware.gpus)) return false;
  const hasSelection = "selected_gpu_index" in hardware || "selected_gpu_uuid" in hardware;
  const gpu = selected === undefined ? (hasSelection ? selectedGpu(hardware) : null) : selected;
  if (hasSelection && !gpu) return false;
  const candidates = gpu ? [gpu] : hardware.gpus;
  const largestFree = Math.max(...candidates.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const free = (item as Record<string, unknown>).free_memory_bytes;
    return typeof free === "number" && Number.isFinite(free) && free > 0 ? [free] : [];
  }), 0);
  return model.weight_bytes * 1.35 <= largestFree * utilisation;
}

const GENERATIVE_PIPELINE_TAGS = new Set([
  "text-generation", "text2text-generation", "image-text-to-text",
  "audio-text-to-text", "video-text-to-text", "any-to-any",
]);

function modelTypeFor(
  pipelineTag: unknown, isEncoderDecoder: unknown,
): ModelType | null {
  if (typeof pipelineTag !== "string" || !pipelineTag.trim()) return null;
  if (isEncoderDecoder !== undefined && typeof isEncoderDecoder !== "boolean") return null;
  const generative = GENERATIVE_PIPELINE_TAGS.has(pipelineTag);
  if (!generative && isEncoderDecoder === true) return null;
  return generative ? "generative" : "encoder";
}

export function languageGroup(language: string): string | null {
  return Object.entries(GROUPS).find(([, codes]) => codes.includes(language))?.[0] || null;
}

type ScopeEntry = { euroeval_version: string; model_type: ModelType; language: string; language_group: string; identity_suffixes: string[]; task_groups: string[]; count?: number; warnings?: string[] };
type ScopePolicy = { policy_version: string; policies: ScopeEntry[] };
function canonicalEuroevalVersion(version: string): string {
  return version.replace(/\.dev$/, ".dev0");
}

export function expectedScope(euroevalVersion: string, modelType: ModelType, language: string): ScopeEntry & { policy_version: string } {
  euroevalVersion = canonicalEuroevalVersion(euroevalVersion);
  const group = languageGroup(language);
  if (!group) throw new BrokerError(422, "The leased language has no trusted language group.");
  if (modelType !== "encoder" && modelType !== "generative") throw new BrokerError(422, "The model has an unsupported EuroEval capability.");
  const raw = process.env.VOLUNTEER_SCOPE_POLICY_JSON || JSON.stringify(generatedScopePolicy);
  let policy: ScopePolicy;
  try { policy = JSON.parse(raw) as ScopePolicy; } catch { throw new ConfigurationError("VOLUNTEER_SCOPE_POLICY_JSON is invalid JSON."); }
  if (typeof policy.policy_version !== "string" || !Array.isArray(policy.policies)) throw new ConfigurationError("VOLUNTEER_SCOPE_POLICY_JSON has an invalid schema.");
  const match = policy.policies.find((item) => item.euroeval_version === euroevalVersion && item.model_type === modelType && item.language === language);
  if (!match || !Array.isArray(match.identity_suffixes) || !match.identity_suffixes.length) throw new BrokerError(422, "This model type and language has no trusted expected scope.");
  if (match.identity_suffixes.some((item) => typeof item !== "string") || new Set(match.identity_suffixes).size !== match.identity_suffixes.length || match.count !== undefined && match.count !== match.identity_suffixes.length) throw new ConfigurationError("Trusted expected scope contains invalid or duplicate identities.");
  if (!Array.isArray(match.task_groups) || !match.task_groups.length || match.task_groups.some((item) => typeof item !== "string" || !item) || new Set(match.task_groups).size !== match.task_groups.length) throw new ConfigurationError("Trusted expected scope contains invalid task groups.");
  if (typeof match.language_group !== "string" || !match.language_group) throw new ConfigurationError("Trusted expected scope has no language_group.");
  if (match.warnings && (!Array.isArray(match.warnings) || match.warnings.some((item) => typeof item !== "string"))) throw new ConfigurationError("Trusted expected scope contains invalid warnings.");
  return { ...match, language: match.language, language_group: match.language_group, policy_version: policy.policy_version, count: match.identity_suffixes.length, task_groups: [...match.task_groups], warnings: match.warnings || [] };
}
