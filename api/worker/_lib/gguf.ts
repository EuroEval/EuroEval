/* Broker-owned GGUF artifact bindings: strict decoding and exact approval lookup. */
import {
  BrokerError, ConfigurationError, optionalEnv, type GgufArtifact, type Lease, type LeaseExecution,
} from "./protocol.js";

const REVISION_RE = /^[0-9a-f]{40}$/;
const SHA256_RE = /^[0-9a-f]{64}$/;
const REPO_ID_RE = /^[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?\/[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?$/;
const GGUF_FILENAME_RE = /^[A-Za-z0-9_][A-Za-z0-9._+-]*\.gguf$/;

const ARTIFACT_FIELDS = "filename,repo_id,revision,sha256,source_repo_id,source_revision";

/** Strictly decoded contents of the checked-in gguf-bindings.json. */
export interface GgufBindings {
  version: 1;
  artifacts: GgufArtifact[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function sortedKeys(value: Record<string, unknown>): string {
  return Object.keys(value).sort().join(",");
}

function artifactField(value: unknown, field: string): string {
  if (typeof value !== "string" || !value) {
    throw new BrokerError(400, `GGUF artifact field '${field}' must be a non-empty string.`);
  }
  return value;
}

/**
 * Strictly decode one GGUF artifact descriptor.
 *
 * Rejects non-objects, missing or extra fields, malformed repo ids, non-hex
 * revisions or hashes, and filenames that are not a single safe `.gguf`
 * basename.
 */
export function decodeGgufArtifact(value: unknown): GgufArtifact {
  if (!isRecord(value)) throw new BrokerError(400, "GGUF artifact must be a JSON object.");
  if (sortedKeys(value) !== ARTIFACT_FIELDS) {
    throw new BrokerError(400, "GGUF artifact must contain exactly the six contract fields.");
  }
  const repoId = artifactField(value.repo_id, "repo_id");
  if (!REPO_ID_RE.test(repoId)) {
    throw new BrokerError(400, "GGUF artifact field 'repo_id' must be an 'org/name' repository id.");
  }
  const revision = artifactField(value.revision, "revision");
  if (!REVISION_RE.test(revision)) {
    throw new BrokerError(400, "GGUF artifact field 'revision' must be 40 lowercase hex characters.");
  }
  const filename = artifactField(value.filename, "filename");
  if (!GGUF_FILENAME_RE.test(filename)) {
    throw new BrokerError(400, "GGUF artifact field 'filename' must be a single safe .gguf basename.");
  }
  const sha256 = artifactField(value.sha256, "sha256");
  if (!SHA256_RE.test(sha256)) {
    throw new BrokerError(400, "GGUF artifact field 'sha256' must be 64 lowercase hex characters.");
  }
  const sourceRepoId = artifactField(value.source_repo_id, "source_repo_id");
  if (!REPO_ID_RE.test(sourceRepoId)) {
    throw new BrokerError(400, "GGUF artifact field 'source_repo_id' must be an 'org/name' repository id.");
  }
  const sourceRevision = artifactField(value.source_revision, "source_revision");
  if (!REVISION_RE.test(sourceRevision)) {
    throw new BrokerError(400, "GGUF artifact field 'source_revision' must be 40 lowercase hex characters.");
  }
  return {
    repo_id: repoId,
    revision,
    filename,
    sha256,
    source_repo_id: sourceRepoId,
    source_revision: sourceRevision,
  };
}

/**
 * Decode the optional lease `execution` field. Absence (null or undefined) is
 * the historical CUDA mode; the only supported kind is `external_server`.
 */
export function decodeLeaseExecution(value: unknown): LeaseExecution | null {
  if (value === undefined || value === null) return null;
  if (!isRecord(value)) throw new BrokerError(400, "Lease execution must be a JSON object.");
  if (sortedKeys(value) !== "artifact,kind") {
    throw new BrokerError(400, "Lease execution must contain exactly 'kind' and 'artifact'.");
  }
  if (value.kind !== "external_server") {
    throw new BrokerError(400, "Lease execution kind is unsupported.");
  }
  return { kind: "external_server", artifact: decodeGgufArtifact(value.artifact) };
}

/**
 * Strictly decode the checked-in bindings document. Malformed configuration
 * fails as a deployment error, never as a worker-facing 400.
 */
export function decodeGgufBindings(value: unknown): GgufBindings {
  if (!isRecord(value) || sortedKeys(value) !== "artifacts,version") {
    throw new ConfigurationError("gguf-bindings must contain exactly 'version' and 'artifacts'.");
  }
  if (value.version !== 1) throw new ConfigurationError("gguf-bindings version must be 1.");
  if (!Array.isArray(value.artifacts) || value.artifacts.length === 0) {
    throw new ConfigurationError("gguf-bindings artifacts must be a nonempty array.");
  }
  const artifacts = value.artifacts.map((entry) => {
    try {
      return decodeGgufArtifact(entry);
    } catch (error) {
      throw new ConfigurationError(`gguf-bindings artifact is invalid: ${error instanceof Error ? error.message : String(error)}`);
    }
  });
  const seen = new Set<string>();
  for (const artifact of artifacts) {
    const key = JSON.stringify([artifact.repo_id, artifact.revision]);
    if (seen.has(key)) throw new ConfigurationError("gguf-bindings contains a duplicate artifact binding.");
    seen.add(key);
  }
  return { version: 1, artifacts };
}

/** Canonical identity of one artifact binding, covering all six fields. */
export function ggufArtifactKey(artifact: GgufArtifact): string {
  return JSON.stringify([
    artifact.repo_id,
    artifact.revision,
    artifact.filename,
    artifact.sha256,
    artifact.source_repo_id,
    artifact.source_revision,
  ]);
}

/**
 * Exact broker approval lookup: a client claim is approved only when every
 * field matches a checked-in binding. Returns the approved artifact, or null
 * when the claim matches no binding.
 */
export function findApprovedGgufArtifact(bindings: GgufBindings, candidate: GgufArtifact): GgufArtifact | null {
  return bindings.artifacts.find((artifact) => ggufArtifactKey(artifact) === ggufArtifactKey(candidate)) ?? null;
}

/**
 * Strictly decode a stored lease's execution discriminator and re-verify that
 * it is still the exact broker-approved binding for the lease identity. A
 * malformed or unapproved stored execution is a deployment failure, never a
 * worker-facing rejection; an omitted execution is the historical CUDA mode.
 */
export function approvedLeaseExecution(lease: Lease): LeaseExecution | null {
  if (lease.execution === undefined || lease.execution === null) return null;
  let execution: LeaseExecution | null;
  try { execution = decodeLeaseExecution(lease.execution); } catch { execution = null; }
  if (!execution) throw new ConfigurationError("Stored lease execution is malformed.");
  const approved = findApprovedGgufArtifact(ggufBindings(), execution.artifact);
  if (!approved) throw new ConfigurationError("Stored lease execution artifact is not broker-approved.");
  if (approved.repo_id.toLowerCase() !== lease.model_id.toLowerCase() ||
      approved.revision !== lease.model_revision) {
    throw new ConfigurationError("Stored lease execution does not match the lease model identity.");
  }
  return { kind: execution.kind, artifact: approved };
}

/**
 * Whether native external-server workers may claim work at all. Off unless
 * explicitly enabled; a disabled gate rejects every external claim.
 */
export function externalWorkersEnabled(): boolean {
  const value = optionalEnv("VOLUNTEER_EXTERNAL_WORKERS_ENABLED", "false").trim().toLowerCase();
  return value === "true" || value === "1";
}

/**
 * The broker's runtime copy of the checked-in api/worker/gguf-bindings.json.
 * Edge endpoints cannot read files or use JSON import attributes, so the
 * binding document is embedded here; tests/api/worker/gguf.test.mjs asserts
 * that this document deep-equals the checked-in JSON, keeping the file the
 * reviewed source of truth.
 */
const bindingsDocument = {
  version: 1,
  artifacts: [
    {
      repo_id: "LiquidAI/LFM3-1.7B-GGUF",
      revision: "cd0c74c575b5c416f3011b5b4c298077f26ca8c2",
      filename: "LFM3-1.7B-Q8_0.gguf",
      sha256: "8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6",
      source_repo_id: "LiquidAI/LFM3-1.7B",
      source_revision: "1da06bec6a819da10b54c67e5c90972b46d81871",
    },
  ],
};

let cachedBindings: GgufBindings | null = null;

/**
 * Decode the checked-in binding configuration once per process. A malformed
 * document raises ConfigurationError, which handlers report as a deployment
 * failure rather than a worker-facing rejection.
 */
export function ggufBindings(): GgufBindings {
  cachedBindings ??= decodeGgufBindings(bindingsDocument);
  return cachedBindings;
}
