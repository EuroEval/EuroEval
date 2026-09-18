import {
  CANARY_CORPUS_REVISION,
  CANARY_CORPUS_SHA256,
  BrokerError,
  ConfigurationError,
  authenticate,
  brokerErrorBody,
  enforceRateLimit,
  fetchCanaryCorpus,
  getLeaseById,
  json,
  method,
  readJson,
  requireProtocol,
} from "./_lib.js";

export const config = { runtime: "edge" };

export async function fetch(req: Request): Promise<Response> {
  const rejected = method(req);
  if (rejected) return rejected;
  try {
    const worker = await authenticate(req);
    await enforceRateLimit(`euroeval:worker:limit:canary-corpus:${worker.hash}`, 20, 3600);
    const body = await readJson(req, 64 * 1024);
    requireProtocol(body);
    for (const field of ["lease_id", "corpus_revision", "corpus_sha256"]) {
      if (typeof body[field] !== "string") throw new BrokerError(400, `${field} is required.`);
    }
    const lease = await getLeaseById(body.lease_id as string);
    if (
      !lease || lease.released || Date.parse(lease.expires_at) <= Date.now() ||
      lease.contributor.toLowerCase() !== worker.contributor.toLowerCase()
    ) {
      throw new BrokerError(409, "Lease is absent or belongs to another contributor.");
    }
    const instruction = lease.contamination_canary;
    if (
      !instruction || instruction.status !== "required" || lease.model_type !== "generative" ||
      body.corpus_revision !== CANARY_CORPUS_REVISION ||
      body.corpus_sha256 !== CANARY_CORPUS_SHA256
    ) throw new BrokerError(409, "Lease does not require this canary corpus.");
    const corpus = await fetchCanaryCorpus();
    return json(200, {
      protocol_version: "volunteer-worker/v1",
      corpus_revision: CANARY_CORPUS_REVISION,
      corpus_sha256: CANARY_CORPUS_SHA256,
      corpus_jsonl: corpus,
    }, { "Cache-Control": "no-store" });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, brokerErrorBody(error, "Unable to deliver canary corpus."), { "Cache-Control": "no-store" });
  }
}

export default fetch;
