import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, VM_MARKER_RE, acquireIssueMutex,
  assignIssue, authenticate, env, extractModelId, fitsGpu, fetchIssue, json, leaseTtl,
  method, patchIssue, putLease, randomToken, readJson, redisDelete, redisGet,
  releaseIssueMutex, issueLeaseKey, leaseKey, listOpenIssues, parseVolunteerMarker, resolveModel,
  selectedLanguages, VolunteerLeaseMarker, Lease, requireProtocol,
} from "./_lib";

export const config = { runtime: "edge" };

function activeMarker(marker: VolunteerLeaseMarker | null): VolunteerLeaseMarker | null {
  if (!marker) return null;
  const leases = marker.leases.filter((lease) => Date.parse(lease.expires_at) > Date.now());
  return leases.length ? { ...marker, leases } : null;
}

function validHardware(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const hardware = value as Record<string, unknown>;
  return typeof hardware.architecture === "string" && Array.isArray(hardware.gpus) &&
    hardware.gpus.length > 0 && typeof hardware.free_disk_bytes === "number";
}

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const identity = await authenticate(req);
    const body = await readJson(req, 32 * 1024);
    requireProtocol(body);
    if (!validHardware(body.hardware)) throw new BrokerError(400, "hardware is required and must contain GPUs.");
    if (typeof body.worker_version !== "string" || !body.worker_version.trim()) throw new BrokerError(400, "worker_version is required.");
    const imageDigest = env("VOLUNTEER_WORKER_IMAGE_DIGEST");
    const euroevalVersion = env("EUROEVAL_VERSION");
    const coordinator = env("WORKER_COORDINATOR_LOGIN");
    const requestedLanguage = typeof body.language === "string" ? body.language : null;
    const issues = await listOpenIssues();
    for (const listed of issues) {
      const modelId = extractModelId(listed.title, listed.body);
      if (!modelId) continue;
      const languages = selectedLanguages(listed.body);
      if (!languages.length || (requestedLanguage && !languages.includes(requestedLanguage))) continue;
      if (listed.body && VM_MARKER_RE.test(listed.body)) continue;
      if (listed.assignees?.some((item) => item.login !== coordinator)) continue;
      const mutex = await acquireIssueMutex(listed.number);
      if (!mutex) continue;
      let lease: Lease | null = null;
      try {
        const snapshot = await fetchIssue(listed.number);
        const markerPresent = !!snapshot.body?.match(/euroeval-volunteer-worker:v1/i);
        const marker = parseVolunteerMarker(snapshot.body);
        if (markerPresent && !marker) continue;
        const current = activeMarker(marker);
        const available = languages.filter((language) => !current?.leases.some((item) => item.language === language));
        const language = requestedLanguage || available[0];
        if (!language || !available.includes(language) || (snapshot.assignees || []).some((item) => item.login !== coordinator)) continue;
        const existingLease = await redisGet<Lease>(issueLeaseKey(snapshot.number, language));
        if (existingLease && Date.parse(existingLease.expires_at) > Date.now()) continue;
        let model;
        try { model = await resolveModel(modelId); }
        catch (error) { if (error instanceof BrokerError && error.status === 422) continue; throw error; }
        if (!fitsGpu(model, body.hardware)) continue;
        const expiresAt = new Date(Date.now() + leaseTtl() * 1000).toISOString();
        lease = {
          issue_number: snapshot.number, language, worker: identity.hash.slice(0, 24), contributor: identity.contributor,
          model_id: model.id, model_revision: model.revision, euroeval_version: euroevalVersion,
          image_digest: imageDigest, worker_version: body.worker_version, expires_at: expiresAt, lease_id: randomToken(18),
        };
        if (!(await putLease(lease))) continue;
        const nextMarker: VolunteerLeaseMarker = {
          protocol_version: PROTOCOL_VERSION, coordinator, submission: "active",
          leases: [...(current?.leases || []), { lease_id: lease.lease_id, language, worker: lease.worker, contributor: identity.contributor, expires_at: expiresAt }],
        };
        const originalBody = snapshot.body || "";
        try {
          await patchIssue(snapshot.number, replaceMarker(originalBody, nextMarker));
          await assignIssue(snapshot.number, coordinator);
          const after = await fetchIssue(snapshot.number);
          const afterMarker = parseVolunteerMarker(after.body);
          if (!afterMarker?.leases.some((item) => item.lease_id === lease?.lease_id) || !(after.assignees || []).some((item) => item.login === coordinator)) throw new BrokerError(409, "GitHub ownership fence lost during claim.");
        } catch (error) {
          await patchIssue(snapshot.number, originalBody).catch(() => undefined);
          await redisDelete(leaseKey(lease.lease_id)).catch(() => undefined);
          await redisDelete(issueLeaseKey(snapshot.number, language)).catch(() => undefined);
          throw error;
        }
        return json(200, {
          protocol_version: PROTOCOL_VERSION, lease_id: lease.lease_id, issue_number: lease.issue_number,
          language, model_id: model.id, model_revision: model.revision, euroeval_version: euroevalVersion,
          image_digest: imageDigest, worker_version: lease.worker_version, expires_at: expiresAt,
        });
      } finally { await releaseIssueMutex(listed.number, mutex); }
    }
    return json(200, { protocol_version: PROTOCOL_VERSION, status: "no_work" });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to claim an evaluation." });
  }
}

function replaceMarker(body: string, marker: VolunteerLeaseMarker): string {
  const without = body.replace(/<!--[\s\S]*?euroeval-volunteer-worker:v1\s+[\s\S]*?-->/i, "").replace(/\n{3,}/g, "\n\n").trimEnd();
  return `${without}\n\n<!-- euroeval-volunteer-worker:v1 ${JSON.stringify(marker)} -->\n`;
}
