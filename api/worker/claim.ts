import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, VM_MARKER_RE, assignIssue, authenticate, env,
  extractModelId, fitsGpu, fetchIssue, json, leaseTtl, method, patchIssue, putLease,
  randomToken, readJson, redisGet, replaceVolunteerMarker, issueLeaseKey, listOpenIssues,
  parseVolunteerMarker, resolveModel, selectedLanguages, VolunteerLeaseMarker, Lease,
} from "./_lib";

export const config = { runtime: "edge" };

function activeMarker(marker: VolunteerLeaseMarker | null): VolunteerLeaseMarker | null {
  if (!marker) return null;
  const leases = marker.leases.filter((lease) => lease.expires_at > Date.now());
  return leases.length ? { winner: marker.winner, leases } : null;
}

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const identity = await authenticate(req);
    const body = await readJson(req, 32 * 1024);
    const hardware = body.hardware;
    if (!hardware || typeof hardware !== "object" || Array.isArray(hardware)) throw new BrokerError(400, "hardware with free_vram_bytes (or free_vram_gib) is required.");
    if (typeof body.worker_version !== "string" || !body.worker_version.trim()) throw new BrokerError(400, "worker_version is required.");
    const image = env("VOLUNTEER_WORKER_IMAGE");
    const coordinator = env("WORKER_COORDINATOR_LOGIN");
    const requestedLanguage = typeof body.language === "string" ? body.language : null;
    const issues = await listOpenIssues();
    for (const snapshot of issues) {
      const modelId = extractModelId(snapshot.title, snapshot.body);
      if (!modelId) continue;
      const languages = selectedLanguages(snapshot.body);
      if (!languages.length || (requestedLanguage && !languages.includes(requestedLanguage))) continue;
      const markerPresent = !!snapshot.body?.match(/euroeval-volunteer-worker:v1/i);
      const marker = parseVolunteerMarker(snapshot.body);
      // Never overwrite a malformed broker marker: it may represent an active owner.
      if (markerPresent && !marker) continue;
      if (snapshot.body && VM_MARKER_RE.test(snapshot.body)) continue;
      const assignees = snapshot.assignees?.map((item) => item.login) || [];
      if (assignees.some((login) => login !== coordinator)) continue;
      if (assignees.includes(coordinator) && !marker) continue;
      const current = activeMarker(marker);
      const available = languages.filter((language) => {
        if (current?.leases.some((lease) => lease.language === language)) return false;
        return true;
      });
      const language = requestedLanguage || available[0];
      if (!language || !available.includes(language)) continue;
      const existingLease = await redisGet<Lease>(issueLeaseKey(snapshot.number, language));
      if (existingLease && existingLease.expires_at > Date.now()) continue;
      let model;
      try {
        model = await resolveModel(modelId);
      } catch (error) {
        // An unsupported model is not a broker outage; let another queue item
        // be considered. Configuration and upstream failures still surface.
        if (error instanceof BrokerError && error.status === 422) continue;
        throw error;
      }
      if (!fitsGpu(model, hardware as Record<string, unknown>)) continue;
      const ttl = leaseTtl();
      const lease: Lease = {
        issue_number: snapshot.number, language, worker: identity.hash.slice(0, 24), contributor: identity.contributor,
        model_id: model.id, revision: model.revision, image, worker_version: body.worker_version,
        expires_at: Date.now() + ttl * 1000, lease_id: randomToken(18),
      };
      const nextMarker: VolunteerLeaseMarker = {
        winner: current?.winner || identity.contributor,
        leases: [...(current?.leases || []), { language, worker: lease.worker, contributor: identity.contributor, expires_at: lease.expires_at }],
      };
      const originalBody = snapshot.body || "";
      await patchIssue(snapshot.number, replaceVolunteerMarker(originalBody, nextMarker));
      try { await assignIssue(snapshot.number, coordinator); } catch (error) {
        await patchIssue(snapshot.number, originalBody).catch(() => undefined);
        throw error;
      }
      // GitHub updates are not transactions. This read-after-write is the ownership fence.
      const after = await fetchIssue(snapshot.number);
      const afterMarker = parseVolunteerMarker(after.body);
      if (!afterMarker?.leases.some((item) => item.language === language && item.worker === lease.worker) ||
          !(after.assignees || []).some((item) => item.login === coordinator)) {
        continue;
      }
      if (!(await putLease(lease))) continue;
      return json(200, {
        protocol_version: PROTOCOL_VERSION, lease_id: lease.lease_id,
        issue_number: lease.issue_number, language, contributor: identity.contributor,
        model: { id: model.id, revision: model.revision, config: model.config, weight_bytes: model.weight_bytes },
        image, worker_version: lease.worker_version, expires_at: lease.expires_at,
        manifest: { language, model_id: model.id, revision: model.revision },
      });
    }
    return json(409, { error: "No compatible unleased evaluation request is currently available." });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { error: error instanceof Error ? error.message : "Unable to claim an evaluation." });
  }
}
