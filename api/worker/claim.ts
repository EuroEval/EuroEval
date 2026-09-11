import {
  BrokerError, ConfigurationError, GROUPS, PROTOCOL_VERSION, VM_MARKER_RE, VOLUNTEER_MARKER_RE,
  acquireRenewableIssueMutex, assignIssue, authenticate, claimableLanguages, env, expectedScope,
  extractModelId, enforceRateLimit, fitsGpu, fetchIssue, json, languageGroup, leaseTtl, selectedGpu,
  listOpenIssues, method,
  patchIssue, putLease, randomToken, readJson, deleteLease,
  getLeaseForIssue, reclaimExpiredLease,
  parseVolunteerMarker, resolveModel, selectedLanguages,
  requireProtocol, signVolunteerMarker, verifyVolunteerMarker, markerSecret, replaceVolunteerMarker,
} from "./_lib.ts";
import type { Lease, VolunteerLeaseMarker } from "./_lib.ts";

export const config = { runtime: "edge" };

function activeMarker(marker: VolunteerLeaseMarker | null): VolunteerLeaseMarker | null {
  if (!marker) return null;
  const leases = marker.leases.filter((lease) => Date.parse(lease.expires_at) > Date.now());
  return { ...marker, leases };
}

function validHardware(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const hardware = value as Record<string, unknown>;
  const freeDisk = hardware.free_disk_bytes;
  const utilisation = hardware.gpu_memory_utilisation;
  if (typeof utilisation !== "number" || !Number.isFinite(utilisation) || utilisation <= 0 || utilisation > 1 ||
      typeof hardware.architecture !== "string" || !hardware.architecture.trim() ||
      typeof freeDisk !== "number" || !Number.isSafeInteger(freeDisk) || freeDisk < 0 ||
      !Number.isSafeInteger(hardware.selected_gpu_index) || (hardware.selected_gpu_index as number) < 0 ||
      typeof hardware.selected_gpu_uuid !== "string" || !hardware.selected_gpu_uuid ||
      !Array.isArray(hardware.gpus) || !hardware.gpus.length) return false;
  return hardware.gpus.every((gpu) => {
    if (!gpu || typeof gpu !== "object") return false;
    const item = gpu as Record<string, unknown>;
    const free = item.free_memory_bytes; const total = item.total_memory_bytes;
    return typeof item.name === "string" && typeof item.uuid === "string" &&
      typeof free === "number" && Number.isSafeInteger(free) && free >= 0 &&
      typeof total === "number" && Number.isSafeInteger(total) && total > 0 && free <= total &&
      (item.index === undefined || typeof item.index === "number" && Number.isSafeInteger(item.index) && item.index >= 0);
  }) && selectedGpu(hardware) !== null;
}

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const identity = await authenticate(req); await enforceRateLimit(`euroeval:worker:limit:claim:${identity.hash}`, 120, 3600);
    const body = await readJson(req, 32 * 1024); requireProtocol(body);
    if (!validHardware(body.hardware)) throw new BrokerError(400, "hardware must contain well-formed GPUs and free disk.");
    if (typeof body.worker_version !== "string" || !body.worker_version.trim()) throw new BrokerError(400, "worker_version is required.");
    markerSecret();
    const imageDigest = env("VOLUNTEER_WORKER_IMAGE_DIGEST");
    const euroevalVersion = env("EUROEVAL_VERSION").replace(/\.dev$/, ".dev0"); const coordinator = env("WORKER_COORDINATOR_LOGIN");
    const requiredWorkerVersion = env("VOLUNTEER_WORKER_VERSION");
    if (body.worker_version !== requiredWorkerVersion) throw new BrokerError(422, "Unsupported worker version.");
    const requestedLanguage = typeof body.language === "string" ? body.language : null;
    const issues = await listOpenIssues();
    for (const listed of issues) {
      const modelId = extractModelId(listed.title, listed.body); if (!modelId) continue;
      const languages = selectedLanguages(listed.body);
      if (!languages.length || requestedLanguage && !languages.includes(requestedLanguage)) continue;
      if (listed.body && VM_MARKER_RE.test(listed.body)) continue;
      if (listed.assignees?.some((item) => item.login !== coordinator)) continue;

      // Resolve immutable metadata before taking the short issue mutex. Hub latency
      // must never hold the lock used by another worker claiming another language.
      let model; try { model = await resolveModel(modelId); } catch (error) {
        if (error instanceof BrokerError && error.status === 422) continue; throw error;
      }
      const selectedHardwareGpu = selectedGpu(body.hardware);
      if (!selectedHardwareGpu || !fitsGpu(model, body.hardware, selectedHardwareGpu)) continue;
      const mutex = await acquireRenewableIssueMutex(listed.number); if (!mutex) continue;
      try {
        const snapshot = await fetchIssue(listed.number);
        if (snapshot.state !== "open" || snapshot.assignees?.some((item) => item.login !== coordinator) || snapshot.body && VM_MARKER_RE.test(snapshot.body)) continue;
        if (extractModelId(snapshot.title, snapshot.body) !== modelId) continue;
        const markerPresent = VOLUNTEER_MARKER_RE.test(snapshot.body || "");
        const marker = parseVolunteerMarker(snapshot.body);
        if (markerPresent && (!marker || !(await verifyVolunteerMarker(snapshot.number, marker)))) continue;
        const current = activeMarker(marker);
        const selected = selectedLanguages(snapshot.body);
        const available = claimableLanguages(selected, current);
        const language = requestedLanguage || available[0];
        if (!language || !available.includes(language)) continue;
        const group = languageGroup(language); if (!group || !GROUPS[group]) continue;
        const staleLease = await getLeaseForIssue(snapshot.number, language);
        if (staleLease && Date.parse(staleLease.expires_at) <= Date.now() && !await reclaimExpiredLease(staleLease)) continue;
        let trusted;
        try { trusted = expectedScope(euroevalVersion, model.model_type, language); }
        catch (error) { if (error instanceof BrokerError && error.status === 422) continue; throw error; }
        const expiresAt = new Date(Date.now() + leaseTtl() * 1000).toISOString();
        const lease: Lease = {
          issue_number: snapshot.number, language, worker: identity.hash.slice(0, 24), contributor: identity.contributor,
          model_id: model.id, model_revision: model.revision, euroeval_version: euroevalVersion,
          image_digest: imageDigest,
          worker_version: body.worker_version,
          gpu_memory_utilisation: body.hardware.gpu_memory_utilisation as number,
          selected_gpu_index: body.hardware.selected_gpu_index as number,
          selected_gpu_uuid: body.hardware.selected_gpu_uuid as string,
          expires_at: expiresAt,
          lease_id: randomToken(18),
          model_type: model.model_type,
          model_metadata: model.model_metadata,
          expected_scope: {
            policy_version: trusted.policy_version,
            language_group: trusted.language_group,
            identity_suffixes: [...trusted.identity_suffixes],
            count: trusted.identity_suffixes.length,
            warnings: trusted.warnings || [],
          },
        };
        if (!(await putLease(lease))) continue;
        const nextMarker: VolunteerLeaseMarker = {
          protocol_version: PROTOCOL_VERSION, coordinator, submission: "active",
          leases: [...(current?.leases || []), { lease_id: lease.lease_id, language, worker: lease.worker, contributor: identity.contributor, expires_at: expiresAt }],
          ...(current?.submissions ? { submissions: current.submissions } : {}),
          ...(current?.completed_languages ? { completed_languages: current.completed_languages } : {}),
        };
        try {
          const signedMarker = await signVolunteerMarker(snapshot.number, nextMarker);
          await mutex.assertOwned();
          await patchIssue(snapshot.number, replaceVolunteerMarker(snapshot.body || "", signedMarker));
          await mutex.assertOwned();
          await assignIssue(snapshot.number, coordinator);
          const after = await fetchIssue(snapshot.number);
          const afterMarker = parseVolunteerMarker(after.body);
          if (VM_MARKER_RE.test(after.body || "") || !afterMarker ||
              !(await verifyVolunteerMarker(snapshot.number, afterMarker)) ||
              !afterMarker.leases.some((item) => item.lease_id === lease.lease_id) ||
              !(after.assignees || []).some((item) => item.login === coordinator)) {
            throw new BrokerError(409, "GitHub ownership fence lost during claim.");
          }
        } catch (error) {
          // Never restore a stale body: another lease may have been added while
          // GitHub was unavailable. Remove only our reservation when still present.
          const live = await fetchIssue(snapshot.number).catch(() => null);
          const liveMarker = live ? parseVolunteerMarker(live.body) : null;
          if (live && liveMarker?.leases.some((item) => item.lease_id === lease.lease_id)) {
            await mutex.assertOwned();
            await patchIssue(snapshot.number, replaceVolunteerMarker(live.body || "", await signVolunteerMarker(live.number, {
              ...liveMarker, leases: liveMarker.leases.filter((item) => item.lease_id !== lease.lease_id),
            }))).catch(() => undefined);
          }
          await deleteLease(lease).catch(() => undefined); throw error;
        }
        return json(200, {
          protocol_version: PROTOCOL_VERSION, lease_id: lease.lease_id, issue_number: lease.issue_number,
          language, model_id: model.id, model_revision: model.revision, euroeval_version: euroevalVersion,
          image_digest: imageDigest, worker_version: lease.worker_version, expires_at: expiresAt,
          selected_gpu_index: lease.selected_gpu_index, selected_gpu_uuid: lease.selected_gpu_uuid,
          model_type: lease.model_type, model_metadata: lease.model_metadata,
          expected_scope: lease.expected_scope,
        });
      } finally { await mutex.release(); }
    }
    return json(200, { protocol_version: PROTOCOL_VERSION, status: "no_work" });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to claim an evaluation." });
  }
}
