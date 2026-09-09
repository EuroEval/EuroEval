import {
  BrokerError, ConfigurationError, GROUPS, PROTOCOL_VERSION, VM_MARKER_RE,
  acquireIssueMutex, assignIssue, authenticate, env, expectedScope, extractModelId,
  enforceRateLimit, fitsGpu, fetchIssue, json, languageGroup, leaseTtl, listOpenIssues, method,
  patchIssue, putLease, randomToken, readJson, releaseIssueMutex, deleteLease,
  parseVolunteerMarker, resolveModel, selectedLanguages, VolunteerLeaseMarker, Lease,
  requireProtocol,
} from "./_lib";

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
  if (typeof hardware.architecture !== "string" || !hardware.architecture.trim() ||
      typeof freeDisk !== "number" || !Number.isSafeInteger(freeDisk) || freeDisk < 0 ||
      !Array.isArray(hardware.gpus) || !hardware.gpus.length) return false;
  return hardware.gpus.every((gpu) => {
    if (!gpu || typeof gpu !== "object") return false;
    const item = gpu as Record<string, unknown>;
    const free = item.free_memory_bytes; const total = item.total_memory_bytes;
    return typeof item.name === "string" && typeof item.uuid === "string" &&
      typeof free === "number" && Number.isSafeInteger(free) && free >= 0 &&
      typeof total === "number" && Number.isSafeInteger(total) && total > 0 && free <= total;
  });
}

export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    const identity = await authenticate(req); await enforceRateLimit(`euroeval:worker:limit:claim:${identity.hash}`, 120, 3600);
    const body = await readJson(req, 32 * 1024); requireProtocol(body);
    if (!validHardware(body.hardware)) throw new BrokerError(400, "hardware must contain well-formed GPUs and free disk.");
    if (typeof body.worker_version !== "string" || !body.worker_version.trim()) throw new BrokerError(400, "worker_version is required.");
    const imageDigest = env("VOLUNTEER_WORKER_IMAGE_DIGEST");
    const euroevalVersion = env("EUROEVAL_VERSION"); const coordinator = env("WORKER_COORDINATOR_LOGIN");
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
      if (!fitsGpu(model, body.hardware)) continue;
      const mutex = await acquireIssueMutex(listed.number); if (!mutex) continue;
      try {
        const snapshot = await fetchIssue(listed.number);
        if (snapshot.state !== "open" || snapshot.assignees?.some((item) => item.login !== coordinator) || snapshot.body && VM_MARKER_RE.test(snapshot.body)) continue;
        const markerPresent = !!snapshot.body?.match(/euroeval-volunteer-worker:v1/i);
        const marker = parseVolunteerMarker(snapshot.body);
        if (markerPresent && !marker) continue;
        const current = activeMarker(marker);
        const selected = selectedLanguages(snapshot.body);
        const available = selected.filter((language) =>
          !current?.leases.some((item) => item.language === language) &&
          !(current?.completed_languages || []).includes(language) &&
          !(current?.submissions || []).some((item) => item.language === language));
        const language = requestedLanguage || available[0];
        if (!language || !available.includes(language)) continue;
        const group = languageGroup(language); if (!group || !GROUPS[group]) continue;
        let trusted;
        try { trusted = expectedScope(euroevalVersion, model.model_profile, language); }
        catch (error) { if (error instanceof BrokerError && error.status === 422) continue; throw error; }
        const expiresAt = new Date(Date.now() + leaseTtl() * 1000).toISOString();
        const lease: Lease = {
          issue_number: snapshot.number, language, worker: identity.hash.slice(0, 24), contributor: identity.contributor,
          model_id: model.id, model_revision: model.revision, euroeval_version: euroevalVersion,
          image_digest: imageDigest, worker_version: body.worker_version, expires_at: expiresAt,
          lease_id: randomToken(18), model_profile: model.model_profile,
          expected_scope: { policy_version: trusted.policy_version, language_group: group,
            identity_suffixes: [...trusted.identity_suffixes], count: trusted.identity_suffixes.length,
            warnings: trusted.warnings || [] },
        };
        if (!(await putLease(lease))) continue;
        const nextMarker: VolunteerLeaseMarker = {
          protocol_version: PROTOCOL_VERSION, coordinator, submission: "active",
          leases: [...(current?.leases || []), { lease_id: lease.lease_id, language, worker: lease.worker, contributor: identity.contributor, expires_at: expiresAt }],
          ...(current?.submissions ? { submissions: current.submissions } : {}),
          ...(current?.completed_languages ? { completed_languages: current.completed_languages } : {}),
        };
        try {
          await patchIssue(snapshot.number, replaceMarker(snapshot.body || "", nextMarker));
          await assignIssue(snapshot.number, coordinator);
          const after = await fetchIssue(snapshot.number);
          const afterMarker = parseVolunteerMarker(after.body);
          if (VM_MARKER_RE.test(after.body || "") || !afterMarker?.leases.some((item) => item.lease_id === lease.lease_id) ||
              !(after.assignees || []).some((item) => item.login === coordinator)) {
            throw new BrokerError(409, "GitHub ownership fence lost during claim.");
          }
        } catch (error) {
          // Never restore a stale body: another lease may have been added while
          // GitHub was unavailable. Remove only our reservation when still present.
          const live = await fetchIssue(snapshot.number).catch(() => null);
          const liveMarker = live ? parseVolunteerMarker(live.body) : null;
          if (live && liveMarker?.leases.some((item) => item.lease_id === lease.lease_id)) {
            await patchIssue(snapshot.number, replaceMarker(live.body || "", {
              ...liveMarker, leases: liveMarker.leases.filter((item) => item.lease_id !== lease.lease_id),
            })).catch(() => undefined);
          }
          await deleteLease(lease).catch(() => undefined); throw error;
        }
        return json(200, {
          protocol_version: PROTOCOL_VERSION, lease_id: lease.lease_id, issue_number: lease.issue_number,
          language, model_id: model.id, model_revision: model.revision, euroeval_version: euroevalVersion,
          image_digest: imageDigest, worker_version: lease.worker_version, expires_at: expiresAt,
          model_profile: lease.model_profile, expected_scope: lease.expected_scope,
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
