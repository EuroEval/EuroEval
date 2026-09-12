import {
  BrokerError, ConfigurationError, GROUPS, PROTOCOL_VERSION, VM_MARKER_RE, VOLUNTEER_MARKER_RE,
  acquireRenewableIssueMutex, assignIssue, assertAssignable, authenticate, brokerErrorBody,
  claimableLanguages, env, expectedScope,
  extractModelId, enforceRateLimit, fitsGpu, fetchIssue, json, languageGroup, leaseTtl, selectedGpu,
  listOpenIssues, method,
  patchIssue, putLease, randomToken, readJson, deleteLease, unassignIssue,
  getLeaseForIssue, reclaimExpiredLease,
  parseVolunteerMarker, resolveModel, selectedLanguages, volunteerAssigneesMatch,
  requireProtocol, signVolunteerMarker, verifyVolunteerMarker, markerSecret, replaceVolunteerMarker,
} from "./_lib.ts";
import type { Lease, VolunteerLeaseMarker } from "./_lib.ts";

export const config = { runtime: "edge" };

function assigneesMatchWithExtras(
  assignees: Array<{ login: string }> | undefined,
  marker: VolunteerLeaseMarker,
  extras: Iterable<string>,
): boolean {
  const actual = new Set((assignees || []).map((assignee) => assignee.login.toLowerCase()));
  const expected = new Set<string>(extras);
  for (const lease of marker.leases) {
    if (Date.parse(lease.expires_at) > Date.now()) expected.add(lease.contributor.toLowerCase());
  }
  for (const submission of marker.submissions || []) {
    if (["submitted", "accepted"].includes(submission.status)) {
      expected.add(submission.verified_contributor.toLowerCase());
    }
  }
  return actual.size === expected.size && [...actual].every((login) => expected.has(login));
}

async function recoverExpiredOwnership(
  issue: Awaited<ReturnType<typeof fetchIssue>>,
  marker: VolunteerLeaseMarker,
  mutex: NonNullable<Awaited<ReturnType<typeof acquireRenewableIssueMutex>>>,
): Promise<{ issue: Awaited<ReturnType<typeof fetchIssue>>; marker: VolunteerLeaseMarker } | null> {
  const now = Date.now();
  const expired = marker.leases.filter((lease) => Date.parse(lease.expires_at) <= now);
  if (!expired.length) return { issue, marker };

  for (const markerLease of expired) {
    const redisLease = await getLeaseForIssue(issue.number, markerLease.language);
    if (redisLease) {
      if (redisLease.lease_id !== markerLease.lease_id ||
          Date.parse(redisLease.expires_at) > Date.now() ||
          !(await reclaimExpiredLease(redisLease))) return null;
    }
  }

  const remainingLeases = marker.leases.filter(
    (lease) => !expired.some((item) => item.lease_id === lease.lease_id),
  );
  const signed = await signVolunteerMarker(issue.number, {
    ...marker,
    leases: remainingLeases,
    submission: marker.submissions?.length ? "submitted" : "active",
  });
  await mutex.assertOwned();
  await patchIssue(issue.number, replaceVolunteerMarker(issue.body || "", signed));

  let current = await fetchIssue(issue.number);
  let currentMarker = parseVolunteerMarker(current.body);
  if (!currentMarker || !(await verifyVolunteerMarker(issue.number, currentMarker))) return null;
  const verifiedMarker = currentMarker;
  if (expired.some((lease) => verifiedMarker.leases.some((item) => item.lease_id === lease.lease_id))) {
    return null;
  }
  const retained = new Set(
    remainingLeases
      .filter((lease) => Date.parse(lease.expires_at) > Date.now())
      .map((lease) => lease.contributor.toLowerCase()),
  );
  for (const submission of verifiedMarker.submissions || []) {
    if (["submitted", "accepted"].includes(submission.status)) {
      retained.add(submission.verified_contributor.toLowerCase());
    }
  }
  const removable = new Set(
    expired.map((lease) => lease.contributor.toLowerCase()).filter((contributor) => !retained.has(contributor)),
  );
  const expectedBeforeUnassignment = new Set(
    [...retained, ...removable],
  );
  const actualBeforeUnassignment = new Set(
    (current.assignees || []).map((assignee) => assignee.login.toLowerCase()),
  );
  if (actualBeforeUnassignment.size !== expectedBeforeUnassignment.size ||
      [...actualBeforeUnassignment].some((login) => !expectedBeforeUnassignment.has(login))) return null;

  for (const contributor of removable) {
    await mutex.assertOwned();
    current = await fetchIssue(issue.number);
    currentMarker = parseVolunteerMarker(current.body);
    if (!currentMarker || !(await verifyVolunteerMarker(issue.number, currentMarker))) return null;
    const verifiedCurrentMarker = currentMarker;
    if (!assigneesMatchWithExtras(current.assignees, verifiedCurrentMarker, removable)) return null;
    const assigned = (current.assignees || []).find(
      (assignee) => assignee.login.toLowerCase() === contributor,
    );
    if (!assigned) continue;
    await unassignIssue(issue.number, assigned.login);
    current = await fetchIssue(issue.number);
    currentMarker = parseVolunteerMarker(current.body);
    if (!currentMarker || !(await verifyVolunteerMarker(issue.number, currentMarker))) return null;
    const pending = new Set(removable);
    pending.delete(contributor);
    if (expired.some((lease) => verifiedCurrentMarker.leases.some((item) => item.lease_id === lease.lease_id)) ||
        !assigneesMatchWithExtras(current.assignees, verifiedCurrentMarker, pending) ||
        (current.assignees || []).some((assignee) => assignee.login.toLowerCase() === contributor)) return null;
  }
  if (!currentMarker) return null;
  const finalMarker = currentMarker;
  if (!volunteerAssigneesMatch(current.assignees, finalMarker)) return null;
  return { issue: current, marker: finalMarker };
}

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
    const euroevalVersion = env("EUROEVAL_VERSION").replace(/\.dev$/, ".dev0");
    const requiredWorkerVersion = env("VOLUNTEER_WORKER_VERSION");
    await assertAssignable(identity.contributor);
    if (body.worker_version !== requiredWorkerVersion) throw new BrokerError(422, "Unsupported worker version.");
    const requestedLanguage = typeof body.language === "string" ? body.language : null;
    const issues = await listOpenIssues();
    for (const listed of issues) {
      const modelId = extractModelId(listed.title, listed.body); if (!modelId) continue;
      const languages = selectedLanguages(listed.body);
      if (!languages.length || requestedLanguage && !languages.includes(requestedLanguage)) continue;
      if (listed.body && VM_MARKER_RE.test(listed.body)) continue;
      if (!volunteerAssigneesMatch(listed.assignees, parseVolunteerMarker(listed.body))) continue;

      // Resolve immutable metadata before taking the short issue mutex. Hub latency
      // must never hold the lock used by another worker claiming another language.
      let model; try { model = await resolveModel(modelId); } catch (error) {
        if (error instanceof BrokerError && error.status === 422) continue; throw error;
      }
      const selectedHardwareGpu = selectedGpu(body.hardware);
      if (!selectedHardwareGpu || !fitsGpu(model, body.hardware, selectedHardwareGpu)) continue;
      const mutex = await acquireRenewableIssueMutex(listed.number); if (!mutex) continue;
      try {
        let snapshot = await fetchIssue(listed.number);
        if (snapshot.state !== "open" || snapshot.body && VM_MARKER_RE.test(snapshot.body)) continue;
        if (extractModelId(snapshot.title, snapshot.body) !== modelId) continue;
        const markerPresent = VOLUNTEER_MARKER_RE.test(snapshot.body || "");
        const marker = parseVolunteerMarker(snapshot.body);
        if (markerPresent && (!marker || !(await verifyVolunteerMarker(snapshot.number, marker)))) continue;
        if (!volunteerAssigneesMatch(snapshot.assignees, marker, Date.now(), true)) continue;
        const recovered = marker
          ? await recoverExpiredOwnership(snapshot, marker, mutex)
          : { issue: snapshot, marker: null };
        if (!recovered) continue;
        snapshot = recovered.issue;
        const selected = selectedLanguages(snapshot.body);
        const current = activeMarker(parseVolunteerMarker(snapshot.body));
        const available = claimableLanguages(selected, current);
        const language = requestedLanguage || available[0];
        if (!language || !available.includes(language)) continue;
        const staleLease = await getLeaseForIssue(snapshot.number, language);
        if (staleLease && Date.parse(staleLease.expires_at) <= Date.now() &&
            !(await reclaimExpiredLease(staleLease))) continue;
        const refreshed = await fetchIssue(snapshot.number);
        const refreshedMarker = parseVolunteerMarker(refreshed.body);
        if (refreshedMarker && !(await verifyVolunteerMarker(refreshed.number, refreshedMarker))) continue;
        if (VOLUNTEER_MARKER_RE.test(refreshed.body || "") && !refreshedMarker) continue;
        if (!volunteerAssigneesMatch(refreshed.assignees, refreshedMarker)) continue;
        snapshot = refreshed;
        const refreshedCurrent = activeMarker(refreshedMarker);
        const refreshedSelected = selectedLanguages(snapshot.body);
        const refreshedAvailable = claimableLanguages(refreshedSelected, refreshedCurrent);
        if (!refreshedAvailable.includes(language)) continue;
        const group = languageGroup(language); if (!group || !GROUPS[group]) continue;
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
            task_groups: [...trusted.task_groups],
            warnings: trusted.warnings || [],
          },
        };
        if (!(await putLease(lease))) continue;
        const nextMarker: VolunteerLeaseMarker = {
          protocol_version: PROTOCOL_VERSION, coordinator: "coordinator", submission: "active",
          leases: [...(current?.leases || []), { lease_id: lease.lease_id, language, worker: lease.worker, contributor: identity.contributor, expires_at: expiresAt }],
          ...(current?.submissions ? { submissions: current.submissions } : {}),
          ...(current?.completed_languages ? { completed_languages: current.completed_languages } : {}),
        };
        let hadAssignment = false;
        try {
          const signedMarker = await signVolunteerMarker(snapshot.number, nextMarker);
          await mutex.assertOwned();
          await patchIssue(snapshot.number, replaceVolunteerMarker(snapshot.body || "", signedMarker));
          await mutex.assertOwned();
          hadAssignment = (snapshot.assignees || []).some((item) =>
            item.login.toLowerCase() === identity.contributor.toLowerCase());
          await assignIssue(snapshot.number, identity.contributor);
          await mutex.assertOwned();
          const after = await fetchIssue(snapshot.number);
          await mutex.assertOwned();
          const afterMarker = parseVolunteerMarker(after.body);
          if (VM_MARKER_RE.test(after.body || "") || !afterMarker ||
              !(await verifyVolunteerMarker(snapshot.number, afterMarker)) ||
              !afterMarker.leases.some((item) => item.lease_id === lease.lease_id) ||
               !volunteerAssigneesMatch(after.assignees, afterMarker) ||
               !(after.assignees || []).some((item) =>
                 item.login.toLowerCase() === identity.contributor.toLowerCase())) {
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
          const cleanup = live ? await fetchIssue(snapshot.number).catch(() => live) : null;
          if (cleanup && !hadAssignment && (cleanup.assignees || []).some((item) =>
            item.login.toLowerCase() === identity.contributor.toLowerCase()) &&
            !parseVolunteerMarker(cleanup.body)?.leases.some((item) => item.lease_id === lease.lease_id)) {
            await mutex.assertOwned();
            const assigned = (cleanup.assignees || []).find((item) =>
              item.login.toLowerCase() === identity.contributor.toLowerCase());
            if (assigned) await unassignIssue(snapshot.number, assigned.login).catch(() => undefined);
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
    return json(status, brokerErrorBody(error, "Unable to claim an evaluation."));
  }
}
