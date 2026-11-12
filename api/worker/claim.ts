import {
  BrokerError, ConfigurationError, GROUPS, PROTOCOL_VERSION, VM_MARKER_RE, VOLUNTEER_MARKER_RE,
  acquireRenewableIssueMutex, authenticate, brokerErrorBody,
  claimableLanguages, env, expectedScope, validateScopePolicyVersion,
  extractModelId, enforceRateLimit, fitsGpu, fetchIssue, json, languageGroup, leaseTtl, selectedGpu,
  listOpenIssues, method,
  patchIssue, putLease, randomToken, readJson, deleteLease, unassignIssue,
  getLeaseForIssue, reclaimExpiredLease,
  modelMayFit, parseVolunteerMarker, resolveModel, selectedLanguages, volunteerAssigneesMatch,
  requireProtocol, signVolunteerMarker, verifyVolunteerMarker, markerSecret, replaceVolunteerMarker,
  reserveCanary,
} from "./_lib.js";
import type { Lease, VolunteerLeaseMarker } from "./_lib.js";

// Reserved only as a Redis mutex key; never used for a GitHub API request.
const GLOBAL_CLAIM_MUTEX = Number.MAX_SAFE_INTEGER;
const CLAIM_IO_TIMEOUT_MS = 2_000;
const CLAIM_CLEANUP_TIMEOUT_MS = 1_200;

async function boundedClaimIo<T>(
  operation: (signal: AbortSignal) => Promise<T>, timeoutMs: number, responseDeadline: number,
): Promise<T> {
  const controller = new AbortController();
  const remaining = Math.max(1, Math.min(timeoutMs, responseDeadline - Date.now()));
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      controller.abort();
      reject(new BrokerError(503, "Claim transaction timed out; retry claim."));
    }, remaining);
  });
  try { return await Promise.race([operation(controller.signal), timeout]); }
  finally { if (timer) clearTimeout(timer); }
}

export const config = { runtime: "edge" };

async function recoverExpiredOwnership(
  issue: Awaited<ReturnType<typeof fetchIssue>>,
  marker: VolunteerLeaseMarker,
  mutex: NonNullable<Awaited<ReturnType<typeof acquireRenewableIssueMutex>>>,
): Promise<{ issue: Awaited<ReturnType<typeof fetchIssue>>; marker: VolunteerLeaseMarker } | null> {
  let current = issue;
  let currentMarker: VolunteerLeaseMarker | null = marker;
  while (true) {
    const expired = currentMarker.leases.filter((lease) => Date.parse(lease.expires_at) <= Date.now());
    if (!expired.length) return { issue: current, marker: currentMarker };
    const expiredLease = expired[0];
    const contributor = expiredLease.contributor.toLowerCase();
    const redisLease = await getLeaseForIssue(current.number, expiredLease.language);
    if (redisLease && (redisLease.lease_id !== expiredLease.lease_id ||
        redisLease.contributor.toLowerCase() !== contributor ||
        Date.parse(redisLease.expires_at) > Date.now())) return null;
    if (!volunteerAssigneesMatch(current.assignees, currentMarker, Date.now(), true, true)) return null;
    const retained = currentMarker.leases.some(
      (lease) => lease.lease_id !== expiredLease.lease_id && lease.ownership_mode !== "public" &&
        lease.contributor.toLowerCase() === contributor,
    ) || (currentMarker.submissions || []).some(
      (submission) => ["submitted", "accepted"].includes(submission.status) &&
        submission.ownership_mode !== "public" &&
        submission.verified_contributor.toLowerCase() === contributor,
    );

    const assigned = (current.assignees || []).find(
      (assignee) => assignee.login.toLowerCase() === contributor,
    );
    if (expiredLease.ownership_mode !== "public" && assigned && !retained) {
      await mutex.assertOwned();
      try {
        await unassignIssue(current.number, assigned.login);
      } catch (error) {
        const observed = await fetchIssue(current.number).catch(() => null);
        if (!observed || (observed.assignees || []).some(
          (assignee) => assignee.login.toLowerCase() === contributor,
        )) throw error;
        current = observed;
      }
      current = await fetchIssue(current.number);
      currentMarker = parseVolunteerMarker(current.body);
      if (!currentMarker || !(await verifyVolunteerMarker(current.number, currentMarker))) return null;
      if (currentMarker.leases.some((lease) => lease.lease_id === expiredLease.lease_id) &&
          (current.assignees || []).some((assignee) => assignee.login.toLowerCase() === contributor)) return null;
      continue;
    }

    await mutex.assertOwned();
    const signed = await signVolunteerMarker(current.number, {
      ...currentMarker,
      submission: currentMarker.submissions?.length ? "submitted" : "active",
      leases: currentMarker.leases.filter((lease) => lease.lease_id !== expiredLease.lease_id),
    });
    try {
      await patchIssue(current.number, replaceVolunteerMarker(current.body || "", signed));
    } catch (error) {
      const observed = await fetchIssue(current.number).catch(() => null);
      const observedMarker = observed && parseVolunteerMarker(observed.body);
      if (!observed || !observedMarker || observedMarker.leases.some(
        (lease) => lease.lease_id === expiredLease.lease_id,
      )) throw error;
      current = observed;
      currentMarker = observedMarker;
    }
    current = await fetchIssue(current.number);
    currentMarker = parseVolunteerMarker(current.body);
    if (!currentMarker || !(await verifyVolunteerMarker(current.number, currentMarker))) return null;
    if (currentMarker.leases.some((lease) => lease.lease_id === expiredLease.lease_id)) return null;

    const redisAfterRecovery = await getLeaseForIssue(current.number, expiredLease.language);
    if (redisAfterRecovery && (redisAfterRecovery.lease_id !== expiredLease.lease_id ||
        Date.parse(redisAfterRecovery.expires_at) > Date.now())) return null;
    if (redisAfterRecovery) {
      await mutex.assertOwned();
      if (!(await reclaimExpiredLease(redisAfterRecovery))) {
        const remaining = await getLeaseForIssue(current.number, expiredLease.language);
        if (remaining) return null;
      }
    }
  }
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
  const requestStarted = Date.now();
  const deadline = requestStarted + 18_000;
  const responseDeadline = requestStarted + 23_000;
  const signal = AbortSignal.timeout(12_000);
  try {
    const identity = await authenticate(req); await enforceRateLimit(`euroeval:worker:limit:claim:${identity.hash}`, 120, 3600);
    const body = await readJson(req, 32 * 1024); requireProtocol(body);
    if (!validHardware(body.hardware)) throw new BrokerError(400, "hardware must contain well-formed GPUs and free disk.");
    if (typeof body.worker_version !== "string" || !body.worker_version.trim()) throw new BrokerError(400, "worker_version is required.");
    markerSecret();
    const architecture = body.hardware.architecture;
    let arm64ImageDigest: string | undefined;
    if (architecture === "arm64" || architecture === "aarch64") {
      try { arm64ImageDigest = env("VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64"); }
      catch (error) { if (!(error instanceof ConfigurationError)) throw error; }
    }
    const imageDigest = arm64ImageDigest || env("VOLUNTEER_WORKER_IMAGE_DIGEST");
    const euroevalVersion = env("EUROEVAL_VERSION").replace(/\.dev$/, ".dev0");
    validateScopePolicyVersion(euroevalVersion);
    const requiredWorkerVersion = env("VOLUNTEER_WORKER_VERSION");
    if (body.worker_version !== requiredWorkerVersion) throw new BrokerError(422, "Unsupported worker version.");
    const requestedLanguage = typeof body.language === "string" ? body.language : null;
    const issues = await listOpenIssues(signal);
    const selectedHardwareGpu = selectedGpu(body.hardware);
    const candidates = issues.flatMap((listed) => {
      const modelId = extractModelId(listed.title, listed.body);
      const languages = selectedLanguages(listed.body);
      return modelId && languages.length && !(requestedLanguage && !languages.includes(requestedLanguage)) &&
        !(listed.body && VM_MARKER_RE.test(listed.body)) && volunteerAssigneesMatch(
          listed.assignees, parseVolunteerMarker(listed.body), Date.now(), true, true,
        ) ? [{ listed, modelId }] : [];
    });
    // Resolve candidates in issue-priority batches. Do not inspect later issues
    // until every earlier candidate has either been ruled out or attempted.
    const resolved = new Map<string, Awaited<ReturnType<typeof resolveModel>> | null>();
    const mayFit = new Map<string, boolean | null>();
    let incompleteDiscovery = false;
    let claimMutexBusy = false;
    const checkDeadline = (): void => {
      if (Date.now() >= deadline) throw new BrokerError(503, "Claim deadline exceeded; retry claim.");
    };
    const checkDiscoveryDeadline = (): void => {
      if (signal.aborted || Date.now() >= deadline - 6_000) {
        throw new BrokerError(503, "Model discovery deadline exceeded; retry claim.");
      }
    };
    const batchSize = 6;
    for (let start = 0; start < candidates.length; start += batchSize) {
      checkDiscoveryDeadline();
      const batch = candidates.slice(start, start + batchSize);
      const batchModels = [...new Set(batch.map(({ modelId }) => modelId))]
        .filter((modelId) => !mayFit.has(modelId));
      const outcomes = await Promise.all(batchModels.map(async (modelId) => {
        try { return [modelId, await modelMayFit(modelId, body.hardware as Record<string, unknown>, selectedHardwareGpu, signal)] as const; }
        catch (error) {
          if (error instanceof BrokerError && error.status === 422) return [modelId, false] as const;
          return [modelId, null] as const;
        }
      }));
      outcomes.forEach(([modelId, fits]) => {
        if (fits === null) incompleteDiscovery = true;
        mayFit.set(modelId, fits);
      });
      checkDiscoveryDeadline();
      for (const { listed, modelId } of batch) {
      checkDiscoveryDeadline();
      if (!mayFit.get(modelId)) continue;
      if (!resolved.has(modelId)) {
        try { resolved.set(modelId, await resolveModel(modelId, signal)); }
        catch (error) {
          if (error instanceof BrokerError && error.status === 422) resolved.set(modelId, null);
          else throw new BrokerError(503, "Model discovery is temporarily unavailable; retry claim.");
        }
      }
      checkDiscoveryDeadline();
      const model = resolved.get(modelId);
      if (!model || !fitsGpu(model, body.hardware, selectedHardwareGpu)) continue;
      // Serialize final fresh metadata, cross-issue scan, and claim transaction.
      const globalMutex = await acquireRenewableIssueMutex(GLOBAL_CLAIM_MUTEX);
      if (!globalMutex) { claimMutexBusy = true; continue; }
      try {
        await globalMutex.assertOwned();
        const freshIssues = await listOpenIssues(signal);
        const blockedLanguages = new Set<string>();
        for (const candidate of freshIssues) {
          if (candidate.number === listed.number ||
              extractModelId(candidate.title, candidate.body) !== modelId) continue;
          const selected = selectedLanguages(candidate.body);
          const markerPresent = VOLUNTEER_MARKER_RE.test(candidate.body || "");
          const marker = parseVolunteerMarker(candidate.body);
          if (candidate.body && VM_MARKER_RE.test(candidate.body) ||
              markerPresent && (!marker || !(await verifyVolunteerMarker(candidate.number, marker))) ||
              !markerPresent && (candidate.assignees || []).length > 0) {
            selected.forEach((language) => blockedLanguages.add(language));
            continue;
          }
          if (!marker || !(await verifyVolunteerMarker(candidate.number, marker))) continue;
          marker.leases.filter((lease) => Date.parse(lease.expires_at) > Date.now())
            .forEach((lease) => blockedLanguages.add(lease.language));
          (marker.submissions || []).filter((submission) =>
            submission.status === "submitted" || submission.status === "accepted")
            .forEach((submission) => blockedLanguages.add(submission.language));
        }
        await globalMutex.assertOwned();
        const mutex = await acquireRenewableIssueMutex(listed.number);
        if (!mutex) { claimMutexBusy = true; continue; }
        try {
        let snapshot = await fetchIssue(listed.number);
        if (snapshot.state !== "open" || snapshot.body && VM_MARKER_RE.test(snapshot.body)) continue;
        if (extractModelId(snapshot.title, snapshot.body) !== modelId) continue;
        const markerPresent = VOLUNTEER_MARKER_RE.test(snapshot.body || "");
        const marker = parseVolunteerMarker(snapshot.body);
        if (markerPresent && (!marker || !(await verifyVolunteerMarker(snapshot.number, marker)))) continue;
        if (!volunteerAssigneesMatch(snapshot.assignees, marker, Date.now(), true, true)) continue;
        checkDeadline();
        const recovered = marker
          ? await recoverExpiredOwnership(snapshot, marker, mutex)
          : { issue: snapshot, marker: null };
        if (!recovered) continue;
        snapshot = recovered.issue;
        const selected = selectedLanguages(snapshot.body);
        const current = activeMarker(parseVolunteerMarker(snapshot.body));
        const available = claimableLanguages(selected, current);
        const language = available.find((item) =>
          (!requestedLanguage || item === requestedLanguage) && !blockedLanguages.has(item));
        if (!language) continue;
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
        // Use the immutable revision/config resolved before taking either mutex;
        // identity and ownership are rechecked against fresh GitHub state above.
        const group = languageGroup(language); if (!group || !GROUPS[group]) continue;
        let trusted;
        try { trusted = expectedScope(euroevalVersion, model.model_type, language); }
        catch (error) { if (error instanceof BrokerError && error.status === 422) continue; throw error; }
        checkDeadline();
        const expiresAt = new Date(Date.now() + leaseTtl() * 1000).toISOString();
        const leaseId = randomToken(18);
        const contaminationCanary = await reserveCanary(
          model.id, model.revision, model.model_type, leaseId,
        );
        const lease: Lease = {
          issue_number: snapshot.number, language, worker: identity.hash.slice(0, 24), contributor: identity.contributor,
          model_id: model.id, model_revision: model.revision, euroeval_version: euroevalVersion,
          image_digest: imageDigest,
          worker_version: body.worker_version,
          gpu_memory_utilisation: body.hardware.gpu_memory_utilisation as number,
          selected_gpu_index: body.hardware.selected_gpu_index as number,
          selected_gpu_uuid: body.hardware.selected_gpu_uuid as string,
          expires_at: expiresAt,
          lease_id: leaseId,
          ownership_mode: "public",
          model_type: model.model_type,
          model_metadata: model.model_metadata,
          contamination_canary: contaminationCanary,
          expected_scope: {
            policy_version: trusted.policy_version,
            language_group: trusted.language_group,
            allowed_identity_suffix_sets: trusted.allowed_identity_suffix_sets.map((set) => [...set]),
            task_groups: [...trusted.task_groups],
            warnings: trusted.warnings || [],
          },
        };
        if (Date.now() >= deadline - 4_000) {
          throw new BrokerError(503, "Claim transaction budget exhausted; retry claim.");
        }
        await globalMutex.assertOwned();
        let leaseStored: boolean;
        try { leaseStored = await boundedClaimIo((signal) => putLease(lease, signal), CLAIM_IO_TIMEOUT_MS, responseDeadline); }
        catch (error) {
          // EVAL may have committed despite a lost/aborted response. Delete is
          // lease-id fenced, so reconciliation cannot remove another worker's lease.
          await boundedClaimIo((signal) => deleteLease(lease, signal), CLAIM_CLEANUP_TIMEOUT_MS, responseDeadline)
            .catch(() => undefined);
          throw error;
        }
        if (!leaseStored) continue;
        const nextMarker: VolunteerLeaseMarker = {
          protocol_version: PROTOCOL_VERSION, coordinator: "coordinator", submission: "active",
          leases: [...(current?.leases || []), { lease_id: lease.lease_id, language, worker: lease.worker, contributor: identity.contributor, expires_at: expiresAt, ownership_mode: "public" }],
          ...(current?.submissions ? { submissions: current.submissions } : {}),
          ...(current?.completed_languages ? { completed_languages: current.completed_languages } : {}),
        };
        try {
          const signedMarker = await signVolunteerMarker(snapshot.number, nextMarker);
          checkDeadline();
          await boundedClaimIo((signal) => globalMutex.assertOwned(signal), CLAIM_IO_TIMEOUT_MS, responseDeadline);
          await boundedClaimIo((signal) => mutex.assertOwned(signal), CLAIM_IO_TIMEOUT_MS, responseDeadline);
          await boundedClaimIo((signal) => patchIssue(snapshot.number,
            replaceVolunteerMarker(snapshot.body || "", signedMarker), signal), CLAIM_IO_TIMEOUT_MS, responseDeadline);
          await boundedClaimIo((signal) => mutex.assertOwned(signal), CLAIM_IO_TIMEOUT_MS, responseDeadline);
          const after = await boundedClaimIo((signal) => fetchIssue(snapshot.number, signal), CLAIM_IO_TIMEOUT_MS, responseDeadline);
          await boundedClaimIo((signal) => mutex.assertOwned(signal), CLAIM_IO_TIMEOUT_MS, responseDeadline);
          const afterMarker = parseVolunteerMarker(after.body);
          if (VM_MARKER_RE.test(after.body || "") || !afterMarker ||
              !(await verifyVolunteerMarker(snapshot.number, afterMarker)) ||
              !afterMarker.leases.some((item) => item.lease_id === lease.lease_id &&
                item.ownership_mode === "public") ||
              !volunteerAssigneesMatch(after.assignees, afterMarker)) {
            throw new BrokerError(409, "GitHub ownership fence lost during claim.");
          }
        } catch (error) {
          // Never restore a stale body: another lease may have been added while
          // GitHub was unavailable. Remove only our reservation when still present.
          const live = await boundedClaimIo((signal) => fetchIssue(snapshot.number, signal), CLAIM_CLEANUP_TIMEOUT_MS, responseDeadline).catch(() => null);
          const liveMarker = live ? parseVolunteerMarker(live.body) : null;
          if (live && liveMarker?.leases.some((item) => item.lease_id === lease.lease_id)) {
            await boundedClaimIo((signal) => mutex.assertOwned(signal), CLAIM_CLEANUP_TIMEOUT_MS, responseDeadline);
            const cleanupMarker = await signVolunteerMarker(live.number, {
              ...liveMarker, leases: liveMarker.leases.filter((item) => item.lease_id !== lease.lease_id),
            });
            await boundedClaimIo((signal) => patchIssue(snapshot.number,
              replaceVolunteerMarker(live.body || "", cleanupMarker), signal), CLAIM_CLEANUP_TIMEOUT_MS, responseDeadline)
              .catch(() => undefined);
          }
          const cleanup = live ? await boundedClaimIo((signal) => fetchIssue(snapshot.number, signal),
            CLAIM_CLEANUP_TIMEOUT_MS, responseDeadline).catch(() => null) : null;
          const cleanupMarker = cleanup ? parseVolunteerMarker(cleanup.body) : null;
          const markerAbsent = !!live && !liveMarker?.leases.some((item) => item.lease_id === lease.lease_id);
          const markerRemoved = !!cleanup && !cleanupMarker?.leases.some((item) => item.lease_id === lease.lease_id);
          // If GitHub state cannot be observed after an ambiguous PATCH, retain
          // the lease rather than risk making a committed marker unbacked.
          if (markerAbsent || markerRemoved) await boundedClaimIo((signal) => deleteLease(lease, signal),
            CLAIM_CLEANUP_TIMEOUT_MS, responseDeadline).catch(() => undefined);
          throw error;
        }
        return json(200, {
          protocol_version: PROTOCOL_VERSION, lease_id: lease.lease_id, issue_number: lease.issue_number,
          language, model_id: model.id, model_revision: model.revision, euroeval_version: euroevalVersion,
          image_digest: imageDigest, worker_version: lease.worker_version, expires_at: expiresAt,
          selected_gpu_index: lease.selected_gpu_index, selected_gpu_uuid: lease.selected_gpu_uuid,
          model_type: lease.model_type, model_metadata: lease.model_metadata,
          expected_scope: lease.expected_scope,
          contamination_canary: lease.contamination_canary,
        });
        } finally {
          await boundedClaimIo((signal) => mutex.release(signal), CLAIM_CLEANUP_TIMEOUT_MS, responseDeadline).catch(() => undefined);
        }
      } finally {
        await boundedClaimIo((signal) => globalMutex.release(signal), CLAIM_CLEANUP_TIMEOUT_MS, responseDeadline).catch(() => undefined);
      }
      }
    }
    if (claimMutexBusy) throw new BrokerError(503, "Claim is busy; retry claim.", "claim_busy");
    if (incompleteDiscovery) throw new BrokerError(503, "Model discovery is temporarily unavailable; retry claim.");
    return json(200, { protocol_version: PROTOCOL_VERSION, status: "no_work" });
  } catch (error) {
    const failure = signal.aborted
      ? new BrokerError(503, "Claim discovery deadline exceeded; retry claim.") : error;
    const status = failure instanceof BrokerError ? failure.status : failure instanceof ConfigurationError ? 503 : 502;
    return json(status, brokerErrorBody(failure, "Unable to claim an evaluation."));
  }
}
