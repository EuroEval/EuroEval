"""Bounded queue discovery and fenced public worker claim transactions."""

from __future__ import annotations

import asyncio
import collections.abc as c
import math
import os
import re
import secrets
import time
import typing as t
from contextlib import suppress
from datetime import UTC, datetime, timedelta

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from upstash_redis.asyncio import Redis

from .auth import authenticate, enforce_rate_limit
from .canary import reserve_canary
from .coverage import narrow_coverage
from .dependencies import (
    PROTOCOL_VERSION,
    get_http,
    get_redis,
    read_json,
    require_env,
    require_protocol,
)
from .github import GitHub, GithubIssue
from .grants import WorkerIdentity
from .hardware import fits_gpu, selected_gpu, valid_hardware
from .leases import Lease
from .markers import (
    VM_MARKER_RE,
    VolunteerMarker,
    claimable_languages,
    extract_model_id,
    has_volunteer_marker,
    parse_volunteer_marker,
    replace_volunteer_marker,
    selected_languages,
    sign_volunteer_marker,
    timestamp,
    verify_volunteer_marker,
    volunteer_assignees_match,
)
from .model import ResolvedModel, model_may_fit, resolve_model
from .ownership import recover_expired_ownership
from .policy import expected_scope, validate_scope_policy_version
from .release import (
    approved_worker_release,
    canonical_worker_version,
    normalize_architecture,
)
from .state import (
    IssueMutex,
    acquire_renewable_issue_mutex,
    delete_lease,
    get_lease_for_issue,
    lease_ttl,
    put_lease,
    reclaim_expired_lease,
)

router = APIRouter()
_GLOBAL_CLAIM_MUTEX = 9_007_199_254_740_991


@router.post("/api/worker/claim")
async def claim(request: Request) -> JSONResponse:
    """Allocate only verified, fitting, incomplete work to the authenticated worker.

    Args:
        request:
            Worker claim and advertised hardware.

    Returns:
        A fenced lease or a complete-discovery no-work result.

    Raises:
        HTTPException:
            If discovery, eligibility, ownership or the transaction cannot complete.
    """
    started = time.monotonic()
    identity = await authenticate(request=request)
    redis = await get_redis(request=request)
    await enforce_rate_limit(
        redis=redis,
        key=f"euroeval:worker:limit:claim:{identity['hash']}",
        limit=120,
        window_seconds=3600,
    )
    body = await read_json(request=request, limit=32 * 1024)
    require_protocol(body=body)
    transaction = _Claim(
        http=get_http(request=request),
        redis=redis,
        identity=identity,
        body=body,
        started=started,
    )
    try:
        async with asyncio.timeout_at(transaction.response_deadline):
            lease = await transaction.run()
        if lease is None:
            return JSONResponse(
                {"protocol_version": PROTOCOL_VERSION, "status": "no_work"}
            )
        view: c.Mapping[str, object] = lease
        fields = (
            "lease_id",
            "issue_number",
            "language",
            "model_id",
            "model_revision",
            "euroeval_version",
            "image_digest",
            "worker_version",
            "expires_at",
            "selected_gpu_index",
            "selected_gpu_uuid",
            "model_type",
            "model_metadata",
            "expected_scope",
            "contamination_canary",
        )
        return JSONResponse(
            {
                "protocol_version": PROTOCOL_VERSION,
                **{name: view[name] for name in fields},
            }
        )
    except TimeoutError as error:
        raise HTTPException(
            status_code=503, detail="Claim transaction timed out; retry claim."
        ) from error
    except HTTPException as error:
        if (
            time.monotonic() >= transaction.discovery_deadline
            and error.status_code != 426
        ):
            raise HTTPException(
                status_code=503,
                detail="Claim discovery deadline exceeded; retry claim.",
            ) from error
        raise


class _Claim:
    """Own one request's discovery budget, eligibility and compensating transaction."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        redis: Redis,
        identity: WorkerIdentity,
        body: dict[str, object],
        started: float,
    ) -> None:
        """Validate the worker before performing queue discovery.

        Args:
            http:
                Outbound connection pool.
            redis:
                Authoritative lease store.
            identity:
                Authenticated worker.
            body:
                Protocol-validated claim object.
            started:
                Monotonic request start.

        Raises:
            HTTPException:
                If hardware or the approved worker release is invalid.
        """
        if not valid_hardware(value=body.get("hardware")):
            raise HTTPException(
                status_code=400,
                detail="hardware must contain well-formed GPUs and free disk.",
            )
        self.hardware = t.cast(dict[str, object], body["hardware"])
        self.gpu = selected_gpu(hardware=self.hardware)
        capabilities = body.get("capabilities")
        self.capabilities = (
            {item for item in capabilities if isinstance(item, str)}
            if isinstance(capabilities, list)
            else set()
        )
        self.requested_language = (
            body.get("language") if isinstance(body.get("language"), str) else None
        )
        self.http, self.redis, self.identity = http, redis, identity
        self.github = GitHub(http=http)
        self.discovery_deadline, self.deadline, self.response_deadline = (
            started + 12,
            started + 18,
            started + 23,
        )
        self.incomplete_discovery = False
        self.mutex_busy = False
        self._approve_release(body=body)

    def _approve_release(self, body: dict[str, object]) -> None:
        """Apply managed attestation without altering the historical worker path.

        Args:
            body:
                Advertised versions and image identity.

        Raises:
            HTTPException:
                If the approved image or worker version is not being used.
        """
        managed = "managed_image_v1" in self.capabilities
        worker_version = body.get("worker_version")
        if not managed and (
            not isinstance(worker_version, str) or not worker_version.strip()
        ):
            raise HTTPException(status_code=400, detail="worker_version is required.")
        require_env(name="VOLUNTEER_MARKER_SECRET")
        architecture = normalize_architecture(
            value=t.cast(str, self.hardware["architecture"])
        )
        if managed:
            if architecture is None:
                raise HTTPException(
                    status_code=426,
                    detail={
                        "error": (
                            "Worker update required; unsupported image architecture."
                        ),
                        "code": "worker_update_required",
                    },
                )
            release = approved_worker_release(architecture=architecture)
            if (
                body.get("image_digest") != release["image_digest"]
                or canonical_worker_version(value=body.get("euroeval_version"))
                != canonical_worker_version(value=release["euroeval_version"])
                or canonical_worker_version(value=worker_version)
                != canonical_worker_version(value=release["worker_version"])
            ):
                raise HTTPException(
                    status_code=426,
                    detail={
                        "error": (
                            "Worker update required; refresh the approved "
                            "worker image and retry."
                        ),
                        "code": "worker_update_required",
                    },
                )
            self.image_digest, self.euroeval_version = (
                release["image_digest"],
                release["euroeval_version"],
            )
        else:
            arm = (
                os.getenv("VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64", "")
                if architecture == "arm64"
                else ""
            )
            self.image_digest = arm or require_env(name="VOLUNTEER_WORKER_IMAGE_DIGEST")
            self.euroeval_version = re.sub(
                r"\.dev$", ".dev0", require_env(name="EUROEVAL_VERSION")
            )
            if worker_version != require_env(name="VOLUNTEER_WORKER_VERSION"):
                raise HTTPException(
                    status_code=422, detail="Unsupported worker version."
                )
        self.worker_version = t.cast(str, worker_version)
        validate_scope_policy_version(euroeval_version=self.euroeval_version)

    async def run(self) -> Lease | None:
        """Discover in priority batches, then serialise final cross-issue allocation.

        Returns:
            Committed lease, or none after complete discovery.

        Raises:
            HTTPException:
                If model discovery or an ownership mutex remains unavailable.
        """
        candidates = await self._candidates()
        possible: dict[str, bool | None] = {}
        resolved: dict[str, ResolvedModel | None] = {}
        for offset in range(0, len(candidates), 6):
            self._check_discovery()
            batch = candidates[offset : offset + 6]
            models = list(
                dict.fromkeys(model for _, model in batch if model not in possible)
            )
            async with asyncio.timeout_at(self.discovery_deadline):
                async with asyncio.TaskGroup() as group:
                    tasks = [
                        group.create_task(self._may_fit(model_id=model))
                        for model in models
                    ]
            for model, task in zip(models, tasks, strict=True):
                possible[model] = task.result()
                self.incomplete_discovery |= task.result() is None
            for issue, model_id in batch:
                self._check_discovery()
                if not possible[model_id]:
                    continue
                if model_id not in resolved:
                    try:
                        async with asyncio.timeout_at(self.discovery_deadline):
                            resolved[model_id] = await resolve_model(
                                http=self.http, model_id=model_id
                            )
                    except HTTPException as error:
                        if error.status_code != 422:
                            raise HTTPException(
                                status_code=503,
                                detail=(
                                    "Model discovery is temporarily unavailable; "
                                    "retry claim."
                                ),
                            ) from error
                        resolved[model_id] = None
                self._check_discovery()
                model = resolved[model_id]
                if not model or not fits_gpu(
                    model=model, hardware=self.hardware, selected=self.gpu
                ):
                    continue
                lease = await self._try_global(issue=issue, model=model)
                if lease is not None:
                    return lease
        if self.mutex_busy:
            raise HTTPException(
                status_code=503,
                detail={"error": "Claim is busy; retry claim.", "code": "claim_busy"},
            )
        if self.incomplete_discovery:
            raise HTTPException(
                status_code=503,
                detail="Model discovery is temporarily unavailable; retry claim.",
            )
        return None

    async def _candidates(self) -> list[tuple[GithubIssue, str]]:
        """Prioritise this contributor's verified unfinished issue groups.

        Returns:
            Eligible issue/model pairs in stable affinity order.
        """
        async with asyncio.timeout_at(self.discovery_deadline):
            issues = await self.github.list_open_issues()
        candidates: list[tuple[GithubIssue, str]] = []
        affinity: dict[int, float] = {}
        now = time.time() * 1000
        for issue in issues:
            model = extract_model_id(title=issue["title"], body=issue["body"])
            languages = selected_languages(body=issue["body"])
            marker = parse_volunteer_marker(body=issue["body"])
            if (
                not model
                or not languages
                or (
                    self.requested_language and self.requested_language not in languages
                )
                or VM_MARKER_RE.search(issue["body"] or "")
                or not volunteer_assignees_match(
                    assignees=issue.get("assignees"),
                    marker=marker,
                    now=now,
                    include_expired_leases=True,
                    allow_expired_unassigned=True,
                )
            ):
                continue
            candidates.append((issue, model))
            if (
                not marker
                or not verify_volunteer_marker(
                    issue_number=issue["number"], marker=marker
                )
                or not claimable_languages(
                    selected=languages, marker=_active(marker=marker)
                )
            ):
                continue
            dates = [
                timestamp(value=entry["submitted_at"])
                for entry in marker.get("submissions", [])
                if entry["status"] in {"submitted", "accepted"}
                and entry["verified_contributor"].lower()
                == self.identity["contributor"].lower()
                and entry["language"] in languages
            ]
            affinity[issue["number"]] = max(
                (date for date in dates if math.isfinite(date)), default=0
            )
        return sorted(
            candidates,
            key=lambda item: affinity.get(item[0]["number"], 0),
            reverse=True,
        )

    def _check_discovery(self) -> None:
        """Refuse to turn incomplete discovery into a false no-work response.

        Raises:
            HTTPException:
                If the discovery budget was exhausted.
        """
        if time.monotonic() >= self.discovery_deadline:
            raise HTTPException(
                status_code=503,
                detail="Model discovery deadline exceeded; retry claim.",
            )

    async def _may_fit(self, model_id: str) -> bool | None:
        """Distinguish permanent exclusions from incomplete model discovery.

        Args:
            model_id:
                Candidate model identity.

        Returns:
            Fit possibility, or none when metadata was unavailable.
        """
        try:
            return await model_may_fit(
                http=self.http,
                model_id=model_id,
                hardware=self.hardware,
                selected=self.gpu,
            )
        except HTTPException as error:
            return False if error.status_code == 422 else None
        except Exception:
            return None

    async def _try_global(
        self, issue: GithubIssue, model: ResolvedModel
    ) -> Lease | None:
        """Hold the cross-issue mutex through the fresh scan and issue transaction.

        Args:
            issue:
                Prioritised candidate.
            model:
                Immutable metadata resolved outside both mutexes.

        Returns:
            Committed lease, or none if this candidate cannot be claimed.

        Raises:
            HTTPException:
                If the transaction no longer has enough response budget.
        """
        global_mutex = await acquire_renewable_issue_mutex(
            redis=self.redis, issue=_GLOBAL_CLAIM_MUTEX
        )
        if global_mutex is None:
            self.mutex_busy = True
            return None
        try:
            await global_mutex.assert_owned()
            async with asyncio.timeout_at(self.discovery_deadline):
                fresh_issues = await self.github.list_open_issues()
            blocked = _blocked_languages(
                issues=fresh_issues, number=issue["number"], model_id=model["id"]
            )
            await global_mutex.assert_owned()
            mutex = await acquire_renewable_issue_mutex(
                redis=self.redis, issue=issue["number"]
            )
            if mutex is None:
                self.mutex_busy = True
                return None
            try:
                async with asyncio.timeout_at(self.deadline):
                    candidate = await self._select(
                        issue=issue, model=model, mutex=mutex, blocked=blocked
                    )
                if candidate is None:
                    return None
                snapshot, marker, lease = candidate
                if time.monotonic() >= self.deadline - 4:
                    raise HTTPException(
                        status_code=503,
                        detail="Claim transaction budget exhausted; retry claim.",
                    )
                return await self._persist(
                    snapshot=snapshot,
                    marker=marker,
                    lease=lease,
                    mutex=mutex,
                    global_mutex=global_mutex,
                )
            finally:
                with suppress(Exception):
                    async with self._io(limit=1.2):
                        await mutex.release()
        finally:
            with suppress(Exception):
                async with self._io(limit=1.2):
                    await global_mutex.release()

    def _io(self, limit: float = 2) -> asyncio.Timeout:
        """Bound an individual transaction operation by the response budget.

        Args:
            limit (optional):
                Operation limit in seconds, defaulting to two.

        Returns:
            Cancellation context for the remaining budget.
        """
        return asyncio.timeout(
            max(0.001, min(limit, self.response_deadline - time.monotonic()))
        )

    async def _persist(
        self,
        snapshot: GithubIssue,
        marker: VolunteerMarker | None,
        lease: Lease,
        mutex: IssueMutex,
        global_mutex: IssueMutex,
    ) -> Lease | None:
        """Fence lease persistence, signed ownership and post-write observations.

        Args:
            snapshot:
                Fresh issue body to extend.
            marker:
                Fresh active signed evidence.
            lease:
                Proposed lease.
            mutex:
                Held issue mutex.
            global_mutex:
                Held cross-issue mutex.

        Returns:
            Committed lease, or none when another live owner won insertion.

        Raises:
            HTTPException:
                If GitHub ownership is lost after persistence.
        """
        async with self._io():
            await global_mutex.assert_owned()
        try:
            async with self._io():
                stored = await put_lease(redis=self.redis, lease=lease)
        except BaseException:
            with suppress(Exception):
                async with self._io(limit=1.2):
                    await delete_lease(redis=self.redis, lease=lease)
            raise
        if not stored:
            return None
        next_marker: VolunteerMarker = {
            **(marker or {}),
            "protocol_version": PROTOCOL_VERSION,
            "coordinator": "coordinator",
            "submission": "active",
            "leases": [
                *(marker["leases"] if marker else []),
                {
                    "lease_id": lease["lease_id"],
                    "language": lease["language"],
                    "worker": lease["worker"],
                    "contributor": lease["contributor"],
                    "expires_at": lease["expires_at"],
                    "ownership_mode": "public",
                },
            ],
        }
        try:
            signed = sign_volunteer_marker(
                issue_number=snapshot["number"], marker=next_marker
            )
            if time.monotonic() >= self.deadline:
                raise HTTPException(
                    status_code=503, detail="Claim deadline exceeded; retry claim."
                )
            async with self._io():
                await global_mutex.assert_owned()
            async with self._io():
                await mutex.assert_owned()
            async with self._io():
                await self.github.patch_issue(
                    number=snapshot["number"],
                    body=replace_volunteer_marker(
                        body=snapshot["body"] or "", marker=signed
                    ),
                )
            async with self._io():
                await mutex.assert_owned()
            async with self._io():
                after = await self.github.get_issue(number=snapshot["number"])
            async with self._io():
                await mutex.assert_owned()
            observed = _verified(issue=after)
            if (
                VM_MARKER_RE.search(after["body"] or "")
                or not observed
                or not any(
                    item["lease_id"] == lease["lease_id"]
                    and item.get("ownership_mode") == "public"
                    for item in observed["leases"]
                )
                or not volunteer_assignees_match(
                    assignees=after.get("assignees"), marker=observed
                )
            ):
                raise HTTPException(
                    status_code=409, detail="GitHub ownership fence lost during claim."
                )
            await self.github.reconcile_public_assignments(
                redis=self.redis,
                issue=after,
                marker=observed,
                assert_owned=mutex.assert_owned,
            )
        except BaseException:
            await self._compensate(snapshot=snapshot, lease=lease, mutex=mutex)
            raise
        return lease

    async def _compensate(
        self, snapshot: GithubIssue, lease: Lease, mutex: IssueMutex
    ) -> None:
        """Remove only observed owned evidence, retaining ambiguous reservations.

        Args:
            snapshot:
                Issue identity, never a rollback body.
            lease:
                This transaction's reservation.
            mutex:
                Ownership fence required before any compensating mutation.
        """
        live = None
        with suppress(Exception):
            async with self._io(limit=1.2):
                live = await self.github.get_issue(number=snapshot["number"])
        if live is None:
            return
        marker = _verified(issue=live)
        contains = bool(
            marker
            and any(item["lease_id"] == lease["lease_id"] for item in marker["leases"])
        )
        if contains and marker:
            with suppress(Exception):
                async with self._io(limit=1.2):
                    await mutex.assert_owned()
                signed = sign_volunteer_marker(
                    issue_number=live["number"],
                    marker={
                        **marker,
                        "leases": [
                            item
                            for item in marker["leases"]
                            if item["lease_id"] != lease["lease_id"]
                        ],
                    },
                )
                async with self._io(limit=1.2):
                    await self.github.patch_issue(
                        number=live["number"],
                        body=replace_volunteer_marker(
                            body=live["body"] or "", marker=signed
                        ),
                    )
            observed = None
            with suppress(Exception):
                async with self._io(limit=1.2):
                    observed = await self.github.get_issue(number=live["number"])
            if observed is None:
                return
            checked = _verified(issue=observed)
            if checked is False or (
                checked
                and any(
                    item["lease_id"] == lease["lease_id"] for item in checked["leases"]
                )
            ):
                return
        elif marker is False:
            return
        with suppress(Exception):
            async with self._io(limit=1.2):
                await delete_lease(redis=self.redis, lease=lease)

    async def _select(
        self,
        issue: GithubIssue,
        model: ResolvedModel,
        mutex: IssueMutex,
        blocked: set[str],
    ) -> tuple[GithubIssue, VolunteerMarker | None, Lease] | None:
        """Revalidate ownership and trusted remaining coverage under the issue mutex.

        Args:
            issue:
                Candidate issue number.
            model:
                Resolved immutable model.
            mutex:
                Held issue mutex.
            blocked:
                Languages owned by another issue for this same model.

        Returns:
            Fresh snapshot, signed ownership and proposed lease, or none.

        Raises:
            HTTPException:
                If remaining canonical coverage needs a newer worker.
        """
        snapshot = await self.github.get_issue(number=issue["number"])
        if (
            snapshot["state"] != "open"
            or VM_MARKER_RE.search(snapshot["body"] or "")
            or extract_model_id(title=snapshot["title"], body=snapshot["body"])
            != model["id"]
        ):
            return None
        marker = _verified(issue=snapshot)
        if marker is False or not volunteer_assignees_match(
            assignees=snapshot.get("assignees"),
            marker=marker,
            include_expired_leases=True,
            allow_expired_unassigned=True,
        ):
            return None
        if marker:
            recovered = await recover_expired_ownership(
                redis=self.redis,
                github=self.github,
                issue=snapshot,
                marker=marker,
                mutex=mutex,
            )
            if recovered is None:
                return None
            snapshot = recovered["issue"]
        current = _active(marker=parse_volunteer_marker(body=snapshot["body"]))
        available = claimable_languages(
            selected=selected_languages(body=snapshot["body"]), marker=current
        )
        language = next(
            (
                item
                for item in available
                if (not self.requested_language or item == self.requested_language)
                and item not in blocked
            ),
            None,
        )
        if language is None:
            return None
        stale = await get_lease_for_issue(
            redis=self.redis, issue=snapshot["number"], language=language
        )
        if (
            stale
            and timestamp(value=stale["expires_at"]) <= time.time() * 1000
            and not await reclaim_expired_lease(redis=self.redis, lease=stale)
        ):
            return None
        snapshot = await self.github.get_issue(number=snapshot["number"])
        if (
            snapshot["state"] != "open"
            or VM_MARKER_RE.search(snapshot["body"] or "")
            or extract_model_id(title=snapshot["title"], body=snapshot["body"])
            != model["id"]
        ):
            return None
        marker = _verified(issue=snapshot)
        if marker is False or not volunteer_assignees_match(
            assignees=snapshot.get("assignees"), marker=marker
        ):
            return None
        current = _active(marker=marker)
        if language not in claimable_languages(
            selected=selected_languages(body=snapshot["body"]), marker=current
        ):
            return None
        try:
            trusted = expected_scope(
                euroeval_version=self.euroeval_version,
                model_type=model["model_type"],
                language=language,
            )
        except HTTPException as error:
            if error.status_code == 422:
                return None
            raise
        coverage = await narrow_coverage(
            http=self.http,
            scope=trusted,
            model_id=model["id"],
            revision=model["revision"],
            requested_version=self.euroeval_version,
        )
        if not coverage["allowed_identity_suffix_sets"]:
            return None
        revision_required = "canonical_coverage_v1" in self.capabilities
        if coverage["covered_identity_suffixes"] and not revision_required:
            raise HTTPException(
                status_code=426,
                detail={
                    "error": (
                        "Worker update required for canonical "
                        "coverage; update the worker image and retry."
                    ),
                    "code": "worker_update_required",
                },
            )
        lease: Lease = {
            "issue_number": snapshot["number"],
            "language": language,
            "worker": self.identity["hash"][:24],
            "contributor": self.identity["contributor"],
            "model_id": model["id"],
            "model_revision": model["revision"],
            "euroeval_version": self.euroeval_version,
            "image_digest": self.image_digest,
            "worker_version": self.worker_version,
            "gpu_memory_utilisation": t.cast(
                float, self.hardware["gpu_memory_utilisation"]
            ),
            "selected_gpu_index": int(t.cast(int, self.hardware["selected_gpu_index"])),
            "selected_gpu_uuid": t.cast(str, self.hardware["selected_gpu_uuid"]),
            "expires_at": (datetime.now(tz=UTC) + timedelta(seconds=lease_ttl()))
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "lease_id": secrets.token_urlsafe(nbytes=18),
            "ownership_mode": "public",
            "record_revision_required": revision_required,
            "model_type": model["model_type"],
            "model_metadata": model["model_metadata"],
            "contamination_canary": t.cast(
                dict[str, object], reserve_canary(model_type=model["model_type"])
            ),
            "expected_scope": {
                "policy_version": trusted["policy_version"],
                "language_group": trusted["language_group"],
                "allowed_identity_suffix_sets": coverage[
                    "allowed_identity_suffix_sets"
                ],
                "original_identity_suffix_sets": coverage[
                    "original_identity_suffix_sets"
                ],
                "covered_identity_suffixes": coverage["covered_identity_suffixes"],
                "task_groups": trusted["task_groups"][:],
                "warnings": trusted["warnings"][:],
            },
        }
        return snapshot, current, lease


def _active(marker: VolunteerMarker | None) -> VolunteerMarker | None:
    """Filter expired slots only after their signed ownership has been checked.

    Args:
        marker:
            Verified marker or no ownership evidence.

    Returns:
        A fresh active projection.
    """
    return (
        {
            **marker,
            "leases": [
                entry
                for entry in marker["leases"]
                if timestamp(value=entry["expires_at"]) > time.time() * 1000
            ],
        }
        if marker
        else None
    )


def _blocked_languages(
    issues: list[GithubIssue], number: int, model_id: str
) -> set[str]:
    """Protect another issue's live, submitted, assigned or ambiguous model work.

    Args:
        issues:
            Fresh queue under the global claim mutex.
        number:
            Candidate issue to exclude.
        model_id:
            Exact model identity.

    Returns:
        Languages that this candidate cannot safely allocate.
    """
    blocked: set[str] = set()
    for issue in issues:
        if (
            issue["number"] == number
            or extract_model_id(title=issue["title"], body=issue["body"]) != model_id
        ):
            continue
        marker = _verified(issue=issue)
        if (
            VM_MARKER_RE.search(issue["body"] or "")
            or marker is False
            or (marker is None and issue.get("assignees"))
        ):
            blocked.update(selected_languages(body=issue["body"]))
            continue
        if marker:
            blocked.update(
                item["language"]
                for item in marker["leases"]
                if timestamp(value=item["expires_at"]) > time.time() * 1000
            )
            blocked.update(
                item["language"]
                for item in marker.get("submissions", [])
                if item["status"] in {"submitted", "accepted"}
            )
    return blocked


def _verified(issue: GithubIssue) -> VolunteerMarker | None | t.Literal[False]:
    """Distinguish absent ownership from malformed or unsigned marker syntax.

    Args:
        issue:
            Observed issue.

    Returns:
        Verified marker, none if absent, or false for untrusted evidence.
    """
    marker = parse_volunteer_marker(body=issue["body"])
    if has_volunteer_marker(body=issue["body"]) and (
        not marker
        or not verify_volunteer_marker(issue_number=issue["number"], marker=marker)
    ):
        return False
    return marker
