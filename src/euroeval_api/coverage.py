"""Read canonical result coverage and narrow trusted identity alternatives."""

from __future__ import annotations

import asyncio
import json
import re
import typing as t
from urllib.parse import quote

import httpx
from fastapi import HTTPException

from .policy import TrustedScope

OBJECT_LIMIT_BYTES = 1_048_576
COVERAGE_PARALLELISM = 6
BUCKET_REPO = "EuroEval/results"
_BUCKET_URL = f"https://huggingface.co/buckets/{BUCKET_REPO}/resolve/"


class Coverage(t.TypedDict):
    """Coverage result for the supplied trusted alternatives."""

    allowed_identity_suffix_sets: list[list[str]]
    original_identity_suffix_sets: list[list[str]]
    covered_identity_suffixes: list[str]


async def narrow_coverage(
    http: httpx.AsyncClient,
    scope: TrustedScope,
    model_id: str,
    revision: str,
    requested_version: str,
) -> Coverage:
    """Narrow alternatives by checking canonical Hugging Face result objects.

    Args:
        http: HTTP client used for canonical bucket reads.
        scope: Validated trusted evaluation scope.
        model_id: Exact model identifier.
        revision: Pinned model revision.
        requested_version: Minimum acceptable EuroEval version.

    Returns:
        Copied original alternatives, narrowed alternatives, and covered suffixes.

    Raises:
        HTTPException: If a canonical check fails or times out.
        asyncio.CancelledError: If the caller cancels this operation.
    """
    suffixes = list(
        dict.fromkeys(
            item for group in scope["allowed_identity_suffix_sets"] for item in group
        )
    )
    cache: dict[str, asyncio.Task[bool]] = {}
    covered: set[str] = set()

    async def check(suffix: str) -> bool:
        key = _result_path(model_id, _identity(suffix))
        if key not in cache:
            cache[key] = asyncio.create_task(
                _lookup(http, key, model_id, revision, requested_version)
            )
        return await cache[key]

    try:
        async with asyncio.timeout(4):
            for offset in range(0, len(suffixes), COVERAGE_PARALLELISM):
                batch = suffixes[offset : offset + COVERAGE_PARALLELISM]
                results = await asyncio.gather(
                    *(check(suffix) for suffix in batch), return_exceptions=True
                )
                failure = next(
                    (result for result in results if isinstance(result, BaseException)),
                    None,
                )
                if failure is not None:
                    raise failure
                covered.update(
                    suffix
                    for suffix, exists in zip(batch, results, strict=True)
                    if exists
                )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=503, detail="Canonical coverage unavailable"
        ) from error
    original = [list(group) for group in scope["allowed_identity_suffix_sets"]]
    narrowed = (
        []
        if any(all(item in covered for item in group) for group in original)
        else [
            remaining
            for group in original
            if (remaining := [item for item in group if item not in covered])
        ]
    )
    return {
        "allowed_identity_suffix_sets": narrowed,
        "original_identity_suffix_sets": original,
        "covered_identity_suffixes": [item for item in suffixes if item in covered],
    }


def _identity(suffix: str) -> tuple[str, bool | None, bool | None]:
    """Parse dataset, validation split, and few-shot identity.

    Returns:
        Parsed identity components.

    Raises:
        _CoverageError: If the suffix is malformed.
    """
    parts = suffix.split("__")
    if len(parts) != 3:
        raise _CoverageError("Malformed identity")
    return parts[0], _bool_or_null(parts[1]), _bool_or_null(parts[2])


class _CoverageError(Exception):
    """A canonical coverage check cannot produce a trustworthy answer."""


def _bool_or_null(value: object) -> bool | None:
    """Convert the accepted persisted boolean representations.

    Returns:
        The boolean value or ``None``.

    Raises:
        _CoverageError: If the value is not an accepted representation.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false", "null"}:
        return {"true": True, "false": False, "null": None}[value.lower()]
    raise _CoverageError("Malformed identity")


async def _lookup(
    http: httpx.AsyncClient,
    key: str,
    model_id: str,
    revision: str,
    requested_version: str,
) -> bool:
    """Check one canonical result object and its persisted identity.

    Returns:
        Whether the canonical result covers this identity.

    Raises:
        _CoverageError: If the object is unavailable or malformed.
        ValueError: If the persisted version is invalid.
        asyncio.CancelledError: If this check is cancelled.
    """
    url = _BUCKET_URL + _encode_path(key)
    try:
        async with http.stream(
            "GET",
            url,
            headers={"Range": f"bytes=0-{OBJECT_LIMIT_BYTES - 1}"},
            follow_redirects=True,
        ) as response:
            if response.status_code == 404:
                return False
            if response.status_code < 200 or response.status_code >= 300:
                raise _CoverageError("Canonical coverage unavailable")
            length = response.headers.get("content-length")
            if length is not None and int(length) > OBJECT_LIMIT_BYTES:
                raise _CoverageError("Canonical result exceeds coverage limit")
            raw = await _read_bounded(response)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        raise _CoverageError("Canonical coverage lookup failed") from error
    try:
        record = json.loads(
            raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError())
        )
        if (
            not isinstance(record, dict)
            or record.get("schema_version") not in {"0.2.1", "0.3.0"}
            or not isinstance(record.get("evaluation_results"), list)
        ):
            raise ValueError
        model, library = record.get("model_info"), record.get("eval_library")
        details = (
            library.get("additional_details") if isinstance(library, dict) else None
        )
        if (
            not isinstance(model, dict)
            or not isinstance(library, dict)
            or library.get("name") != "euroeval"
            or not isinstance(library.get("version"), str)
            or not library["version"]
            or not isinstance(details, dict)
            or not isinstance(details.get("dataset"), str)
            or "validation_split" not in details
            or "few_shot" not in details
        ):
            raise ValueError
        split, shot = (
            _bool_or_null(details["validation_split"]),
            _bool_or_null(details["few_shot"]),
        )
    except (ValueError, TypeError, _CoverageError) as error:
        raise _CoverageError("Canonical result is malformed") from error
    aliases = [
        model.get("id"),
        model.get("name"),
        *(model.get("aliases") if isinstance(model.get("aliases"), list) else []),
    ]
    dataset, path_split, path_shot = _identity(key.rsplit("/", 1)[1][:-5])
    if model_id not in aliases or model.get("revision") != revision:
        return False
    if details["dataset"].replace("/", "_") != dataset or (split, shot) != (
        path_split,
        path_shot,
    ):
        return False
    compared = _compare_version(library["version"], requested_version)
    if compared is None:
        raise _CoverageError("Canonical result version is malformed")
    return compared >= 0


def _compare_version(left: str, right: str) -> int | None:
    """Compare versions using the canonical restricted EuroEval grammar.

    Returns:
        Ordering value, or ``None`` for an invalid version.
    """
    pattern = re.compile(r"^(\d+(?:\.\d+)*)(?:(a|b|rc|\.dev)(\d*))?$", re.IGNORECASE)

    def parse(value: str) -> list[int] | None:
        value = value[:-4] + ".dev0" if value.endswith(".dev") else value
        match = pattern.fullmatch(value)
        if match is None:
            return None
        phase = (match.group(2) or "").lower()
        stage = {".dev": -4, "a": -3, "b": -2, "rc": -1, "": 0}[phase]
        return [
            *(int(part) for part in match.group(1).split(".")),
            stage,
            int(match.group(3) or 0),
        ]

    first, second = parse(left), parse(right)
    if first is None or second is None:
        return None
    width = max(len(first), len(second))
    first.extend([0] * (width - len(first)))
    second.extend([0] * (width - len(second)))
    return (first > second) - (first < second)


def _encode_path(path: str) -> str:
    """URI-encode each path segment while retaining separators.

    Returns:
        URI-encoded path.
    """
    return "/".join(quote(part, safe="") for part in path.split("/"))


async def _read_bounded(response: httpx.Response) -> bytes:
    """Read response content without exceeding the decoded-byte limit.

    Returns:
        Response body bytes.

    Raises:
        _CoverageError: If the body exceeds the limit.
    """
    data = bytearray()
    async for chunk in response.aiter_bytes():
        data.extend(chunk)
        if len(data) > OBJECT_LIMIT_BYTES:
            raise _CoverageError("Canonical result exceeds coverage limit")
    return bytes(data)


def _result_path(model_id: str, identity: tuple[str, bool | None, bool | None]) -> str:
    """Build the canonical bucket-relative result path.

    Returns:
        Bucket-relative result path.
    """
    dataset, split, shot = identity

    def label(value: bool | None) -> str:
        return "null" if value is None else str(value).lower()

    model = model_id.replace("/", "_")
    return f"results/{model}/{dataset}__{label(split)}__{label(shot)}.json"
