"""Trusted volunteer scope policy loading and validation."""

from __future__ import annotations

import json
import os
import re
import typing as t
from pathlib import Path

from fastapi import HTTPException
from pydantic_core import from_json

from .markers import GROUPS

ModelType = t.Literal["encoder", "generative"]


class TrustedScope(t.TypedDict):
    """Validated trusted scope returned to volunteer workflows."""

    euroeval_version: str
    model_type: ModelType
    language: str
    language_group: str
    policy_version: str
    allowed_identity_suffix_sets: list[list[str]]
    task_groups: list[str]
    warnings: list[str]


LANGUAGE_PARENTS: dict[str, str] = {
    "nb": "no",
    "nn": "no",
    "pt-br": "pt",
    "pt-pt": "pt",
}
_POLICY_PATH = Path(__file__).parents[2] / "api/worker/scope-policy.json"


def expected_scope(
    euroeval_version: str, model_type: ModelType, language: str
) -> TrustedScope:
    """Return an isolated trusted expected scope for a model and language.

    Args:
        euroeval_version:
            EuroEval version used for the evaluation.
        model_type:
            Requested model capability.
        language:
            Exact requested language code.

    Returns:
        Validated scope with fresh mutable lists.

    Raises:
        HTTPException:
            Configuration is malformed or the request is unsupported.
    """
    group = language_group(language)
    if group is None or model_type not in ("encoder", "generative"):
        raise HTTPException(
            status_code=422, detail="The leased language or model type is unsupported."
        )
    version = _canonical_version(euroeval_version)
    policy = _load_policy()
    entries = t.cast(list[object], policy["policies"])
    match = next(
        (
            item
            for item in entries
            if isinstance(item, dict)
            and item.get("euroeval_version") == version
            and item.get("model_type") == model_type
            and item.get("language") == language
        ),
        None,
    )
    if match is None:
        raise HTTPException(
            status_code=422,
            detail="This model type and language has no trusted expected scope.",
        )
    if "identity_suffixes" in match:
        raise HTTPException(
            status_code=503,
            detail="Trusted expected scope mixes legacy and alternative identities.",
        )
    alternatives = match.get("allowed_identity_suffix_sets")
    if not isinstance(alternatives, list) or not alternatives:
        raise HTTPException(
            status_code=422,
            detail="This model type and language has no trusted expected scope.",
        )
    clean: list[list[str]] = []
    for alternative in alternatives:
        if not isinstance(alternative, list) or not alternative:
            raise HTTPException(
                status_code=503,
                detail="Trusted expected scope contains an empty identity alternative.",
            )
        suffixes = [_canonical_suffix(suffix) for suffix in alternative]
        if len(set(suffixes)) != len(suffixes):
            raise HTTPException(
                status_code=503,
                detail="Trusted expected scope contains duplicate identities.",
            )
        clean.append(suffixes)
    keys = [tuple(sorted(item)) for item in clean]
    if len(set(keys)) != len(keys):
        raise HTTPException(
            status_code=503,
            detail="Trusted expected scope contains duplicate identity alternatives.",
        )
    tasks = match.get("task_groups")
    if (
        not isinstance(tasks, list)
        or not tasks
        or any(not isinstance(task, str) or not task for task in tasks)
        or len(set(tasks)) != len(tasks)
    ):
        raise HTTPException(
            status_code=503,
            detail="Trusted expected scope contains invalid task groups.",
        )
    language_group_value = match.get("language_group")
    if not isinstance(language_group_value, str) or not language_group_value:
        raise HTTPException(
            status_code=503, detail="Trusted expected scope has no language_group."
        )
    warnings = match.get("warnings", [])
    if not isinstance(warnings, list) or any(
        not isinstance(item, str) for item in warnings
    ):
        raise HTTPException(
            status_code=503, detail="Trusted expected scope contains invalid warnings."
        )
    return {
        "euroeval_version": version,
        "model_type": model_type,
        "language": language,
        "language_group": language_group_value,
        "policy_version": t.cast(str, policy["policy_version"]),
        "allowed_identity_suffix_sets": clean,
        "task_groups": list(tasks),
        "warnings": list(warnings),
    }


def _canonical_suffix(value: object) -> str:
    """Validate the canonical JSON encoding of one identity suffix.

    Args:
        value:
            Configured suffix.

    Returns:
        Validated canonical suffix.

    Raises:
        HTTPException:
            If trusted configuration contains a malformed suffix.
    """
    if not isinstance(value, str):
        raise HTTPException(
            status_code=503,
            detail="Trusted expected scope contains a non-canonical identity.",
        )
    try:
        parsed = from_json(value, allow_inf_nan=False)
    except ValueError as error:
        raise HTTPException(
            status_code=503,
            detail="Trusted expected scope contains invalid identity JSON.",
        ) from error
    if (
        not isinstance(parsed, list)
        or len(parsed) != 3
        or not isinstance(parsed[0], str)
        or (parsed[1] is not None and not isinstance(parsed[1], bool))
        or (parsed[2] is not None and not isinstance(parsed[2], bool))
        or json.dumps(parsed, ensure_ascii=False, separators=(",", ":")) != value
    ):
        raise HTTPException(
            status_code=503,
            detail="Trusted expected scope contains a non-canonical identity.",
        )
    return value


def _canonical_version(version: str) -> str:
    """Normalise development versions to the policy's canonical spelling.

    Args:
        version:
            Requested EuroEval version.

    Returns:
        Version with an explicit development sequence.
    """
    return re.sub(r"\.dev$", ".dev0", version)


def _load_policy() -> dict[str, object]:
    """Load and minimally validate the trusted policy configuration.

    Returns:
        Configured policy object.

    Raises:
        HTTPException:
            If the configuration cannot be read or has an invalid schema.
    """
    raw = os.environ.get("VOLUNTEER_SCOPE_POLICY_JSON")
    try:
        if not raw:
            raw = _POLICY_PATH.read_text(encoding="utf-8")
        policy = from_json(raw, allow_inf_nan=False)
    except (ValueError, OSError, TypeError) as error:
        raise HTTPException(
            status_code=503, detail="VOLUNTEER_SCOPE_POLICY_JSON is invalid JSON."
        ) from error
    if (
        not isinstance(policy, dict)
        or not isinstance(policy.get("policy_version"), str)
        or not policy["policy_version"]
        or not isinstance(policy.get("policies"), list)
    ):
        raise HTTPException(
            status_code=503, detail="VOLUNTEER_SCOPE_POLICY_JSON has an invalid schema."
        )
    return t.cast(dict[str, object], policy)


def language_group(language: str) -> str | None:
    """Return the checked language group for a language code.

    Args:
        language:
            Requested language code.

    Returns:
        Matching group label, or None when unsupported.
    """
    canonical = LANGUAGE_PARENTS.get(language, language)
    return next((name for name, codes in GROUPS.items() if canonical in codes), None)


def validate_scope_policy_version(euroeval_version: str) -> None:
    """Ensure trusted configuration contains entries for the EuroEval version.

    Args:
        euroeval_version:
            Installed or requested EuroEval version.

    Raises:
        HTTPException:
            Configuration is invalid or version is unsupported.
    """
    policy = _load_policy()
    version = _canonical_version(euroeval_version)
    entries = t.cast(list[object], policy["policies"])
    if not any(
        isinstance(item, dict) and item.get("euroeval_version") == version
        for item in entries
    ):
        raise HTTPException(
            status_code=503,
            detail=f"Trusted scope policy has no entries for EuroEval {version}.",
        )
