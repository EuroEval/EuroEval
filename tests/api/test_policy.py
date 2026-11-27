"""Tests for trusted volunteer scope policy behaviour."""

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from euroeval_api.policy import (
    expected_scope,
    language_group,
    validate_scope_policy_version,
)


@pytest.fixture
def checked_version(monkeypatch: pytest.MonkeyPatch) -> str:
    """Select the checked-in release without inheriting a service policy override.

    Args:
        monkeypatch:
            Isolated test environment.

    Returns:
        EuroEval version carried by the checked-in policy.
    """
    monkeypatch.delenv("VOLUNTEER_SCOPE_POLICY_JSON", raising=False)
    policy = json.loads(Path("api/worker/scope-policy.json").read_text())
    return policy["policies"][0]["euroeval_version"]


def test_expected_scope_returns_independent_lists(checked_version: str) -> None:
    """Returned scope can be mutated without changing subsequent trusted results.

    Args:
        checked_version:
            Checked-in policy release.
    """
    scope = expected_scope(checked_version, "encoder", "no")
    scope["task_groups"].clear()
    assert expected_scope(checked_version, "encoder", "no")["task_groups"]


def test_grouped_variants_require_an_exact_trusted_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Language aliases share a group but cannot inherit another language's scope.

    Args:
        monkeypatch:
            Isolated trusted policy override.
    """
    suffix = '["norec",true,false]'
    policy = {
        "policy_version": "test",
        "policies": [
            {
                "euroeval_version": "1",
                "model_type": "encoder",
                "language": "no",
                "language_group": language_group(language="no"),
                "allowed_identity_suffix_sets": [[suffix]],
                "task_groups": ["classification"],
                "warnings": [],
            }
        ],
    }
    monkeypatch.setenv("VOLUNTEER_SCOPE_POLICY_JSON", json.dumps(policy))
    validate_scope_policy_version(euroeval_version="1")
    assert (
        language_group(language="nb")
        == language_group(language="nn")
        == language_group(language="no")
    )
    assert (
        language_group(language="pt-br")
        == language_group(language="pt-pt")
        == language_group(language="pt")
    )
    scope = expected_scope(euroeval_version="1", model_type="encoder", language="no")
    assert scope["allowed_identity_suffix_sets"] == [[suffix]]
    with pytest.raises(HTTPException) as error:
        expected_scope(euroeval_version="1", model_type="encoder", language="nb")
    assert error.value.status_code == 422


def test_override_rejects_duplicate_identity_alternative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed trusted alternatives fail as configuration errors.

    Args:
        monkeypatch:
            Isolated trusted policy override.
    """
    policy = {
        "policy_version": "test",
        "policies": [
            {
                "euroeval_version": "1",
                "model_type": "encoder",
                "language": "en",
                "language_group": "English",
                "allowed_identity_suffix_sets": [
                    ['["x",true,true]'],
                    ['["x",true,true]'],
                ],
                "task_groups": ["classification"],
                "warnings": [],
            }
        ],
    }
    monkeypatch.setenv("VOLUNTEER_SCOPE_POLICY_JSON", json.dumps(policy))
    with pytest.raises(HTTPException) as error:
        expected_scope("1", "encoder", "en")
    assert error.value.status_code == 503
