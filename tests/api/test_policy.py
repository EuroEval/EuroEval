"""Tests for trusted volunteer scope policy behavior."""

import json

import pytest

from euroeval_api.policy import (
    ScopePolicyError,
    expected_scope,
    language_group,
    validate_scope_policy_version,
)


def test_checked_in_policy_supports_grouped_language_variants() -> None:
    """Language aliases share a group while scope records remain language-specific."""
    validate_scope_policy_version("18.4.0.dev")
    assert language_group("nb") == language_group("nn") == language_group("no")
    assert language_group("pt-br") == language_group("pt-pt") == language_group("pt")
    with pytest.raises(ScopePolicyError) as error:
        expected_scope("18.4.0.dev0", "encoder", "nb")
    assert error.value.status_code == 422


def test_expected_scope_returns_independent_lists() -> None:
    """Returned scope can be mutated without changing subsequent trusted results."""
    scope = expected_scope("18.4.0.dev", "encoder", "no")
    scope["task_groups"].clear()
    assert expected_scope("18.4.0.dev0", "encoder", "no")["task_groups"]


def test_override_rejects_duplicate_identity_alternative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed trusted alternatives fail as configuration errors."""
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
    with pytest.raises(ScopePolicyError) as error:
        expected_scope("1", "encoder", "en")
    assert error.value.status_code == 503
