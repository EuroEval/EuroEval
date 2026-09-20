"""Tests for the generated volunteer scope policy."""

import json
import tomllib
import typing as t
from pathlib import Path

import pytest

import src.scripts.generate_volunteer_scope_policy as policy_module
from euroeval.benchmarker import Benchmarker
from src.scripts.generate_volunteer_scope_policy import build_policy


class PolicyEntry(t.TypedDict):
    """One generated model-type/language policy entry."""

    model_type: str
    language: str
    identity_suffixes: list[str]
    language_group: str


class Policy(t.TypedDict):
    """Generated policy shape used by these tests."""

    policy_version: str
    policies: list[PolicyEntry]


def test_checked_in_json_and_typescript_policies_are_synchronised() -> None:
    """The checked-in cross-language artifacts encode one policy."""
    json_path = Path("api/worker/scope-policy.json")
    typescript_path = Path("api/worker/_lib/scope-policy.generated.ts")
    policy = json.loads(json_path.read_text(encoding="utf-8"))

    assert (
        policy_module.encode_typescript_policy(policy) == typescript_path.read_bytes()
    )


def test_development_policy_version_uses_pep440_normalisation() -> None:
    """Policy versions normalise the package's trailing development marker."""
    policy = build_policy(
        "18.1.0.dev", {("multi-wiki-qa-da", "da")}, model_types=("encoder",)
    )

    assert policy["policy_version"] == "volunteer-scope/18.1.0.dev0"


def test_package_version_matches_authoritative_release() -> None:
    """The package uses the current development release notation."""
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["version"] == "18.1.0.dev"


def test_policy_does_not_share_a_group_scope() -> None:
    """A policy entry must not widen one language to its checkbox group."""
    policy = build_policy(
        "18.1.0", {("multi-wiki-qa-da", "da")}, model_types=("generative",)
    )

    entry = t.cast(Policy, policy)["policies"][0]
    assert entry["language"] == "da"
    assert entry["language_group"] == "da"


def test_policy_generation_fails_on_config_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config loading failure must not widen scope to every pair."""

    def fail() -> dict[str, object]:
        raise RuntimeError("config lookup failed")

    monkeypatch.setattr(policy_module, "_configs_by_name", fail)
    with pytest.raises(RuntimeError, match="config lookup failed"):
        build_policy(
            "18.0.0", {("multi-wiki-qa-da", "da")}, model_types=("generative",)
        )


def test_policy_is_versioned_and_exact_language() -> None:
    """Policies pin version, model type, and individual ISO languages."""
    policy = build_policy(
        "18.1.0",
        {("multi-wiki-qa-da", "da"), ("multi-wiki-qa-en", "en")},
        model_types=("encoder",),
    )

    assert policy["policy_version"] == "volunteer-scope/18.1.0"
    entries = t.cast(Policy, policy)["policies"]
    assert {(entry["model_type"], entry["language"]) for entry in entries} == {
        ("encoder", "da"),
        ("encoder", "en"),
    }
    assert entries[0]["identity_suffixes"] == ['["multi-wiki-qa-da",false,true]']


def test_policy_matches_benchmarker_defaults_for_encoder_and_decoder() -> None:
    """Generated identities mirror planned Benchmarker values by model type."""
    planned = Benchmarker(
        progress_bar=False,
        save_results=False,
        language="da",
        dataset="multi-wiki-qa-da",
        few_shot=True,
    )
    policy = build_policy(
        "18.1.0.dev0",
        {("multi-wiki-qa-da", "da"), ("ifeval-da", "da")},
        model_types=("encoder", "generative"),
    )
    by_type = {
        entry["model_type"]: entry for entry in t.cast(Policy, policy)["policies"]
    }
    assert planned.benchmark_config_default_params.few_shot is True
    assert planned.benchmark_config_default_params.evaluate_test_split is False
    assert by_type["encoder"]["identity_suffixes"] == [
        '["multi-wiki-qa-da",false,true]'
    ]
    assert by_type["generative"]["identity_suffixes"] == [
        '["ifeval-da",false,true]',
        '["multi-wiki-qa-da",false,true]',
    ]
