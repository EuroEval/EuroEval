"""Tests for the generated volunteer scope policy."""

from src.scripts.generate_volunteer_scope_policy import build_policy


def test_policy_is_versioned_and_exact_language() -> None:
    """Policies pin version, profile, and individual ISO languages."""
    policy = build_policy(
        "18.0.0",
        {("dataset-da", "da"), ("dataset-en", "en")},
        profiles=("bert",),
    )

    assert policy["policy_version"] == "volunteer-scope/18.0.0"
    entries = policy["policies"]
    assert {(entry["model_profile"], entry["language"]) for entry in entries} == {
        ("bert", "da"),
        ("bert", "en"),
    }
    assert entries[0]["identity_suffixes"] == ['["dataset-da",true,false]']


def test_policy_does_not_share_a_group_scope() -> None:
    """A policy entry must not widen one language to its checkbox group."""
    policy = build_policy("18.0.0", {("dataset-da", "da")}, profiles=("qwen",))

    entry = policy["policies"][0]
    assert entry["language"] == "da"
    assert entry["language_group"] == "da"
