"""Tests for the public terminal adapter for result conflicts."""

from __future__ import annotations

import pytest

from leaderboards.result_conflict_prompt import resolve_result_interactively
from leaderboards.result_conflicts import ConflictChoice, ConflictReason
from leaderboards.result_identity import UnresolvedResultConflictError


def test_automatic_policy_does_not_prompt() -> None:
    """A richer stored metadata record wins without asking for confirmation."""
    current = _record()
    candidate = {
        **current,
        "model_info": {
            "id": "org/model",
            "additional_details": {"num_model_parameters": "-1"},
        },
    }
    decision = resolve_result_interactively(
        current=current,
        candidate=candidate,
        input_fn=lambda _prompt: (_ for _ in ()).throw(
            AssertionError("automatic policy must not prompt")
        ),
        interactive=True,
    )
    assert decision.selected == ConflictChoice.STORED
    assert decision.reason == ConflictReason.STORED_METADATA_SUPERSET


def _record() -> dict[str, object]:
    """Return a complete record with a stable identity and evaluation time."""
    return {
        "model_info": {
            "id": "org/model",
            "additional_details": {
                "num_model_parameters": 8030261248,
                "trained_from_scratch": False,
                "commercially_licensed": True,
                "open": True,
                "release_date": "2024-06-01",
                "model_url": "https://example.test/model",
            },
        },
        "dataset": "task",
        "validation_split": None,
        "few_shot": True,
        "eval_library": {"additional_details": {"dataset": "task"}},
        "scores": {"accuracy": 0.5},
        "retrieved_timestamp": 100,
    }


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("s", ConflictChoice.STORED), ("i", ConflictChoice.INCOMING)],
)
def test_explicit_choice_returns_bound_decision(
    answer: str, expected: ConflictChoice
) -> None:
    """Both supported choices are represented as explicit decisions."""
    current = _record()
    candidate = {**current, "scores": {"accuracy": 0.8}}
    decision = resolve_result_interactively(
        current=current,
        candidate=candidate,
        input_fn=lambda _prompt: answer,
        interactive=True,
    )
    assert decision.selected == expected
    assert decision.reason == ConflictReason.EXPLICIT_CHOICE


def test_invalid_answer_reprompts_and_shows_field_context() -> None:
    """Invalid input is retried and output identifies changed fields and values."""
    answers = iter(["maybe", "incoming"])
    output: list[str] = []
    decision = resolve_result_interactively(
        current=_record(),
        candidate={**_record(), "scores": {"accuracy": 0.8}},
        input_fn=lambda _prompt: next(answers),
        output_fn=output.append,
        interactive=True,
    )
    assert decision.selected == ConflictChoice.INCOMING
    assert any("scores.accuracy" in line for line in output)
    assert any("stored=" in line and "incoming=" in line for line in output)
    assert any("Evaluation timestamps" in line for line in output)
    assert any("Invalid choice" in line for line in output)


@pytest.mark.parametrize("failure", [EOFError, KeyboardInterrupt])
def test_terminal_input_abort_is_controlled(failure: type[BaseException]) -> None:
    """EOF and interrupts become the adapter's controlled abort exception."""

    def fail(_prompt: str) -> str:
        raise failure

    with pytest.raises(UnresolvedResultConflictError, match="aborted"):
        resolve_result_interactively(
            current=_record(),
            candidate={**_record(), "scores": {"accuracy": 0.8}},
            input_fn=fail,
            interactive=True,
        )


def test_unresolved_conflict_fails_closed_without_tty() -> None:
    """A non-interactive process cannot silently choose a conflicting record."""
    with pytest.raises(UnresolvedResultConflictError):
        resolve_result_interactively(
            current=_record(),
            candidate={**_record(), "scores": {"accuracy": 0.8}},
            interactive=False,
        )
