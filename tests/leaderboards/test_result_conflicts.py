"""Public-interface tests for result conflict resolution."""

from __future__ import annotations

import pytest

from leaderboards.result_conflicts import (
    ConflictChoice,
    ConflictReason,
    resolve_result_conflict,
)


class TestResolveResultConflict:
    """Exercise policy via its immutable public decision interface."""

    @pytest.mark.parametrize(
        ("stored", "incoming"),
        [
            ({"open": True}, {"open": False}),
            ({"open": True}, {"commercially_licensed": True}),
        ],
    )
    def test_conflicting_or_complementary_metadata_needs_choice(
        self, stored: dict[str, object], incoming: dict[str, object]
    ) -> None:
        """Neither contradictory nor complementary metadata is silently merged."""
        base = _record(model_info={"id": "org/model"})
        decision = resolve_result_conflict(
            current={**base, **stored}, candidate={**base, **incoming}
        )
        assert decision.needs_choice
        assert decision.reason == ConflictReason.UNRESOLVED_METADATA

    def test_eval_library_configuration_change_is_preserved(self) -> None:
        """Evaluation library configuration is substantive, not envelope noise."""
        current = _record(eval_library={"name": "euroeval", "config": {"batch": 8}})
        candidate = _record(eval_library={"name": "euroeval", "config": {"batch": 4}})
        decision = resolve_result_conflict(current=current, candidate=candidate)
        assert decision.needs_choice
        assert any(
            "eval_library.config.batch" == item.field for item in decision.differences
        )

    def test_explicit_choice_evidence_binds_exact_bytes(self) -> None:
        """Evidence carries the selected source and exact supplied byte hashes."""
        current_bytes = b'{ "stored": true }'
        candidate_bytes = b'{"incoming":true}'
        decision = resolve_result_conflict(
            current=_record(scores={"accuracy": 0.4}),
            candidate=_record(scores={"accuracy": 0.6}),
            current_bytes=current_bytes,
            candidate_bytes=candidate_bytes,
            choice=ConflictChoice.INCOMING,
        )
        assert decision.selected == ConflictChoice.INCOMING
        assert decision.evidence.current_sha256
        assert decision.evidence.candidate_sha256
        assert decision.evidence.choice == ConflictChoice.INCOMING
        with pytest.raises((AttributeError, TypeError)):
            setattr(decision.evidence, "choice", ConflictChoice.STORED)

    def test_false_and_zero_are_useful_values(self) -> None:
        """False and zero survive comparison rather than becoming missing."""
        base = _record(model_info={"id": "org/model"})
        decision = resolve_result_conflict(
            current={**base, "open": False, "parameter_count": 0}, candidate=base
        )
        assert decision.selected == ConflictChoice.STORED

    def test_identity_mismatch_raises(self) -> None:
        """Resolver refuses to decide across distinct result identities."""
        with pytest.raises(ValueError, match="different identities"):
            resolve_result_conflict(
                current=_record(), candidate=_record(dataset="other")
            )

    @pytest.mark.parametrize(
        ("stored", "incoming", "expected"),
        [
            ({"open": True}, {}, ConflictChoice.STORED),
            ({}, {"open": True}, ConflictChoice.INCOMING),
            ({"open": False}, {}, ConflictChoice.STORED),
            (
                {"num_model_parameters": 0},
                {"num_model_parameters": -1},
                ConflictChoice.STORED,
            ),
            (
                {"num_model_parameters": 803},
                {"num_model_parameters": "-1"},
                ConflictChoice.STORED,
            ),
        ],
    )
    def test_known_metadata_superset_both_directions(
        self,
        stored: dict[str, object],
        incoming: dict[str, object],
        expected: ConflictChoice,
    ) -> None:
        """Known values, including False and zero, outrank placeholders."""
        base = _record(model_info={"id": "org/model"})
        current = {**base, **stored}
        candidate = {**base, **incoming}
        decision = resolve_result_conflict(current=current, candidate=candidate)
        assert decision.selected == expected

    def test_library_version_change_does_not_establish_recency(self) -> None:
        """Version changes remain substantive and are not recency evidence."""
        current = _record(eval_library={"version": "99.0"})
        candidate = _record(eval_library={"version": "1.0"})
        decision = resolve_result_conflict(current=current, candidate=candidate)
        assert decision.needs_choice

    def test_mixed_metadata_and_score_difference_needs_choice(self) -> None:
        """Timestamp evidence cannot auto-resolve mixed substantive changes."""
        current = _record(evaluation_timestamp="2024-01-01T00:00:00Z")
        candidate = _record(
            evaluation_timestamp="2025-01-01T00:00:00Z",
            scores={"accuracy": 0.6},
            open=False,
        )
        decision = resolve_result_conflict(current=current, candidate=candidate)
        assert decision.needs_choice

    def test_newer_evaluation_wins_not_newer_library_version(self) -> None:
        """Reliable evaluation chronology wins independently of library version."""
        current = _record(
            evaluation_timestamp="2024-01-01T00:00:00Z",
            eval_library={"version": "1.0"},
            scores={"accuracy": 0.4},
        )
        candidate = _record(
            evaluation_timestamp="2025-01-01T00:00:00Z",
            eval_library={"version": "1.0"},
            scores={"accuracy": 0.5},
        )
        decision = resolve_result_conflict(current=current, candidate=candidate)
        assert decision.selected == ConflictChoice.INCOMING
        assert decision.reason == ConflictReason.NEWER_EVALUATION

    def test_raw_results_json_formatting_is_comparison_only(self) -> None:
        """Embedded JSON formatting does not create a substantive difference."""
        decision = resolve_result_conflict(
            current=_record(raw_results='{"a": 1, "b": [2, 3]}'),
            candidate=_record(raw_results='{"b":[2,3],"a":1}'),
        )
        assert decision.selected == ConflictChoice.STORED
        assert decision.reason == ConflictReason.IDENTICAL

    @pytest.mark.parametrize(
        ("current_key", "candidate_key", "expected"),
        [
            ("stored", "incoming", ConflictChoice.STORED),
            ("incoming", "stored", ConflictChoice.INCOMING),
        ],
    )
    def test_realistic_eee_metadata_regression_both_directions(
        self, current_key: str, candidate_key: str, expected: ConflictChoice
    ) -> None:
        """Nested EEE model metadata supersets survive placeholder regressions."""
        records = {
            "stored": _eee_model_record(
                num_model_parameters=8030261248,
                trained_from_scratch=False,
                commercially_licensed=True,
                open=True,
                release_date="2024-06-01",
                model_url="https://example.test/model",
            ),
            "incoming": _eee_model_record(num_model_parameters="-1"),
        }
        decision = resolve_result_conflict(
            current=records[current_key], candidate=records[candidate_key]
        )
        assert decision.selected == expected
        assert decision.reason in {
            ConflictReason.STORED_METADATA_SUPERSET,
            ConflictReason.INCOMING_METADATA_SUPERSET,
        }
        assert any(
            "model_info.additional_details.open" == item.field
            for item in decision.differences
        )

    def test_unknown_metadata_change_requires_choice(self) -> None:
        """Unknown custom fields are not treated as safe metadata additions."""
        decision = resolve_result_conflict(
            current=_record(custom_metadata="-1"), candidate=_record()
        )
        assert decision.needs_choice

    @pytest.mark.parametrize(
        "times",
        [
            (None, "2025-01-01T00:00:00Z"),
            ("bad", "2025-01-01T00:00:00Z"),
            ("2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
        ],
    )
    def test_unreliable_timestamp_needs_choice(
        self, times: tuple[object, object]
    ) -> None:
        """Missing, malformed and tied evaluation evidence never picks a winner."""
        current = _record(evaluation_timestamp=times[0], scores={"accuracy": 0.4})
        candidate = _record(evaluation_timestamp=times[1], scores={"accuracy": 0.5})
        decision = resolve_result_conflict(current=current, candidate=candidate)
        assert decision.needs_choice
        assert decision.reason == ConflictReason.UNRESOLVED_TIMESTAMP


def _eee_model_record(**details: object) -> dict[str, object]:
    """Build an EEE-shaped record with nested model details.

    Args:
        details: Model metadata values.

    Returns:
        A result record with a stable identity.
    """
    return {
        **_record(),
        "model_info": {
            "id": "AI-Sweden-Models/Llama-3-8B-instruct",
            "additional_details": details,
        },
    }


def _record(**changes: object) -> dict[str, object]:
    record: dict[str, object] = {
        "model_info": {"id": "AI-Sweden-Models/Llama-3-8B-instruct"},
        "dataset": "alba-mcq-pt",
        "validation_split": None,
        "few_shot": True,
        "eval_library": {
            "name": "euroeval",
            "additional_details": {
                "dataset": "alba-mcq-pt",
                "validation_split": None,
                "few_shot": True,
            },
        },
        "scores": {"accuracy": 0.5},
        "model_parameters": 8030261248,
        "trained_from_scratch": False,
        "commercially_licensed": True,
        "open": True,
        "release_date": "2024-06-01",
        "model_url": "https://example.test/model",
    }
    record.update(changes)
    return record
