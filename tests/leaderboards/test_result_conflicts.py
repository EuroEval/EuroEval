"""Public-interface tests for result conflict resolution."""

from __future__ import annotations

import copy
import typing as t

import pytest

from leaderboards.result_conflicts import (
    ConflictChoice,
    ConflictReason,
    resolve_result_conflict,
)


class TestResolveResultConflict:
    """Exercise policy via its immutable public decision interface."""

    def test_anchor_normalisation_does_not_cross_model_revisions(self) -> None:
        """Identifier presentation cannot erase model revision identity."""
        anchored = _record(
            model_info={"id": "<a href='https://example.org'>org/model@v1</a>"}
        )
        plain = _record(model_info={"id": "org/model@v2"})
        with pytest.raises(ValueError, match="different identities"):
            resolve_result_conflict(current=anchored, candidate=plain)

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

    def test_eval_library_configuration_addition_requires_choice(self) -> None:
        """Configuration additions are substantive even when named like metadata."""
        current = _record(eval_library={"name": "euroeval", "config": {}})
        candidate = _record(
            eval_library={"name": "euroeval", "config": {"few_shot": True}}
        )
        decision = resolve_result_conflict(current=current, candidate=candidate)
        assert decision.needs_choice

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
        model_id = "AI-Sweden-Models/Llama-3-8B-instruct"
        current = _eee_model_record(open=False, custom_zero_extension=0)
        candidate = _record(model_info={"id": model_id})
        decision = resolve_result_conflict(current=current, candidate=candidate)
        assert decision.selected == ConflictChoice.STORED

    def test_identifier_format_does_not_hide_score_or_metadata_changes(self) -> None:
        """Plain identifiers do not resolve substantive score or metadata changes."""
        anchored = _record(
            model_info={"id": "<a href='https://example.org'>org/model</a>"},
            scores={"accuracy": 0.4},
        )
        score_change = _record(model_info={"id": "org/model"}, scores={"accuracy": 0.5})
        anchored["open"] = False
        metadata_change = _record(
            model_info={"id": "org/model"}, open=True, scores={"accuracy": 0.4}
        )
        assert resolve_result_conflict(
            current=anchored, candidate=score_change
        ).needs_choice
        assert resolve_result_conflict(
            current=anchored, candidate=metadata_change
        ).needs_choice

    def test_identity_mismatch_raises(self) -> None:
        """Resolver refuses to decide across distinct result identities."""
        with pytest.raises(ValueError, match="different identities"):
            resolve_result_conflict(
                current=_record(),
                candidate=_record(
                    eval_library={"additional_details": {"dataset": "other"}}
                ),
            )

    @pytest.mark.parametrize(
        ("stored", "incoming", "expected"),
        [
            ({"open": True}, {}, ConflictChoice.STORED),
            ({}, {"open": True}, ConflictChoice.INCOMING),
            ({"open": False}, {}, ConflictChoice.STORED),
            (
                {"num_model_parameters": 803},
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

    @pytest.mark.parametrize(
        ("current_is_plain", "expected"),
        [(False, ConflictChoice.INCOMING), (True, ConflictChoice.STORED)],
    )
    def test_legacy_identifier_anchors_prefer_plain_record_without_mutation(
        self, current_is_plain: bool, expected: ConflictChoice
    ) -> None:
        """Presentation-only anchors select the exact plain record."""
        plain = _record(
            model_info={
                "id": "ollama_chat/gemma3:4b-it-q8_0",
                "name": "ollama_chat/gemma3:4b-it-q8_0",
            },
            evaluation_id=("angry-tweets/ollama_chat/gemma3:4b-it-q8_0/1781766560"),
        )
        anchored = copy.deepcopy(plain)
        anchor = (
            "<a href='https://ollama.com/library/gemma3:4b-it-q8_0'>"
            "ollama_chat/gemma3:4b-it-q8_0</a>"
        )
        anchored_model = t.cast(dict[str, object], anchored["model_info"])
        anchored_model["id"] = anchor
        anchored_model["name"] = anchor
        anchored["evaluation_id"] = f"angry-tweets/{anchor}/1781766560"
        original = copy.deepcopy(anchored)
        current, candidate = (
            (plain, anchored) if current_is_plain else (anchored, plain)
        )
        decision = resolve_result_conflict(current=current, candidate=candidate)
        assert decision.selected == expected
        assert not decision.needs_choice
        assert current == (plain if current_is_plain else original)
        assert candidate == (original if current_is_plain else plain)

    def test_library_version_only_change_is_bookkeeping(self) -> None:
        """A library version difference neither changes results nor establishes age."""
        current = _record(eval_library={"version": "99.0"})
        candidate = _record(eval_library={"version": "1.0"})
        decision = resolve_result_conflict(current=current, candidate=candidate)
        assert decision.selected == ConflictChoice.STORED
        assert decision.reason == ConflictReason.IDENTICAL

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

    @pytest.mark.parametrize(
        ("current_key", "candidate_key", "expected"),
        [
            ("sparse", "rich", ConflictChoice.INCOMING),
            ("rich", "sparse", ConflictChoice.STORED),
        ],
    )
    def test_model_metadata_enrichment_ignores_library_version(
        self, current_key: str, candidate_key: str, expected: ConflictChoice
    ) -> None:
        """Known and extension model metadata wins regardless of library version."""
        records = {
            "sparse": _eee_model_record(
                eval_library={"name": "euroeval", "version": "99.0"}
            ),
            "rich": _eee_model_record(
                generative_type="instruction-tuned",
                custom_model_metadata="verified",
                eval_library={"name": "euroeval", "version": "1.0"},
            ),
        }
        decision = resolve_result_conflict(
            current=records[current_key], candidate=records[candidate_key]
        )
        assert decision.selected == expected

    def test_nested_eee_raw_payload_change_uses_actual_evaluation_time(self) -> None:
        """Only evaluation chronology, not raw metric timestamps, selects a payload."""
        current = _record(
            evaluation_timestamp="2024-01-01T00:00:00Z",
            eval_library={
                "name": "euroeval",
                "additional_details": {
                    "raw_results": '{"timestamp":"2030-01-01", "score":0.4}'
                },
            },
        )
        candidate = _record(
            evaluation_timestamp="2025-01-01T00:00:00Z",
            eval_library={
                "name": "euroeval",
                "additional_details": {
                    "raw_results": '{"timestamp":"2020-01-01", "score":0.5}'
                },
            },
        )
        decision = resolve_result_conflict(current=current, candidate=candidate)
        assert decision.selected == ConflictChoice.INCOMING
        assert decision.reason == ConflictReason.NEWER_EVALUATION

    @pytest.mark.parametrize(
        "representations",
        [
            ('{"a": 1, "b": [2, 3]}', {"b": [2, 3], "a": 1}),
            ('[{"score": 0.5}]', [{"score": 0.5}]),
        ],
    )
    def test_nested_eee_raw_results_representations_are_equivalent(
        self, representations: tuple[object, object]
    ) -> None:
        """EEE raw results compare equally as JSON strings and decoded values."""
        current = _record(
            eval_library={
                "name": "euroeval",
                "additional_details": {"raw_results": representations[0]},
            }
        )
        candidate = _record(
            eval_library={
                "name": "euroeval",
                "additional_details": {"raw_results": representations[1]},
            }
        )
        decision = resolve_result_conflict(current=current, candidate=candidate)
        assert decision.selected == ConflictChoice.STORED
        assert decision.reason == ConflictReason.IDENTICAL

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

    def test_raw_metric_timestamp_does_not_establish_recency(self) -> None:
        """A timestamp inside EEE score data is payload, not chronology evidence."""
        current = _record(
            eval_library={
                "name": "euroeval",
                "additional_details": {
                    "raw_results": '{"timestamp":"2024-01-01", "score":0.4}'
                },
            }
        )
        candidate = _record(
            eval_library={
                "name": "euroeval",
                "additional_details": {
                    "raw_results": '{"timestamp":"2025-01-01", "score":0.5}'
                },
            }
        )
        decision = resolve_result_conflict(current=current, candidate=candidate)
        assert decision.needs_choice
        assert decision.reason == ConflictReason.UNRESOLVED_TIMESTAMP

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

    def test_richer_anchored_record_beats_poorer_plain_record(self) -> None:
        """Plain presentation cannot discard useful metadata."""
        anchored = _record(
            model_info={
                "id": "<a href='https://example.org'>org/model</a>",
                "num_model_parameters": 803,
            }
        )
        plain = _record(model_info={"id": "org/model"})
        decision = resolve_result_conflict(current=anchored, candidate=plain)
        assert decision.selected == ConflictChoice.STORED

    def test_root_score_alias_is_payload_and_newer_evaluation_wins(self) -> None:
        """A singular root score is payload alongside canonical results."""
        current = _record(
            eval_timestamp="2025-01-01T00:00:00Z",
            score=0.4,
            evaluation_results={"accuracy": 0.4},
        )
        candidate = _record(
            eval_timestamp="2026-01-01T00:00:00Z",
            score=0.5,
            evaluation_results={"accuracy": 0.5},
        )

        decision = resolve_result_conflict(current=current, candidate=candidate)

        assert decision.selected == ConflictChoice.INCOMING
        assert decision.reason == ConflictReason.NEWER_EVALUATION
        assert any(item.field == "score" for item in decision.differences)

    def test_unknown_metadata_change_requires_choice(self) -> None:
        """Unknown custom fields are not treated as safe metadata additions."""
        decision = resolve_result_conflict(
            current=_record(custom_metadata="-1"), candidate=_record()
        )
        assert decision.needs_choice

    def test_unrelated_html_payload_remains_substantive(self) -> None:
        """HTML outside recognised identifier fields is not normalised."""
        current = _record(raw_results="<a href='x'>same</a>")
        candidate = _record(raw_results="same")
        decision = resolve_result_conflict(current=current, candidate=candidate)
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
    eval_library = details.pop("eval_library", None)
    record = (
        _record(eval_library=eval_library) if eval_library is not None else _record()
    )
    record["model_info"] = {
        "id": "AI-Sweden-Models/Llama-3-8B-instruct",
        "additional_details": details,
    }
    return record


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
    }
    supplied_library = changes.get("eval_library")
    record.update(
        {key: value for key, value in changes.items() if key != "eval_library"}
    )
    if isinstance(supplied_library, dict):
        base_library = t.cast(dict[str, object], copy.deepcopy(record["eval_library"]))
        supplied_details = t.cast(
            dict[str, object], supplied_library.get("additional_details", {})
        )
        base_details = t.cast(dict[str, object], base_library["additional_details"])
        library = {**base_library, **copy.deepcopy(supplied_library)}
        library["additional_details"] = {**base_details, **supplied_details}
        record["eval_library"] = library
    return record
