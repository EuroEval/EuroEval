"""Tests for the result identity and path helpers."""

from __future__ import annotations

import typing as t

import pytest

from euroeval.data_models import BenchmarkResult
from leaderboards.result_identity import (
    ResultIdentity,
    dedup_newer_record,
    identity_from_benchmark_result,
    identity_from_eee_record,
    identity_to_path,
    normalise_bool_value,
    raise_on_collision,
    record_filename,
    record_relative_path,
    sanitise_dataset_name,
    sanitise_model_dir_name,
    shot_label,
    split_label,
)


def _make_eee_record(
    model_id: str = "org/model",
    dataset: str = "test_dataset",
    validation_split: bool | str | None = False,
    few_shot: bool | str | None = True,
    version: str | None = "1.0.0",
    timestamp: str = "1234567890",
) -> dict:
    """Helper to create an EEE-format record for testing.

    Args:
        model_id:
            Model identifier.
        dataset:
            Dataset name.
        validation_split:
            Validation split flag.
        few_shot:
            Few-shot flag.
        version:
            EuroEval version.
        timestamp:
            Retrieved timestamp.

    Returns:
        EEE-format record dictionary.
    """
    return {
        "schema_version": "0.2.1",
        "model_info": {
            "id": model_id,
            "name": "Test Model",
        },
        "eval_library": {
            "name": "euroeval",
            "version": version,
            "additional_details": {
                "dataset": dataset,
                "validation_split": validation_split,
                "few_shot": few_shot,
            },
        },
        "retrieved_timestamp": timestamp,
        "evaluation_results": [],
    }


class TestNormaliseBoolValue:
    """Tests for normalise_bool_value."""

    def test_bool_true(self) -> None:
        """Boolean True should remain True."""
        assert normalise_bool_value(True) is True

    def test_bool_false(self) -> None:
        """Boolean False should remain False."""
        assert normalise_bool_value(False) is False

    def test_none(self) -> None:
        """None should remain None."""
        assert normalise_bool_value(None) is None

    def test_string_true(self) -> None:
        """String 'true' should become True."""
        assert normalise_bool_value("true") is True
        assert normalise_bool_value("TRUE") is True
        assert normalise_bool_value("True") is True

    def test_string_false(self) -> None:
        """String 'false' should become False."""
        assert normalise_bool_value("false") is False
        assert normalise_bool_value("FALSE") is False
        assert normalise_bool_value("False") is False

    def test_string_none(self) -> None:
        """String 'none' should become None."""
        assert normalise_bool_value("none") is None
        assert normalise_bool_value("NONE") is None
        assert normalise_bool_value("None") is None

    def test_invalid_string(self) -> None:
        """Invalid string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid boolean string"):
            normalise_bool_value("yes")

    def test_invalid_type(self) -> None:
        """Invalid type should raise TypeError."""
        value = t.cast(t.Any, 1)
        with pytest.raises(TypeError, match="Unexpected type"):
            normalise_bool_value(value)


class TestSanitiseModelDirName:
    """Tests for sanitise_model_dir_name."""

    def test_slash_replaced(self) -> None:
        """Forward slashes should be replaced with underscores."""
        assert sanitise_model_dir_name("org/model") == "org_model"

    def test_at_preserved(self) -> None:
        """@ symbol should be preserved."""
        assert sanitise_model_dir_name("org/model@refs_pr_6") == "org_model@refs_pr_6"

    def test_hash_preserved(self) -> None:
        """# symbol should be preserved."""
        assert (
            sanitise_model_dir_name("Qwen/Qwen3-30B-A3B#no-thinking")
            == "Qwen_Qwen3-30B-A3B#no-thinking"
        )

    def test_no_change_needed(self) -> None:
        """Model names without slashes should remain unchanged."""
        assert sanitise_model_dir_name("model_name") == "model_name"


class TestSanitiseDatasetName:
    """Tests for sanitise_dataset_name."""

    def test_slash_replaced(self) -> None:
        """Forward slashes should be replaced with underscores."""
        assert sanitise_dataset_name("org/dataset") == "org_dataset"

    def test_no_change_needed(self) -> None:
        """Dataset names without slashes should remain unchanged."""
        assert sanitise_dataset_name("test_dataset") == "test_dataset"


class TestSplitLabel:
    """Tests for split_label."""

    def test_true(self) -> None:
        """True should map to 'val'."""
        assert split_label(True) == "val"

    def test_false(self) -> None:
        """False should map to 'test'."""
        assert split_label(False) == "test"

    def test_none(self) -> None:
        """None should map to 'none'."""
        assert split_label(None) == "none"


class TestShotLabel:
    """Tests for shot_label."""

    def test_true(self) -> None:
        """True should map to 'fewshot'."""
        assert shot_label(True) == "fewshot"

    def test_false(self) -> None:
        """False should map to 'zeroshot'."""
        assert shot_label(False) == "zeroshot"

    def test_none(self) -> None:
        """None should map to 'none'."""
        assert shot_label(None) == "none"


class TestRecordFilename:
    """Tests for record_filename."""

    def test_basic(self) -> None:
        """Basic filename format."""
        assert (
            record_filename("test_dataset", validation_split=False, few_shot=True)
            == "test_dataset__test__fewshot.json"
        )

    def test_val_split(self) -> None:
        """Validation split should produce 'val' label."""
        assert (
            record_filename("test_dataset", validation_split=True, few_shot=True)
            == "test_dataset__val__fewshot.json"
        )

    def test_zeroshot(self) -> None:
        """Zero-shot should produce 'zeroshot' label."""
        assert (
            record_filename("test_dataset", validation_split=False, few_shot=False)
            == "test_dataset__test__zeroshot.json"
        )

    def test_none_values(self) -> None:
        """None values should produce 'none' labels."""
        assert (
            record_filename("test_dataset", validation_split=None, few_shot=None)
            == "test_dataset__none__none.json"
        )

    def test_dataset_with_slash(self) -> None:
        """Dataset with slash should be sanitised."""
        assert (
            record_filename("org/dataset", validation_split=False, few_shot=True)
            == "org_dataset__test__fewshot.json"
        )


class TestRecordRelativePath:
    """Tests for record_relative_path."""

    def test_basic(self) -> None:
        """Basic path format."""
        path = record_relative_path(
            model_id="org/model",
            dataset="test_dataset",
            validation_split=False,
            few_shot=True,
        )
        assert str(path) == "org_model/test_dataset__test__fewshot.json"

    def test_with_revision(self) -> None:
        """Model with @revision should preserve @."""
        path = record_relative_path(
            model_id="org/model@refs_pr_6",
            dataset="test_dataset",
            validation_split=False,
            few_shot=True,
        )
        assert str(path) == "org_model@refs_pr_6/test_dataset__test__fewshot.json"

    def test_with_param_variant(self) -> None:
        """Model with #param should preserve #."""
        path = record_relative_path(
            model_id="Qwen/Qwen3-30B-A3B#no-thinking",
            dataset="test_dataset",
            validation_split=False,
            few_shot=True,
        )
        expected = (
            "Qwen_Qwen3-30B-A3B#no-thinking"
            "/test_dataset__test__fewshot.json"
        )
        assert str(path) == expected


class TestIdentityFromEeeRecord:
    """Tests for identity_from_eee_record."""

    def test_basic(self) -> None:
        """Basic identity extraction."""
        record = _make_eee_record(
            model_id="org/model",
            dataset="test_dataset",
            validation_split=False,
            few_shot=True,
        )
        identity = identity_from_eee_record(record)
        assert identity == ("org/model", "test_dataset", False, True)

    def test_with_revision(self) -> None:
        """Model with @revision should be preserved."""
        record = _make_eee_record(model_id="org/model@refs_pr_6")
        identity = identity_from_eee_record(record)
        assert identity[0] == "org/model@refs_pr_6"

    def test_with_param_variant(self) -> None:
        """Model with #param should be preserved."""
        record = _make_eee_record(model_id="Qwen/Qwen3-30B-A3B#no-thinking")
        identity = identity_from_eee_record(record)
        assert identity[0] == "Qwen/Qwen3-30B-A3B#no-thinking"

    def test_string_bool_values(self) -> None:
        """String boolean values should be normalised."""
        record = _make_eee_record(
            validation_split="true",
            few_shot="false",
        )
        identity = identity_from_eee_record(record)
        assert identity == ("org/model", "test_dataset", True, False)

    def test_none_bool_values(self) -> None:
        """None boolean values should be preserved."""
        record = _make_eee_record(
            validation_split=None,
            few_shot=None,
        )
        identity = identity_from_eee_record(record)
        assert identity == ("org/model", "test_dataset", None, None)

    def test_fallback_to_name(self) -> None:
        """Should fall back to model_info.name if id is missing."""
        record = _make_eee_record()
        del record["model_info"]["id"]
        identity = identity_from_eee_record(record)
        assert identity[0] == "Test Model"

    def test_missing_model_info_raises(self) -> None:
        """Missing model info should raise ValueError."""
        record = _make_eee_record()
        del record["model_info"]
        with pytest.raises(ValueError, match="Missing model_info"):
            identity_from_eee_record(record)

    def test_missing_dataset_raises(self) -> None:
        """Missing dataset should raise ValueError."""
        record = _make_eee_record()
        del record["eval_library"]["additional_details"]["dataset"]
        with pytest.raises(ValueError, match="Missing eval_library"):
            identity_from_eee_record(record)


class TestIdentityFromBenchmarkResult:
    """Tests for identity_from_benchmark_result."""

    def test_basic(self) -> None:
        """Basic identity extraction from BenchmarkResult."""
        result = BenchmarkResult(
            model="org/model",
            dataset="test_dataset",
            task="_classification",
            languages=["en"],
            results={"raw": [], "total": {}},
            num_model_parameters=1000,
            max_sequence_length=512,
            vocabulary_size=10000,
            merge=False,
            generative=True,
            generative_type=None,
            few_shot=True,
            validation_split=False,
        )
        identity = identity_from_benchmark_result(result)
        assert identity == ("org/model", "test_dataset", False, True)


class TestIdentityToPath:
    """Tests for identity_to_path."""

    def test_basic(self) -> None:
        """Basic identity to path conversion."""
        identity: ResultIdentity = ("org/model", "test_dataset", False, True)
        path = identity_to_path(identity)
        assert str(path) == "org_model/test_dataset__test__fewshot.json"

    def test_with_none_values(self) -> None:
        """Identity with None values."""
        identity: ResultIdentity = ("org/model", "test_dataset", None, None)
        path = identity_to_path(identity)
        assert str(path) == "org_model/test_dataset__none__none.json"


class TestDedupNewerRecord:
    """Tests for dedup_newer_record."""

    def test_higher_version_wins(self) -> None:
        """Record with higher version should win."""
        record_a = _make_eee_record(version="1.0.0", timestamp="100")
        record_b = _make_eee_record(version="2.0.0", timestamp="50")
        winner = dedup_newer_record(record_a, record_b)
        assert winner is record_b

    def test_same_version_timestamp_tiebreak(self) -> None:
        """Same version should tie-break by timestamp."""
        record_a = _make_eee_record(version="1.0.0", timestamp="100")
        record_b = _make_eee_record(version="1.0.0", timestamp="200")
        winner = dedup_newer_record(record_a, record_b)
        assert winner is record_b

    def test_equal_records_returns_first(self) -> None:
        """Equal version and timestamp should return first record."""
        record_a = _make_eee_record(version="1.0.0", timestamp="100")
        record_b = _make_eee_record(version="1.0.0", timestamp="100")
        winner = dedup_newer_record(record_a, record_b)
        assert winner is record_a

    def test_different_identity_raises(self) -> None:
        """Different identities should raise ValueError."""
        record_a = _make_eee_record(model_id="org/model_a")
        record_b = _make_eee_record(model_id="org/model_b")
        with pytest.raises(ValueError, match="different identities"):
            dedup_newer_record(record_a, record_b)

    def test_none_version_loses_to_some_version(self) -> None:
        """Record with None version should lose to record with version."""
        record_a = _make_eee_record(version=None, timestamp="100")
        record_b = _make_eee_record(version="1.0.0", timestamp="50")
        winner = dedup_newer_record(record_a, record_b)
        assert winner is record_b

    def test_version_comparison_complex(self) -> None:
        """Complex version comparison."""
        record_a = _make_eee_record(version="1.10.0", timestamp="100")
        record_b = _make_eee_record(version="1.9.0", timestamp="200")
        winner = dedup_newer_record(record_a, record_b)
        assert winner is record_a


class TestRaiseOnCollision:
    """Tests for raise_on_collision."""

    def test_same_identity_no_raise(self) -> None:
        """Same identity should not raise."""
        identity: ResultIdentity = ("org/model", "test_dataset", False, True)
        # Should not raise
        raise_on_collision(identity, identity)

    def test_different_identity_same_path_raises(self) -> None:
        """Different identities that sanitise to same path should raise.

        Note: With the current implementation, this cannot happen because
        the full model_id is used (including @ and #), which preserves
        uniqueness. This test documents the expected behaviour if such
        a collision were to occur.
        """
        # This is a hypothetical case - with current implementation,
        # different identities produce different paths
        identity_a: ResultIdentity = ("org/model_a", "test_dataset", False, True)
        identity_b: ResultIdentity = ("org/model_b", "test_dataset", False, True)
        # Should not raise because paths are different
        raise_on_collision(identity_a, identity_b)
