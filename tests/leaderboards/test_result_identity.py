"""Tests for the result identity and path helpers."""

from __future__ import annotations

import typing as t

import pytest

from euroeval.data_models import BenchmarkResult
from leaderboards.result_identity import (
    ResultIdentity,
    UnresolvedResultConflictError,
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


class TestDedupNewerRecord:
    """Tests for the legacy automatic-selection adapter."""

    def test_different_identity_raises(self) -> None:
        """Different identities should raise ValueError."""
        record_a = _make_eee_record(model_id="org/model_a")
        record_b = _make_eee_record(model_id="org/model_b")
        with pytest.raises(ValueError, match="different identities"):
            dedup_newer_record(record_a, record_b)

    def test_known_metadata_superset_wins(self) -> None:
        """Useful stored metadata wins over placeholder candidate metadata."""
        record_a = _make_eee_record()
        record_b = _make_eee_record()
        record_a["model_info"]["num_model_parameters"] = 8030261248
        record_b["model_info"]["num_model_parameters"] = -1
        assert dedup_newer_record(record_a, record_b) is record_a

    def test_unresolved_conflict_has_distinct_error(self) -> None:
        """Ambiguous score changes are not reported as malformed-record errors."""
        record_a = _make_eee_record()
        record_b = _make_eee_record()
        record_a["scores"] = {"accuracy": 0.5}
        record_b["scores"] = {"accuracy": 0.6}
        with pytest.raises(UnresolvedResultConflictError):
            dedup_newer_record(record_a, record_b)


def _make_eee_record(
    model_id: str = "org/model",
    dataset: str = "test_dataset",
    validation_split: bool | str | None = False,
    few_shot: bool | str | None = True,
    version: str | None = "1.0.0",
    timestamp: str | int | None = "1234567890",
) -> dict:
    """Create an EEE-format test record.

    Args:
        model_id: Model identifier.
        dataset: Dataset name.
        validation_split: Evaluation split flag.
        few_shot: Few-shot flag.
        version: Evaluation library version.
        timestamp: Retrieval timestamp.

    Returns:
        An EEE-format record.
    """
    return {
        "schema_version": "0.2.1",
        "model_info": {"id": model_id, "name": "Test Model"},
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


class TestIdentityFromEeeRecord:
    """Tests for identity_from_eee_record."""

    def test_anchor_is_stripped_from_name_fallback(self) -> None:
        """Fallback model names are normalised by the same anchor rule."""
        record = _make_eee_record()
        del record["model_info"]["id"]
        record["model_info"]["name"] = (
            "<a href='https://example.com'>org/model@revision#variant</a>"
        )

        assert identity_from_eee_record(record)[0] == "org/model@revision#variant"

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

    def test_fallback_to_name(self) -> None:
        """Should fall back to model_info.name if id is missing."""
        record = _make_eee_record()
        del record["model_info"]["id"]
        identity = identity_from_eee_record(record)
        assert identity[0] == "Test Model"

    def test_missing_dataset_raises(self) -> None:
        """Missing dataset should raise ValueError."""
        record = _make_eee_record()
        del record["eval_library"]["additional_details"]["dataset"]
        with pytest.raises(ValueError, match="Missing eval_library"):
            identity_from_eee_record(record)

    def test_missing_id_and_name_raises(self) -> None:
        """Missing both id and name should raise ValueError."""
        record = _make_eee_record()
        del record["model_info"]["id"]
        del record["model_info"]["name"]
        with pytest.raises(ValueError, match="model_info.id and model_info.name"):
            identity_from_eee_record(record)

    def test_missing_model_info_raises(self) -> None:
        """Missing model info should raise ValueError."""
        record = _make_eee_record()
        del record["model_info"]
        with pytest.raises(ValueError, match="model_info.id and model_info.name"):
            identity_from_eee_record(record)

    def test_none_bool_values(self) -> None:
        """None boolean values should be preserved."""
        record = _make_eee_record(validation_split=None, few_shot=None)
        identity = identity_from_eee_record(record)
        assert identity == ("org/model", "test_dataset", None, None)

    def test_string_bool_values(self) -> None:
        """String boolean values should be normalised."""
        record = _make_eee_record(validation_split="true", few_shot="false")
        identity = identity_from_eee_record(record)
        assert identity == ("org/model", "test_dataset", True, False)

    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            (
                '<a href="https://example.com">org/model@revision#variant</a>',
                "org/model@revision#variant",
            ),
            (
                "<a href='https://example.com'>org/model@revision#variant</a>",
                "org/model@revision#variant",
            ),
        ],
    )
    def test_strips_anchor_preserving_model_suffixes(
        self, model_id: str, expected: str
    ) -> None:
        """Anchors are removed without losing revision or parameter suffixes."""
        record = _make_eee_record(model_id=model_id)
        original = {"model_info": dict(record["model_info"])}

        identity = identity_from_eee_record(record)

        assert identity[0] == expected
        assert record["model_info"] == original["model_info"]

    def test_with_param_variant(self) -> None:
        """Model with #param should be preserved."""
        record = _make_eee_record(model_id="Qwen/Qwen3-30B-A3B#no-thinking")
        identity = identity_from_eee_record(record)
        assert identity[0] == "Qwen/Qwen3-30B-A3B#no-thinking"

    def test_with_revision(self) -> None:
        """Model with @revision should be preserved."""
        record = _make_eee_record(model_id="org/model@refs_pr_6")
        identity = identity_from_eee_record(record)
        assert identity[0] == "org/model@refs_pr_6"


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


class TestNormaliseBoolValue:
    """Tests for normalise_bool_value."""

    def test_invalid_string(self) -> None:
        """Invalid string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid boolean string"):
            normalise_bool_value("yes")

    def test_invalid_type(self) -> None:
        """Invalid type should raise TypeError."""
        value = t.cast(t.Any, 1)
        with pytest.raises(TypeError, match="Unexpected type"):
            normalise_bool_value(value)

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ((False,), False),
            ((True,), True),
            ((None,), None),
            (("false", "FALSE", "False"), False),
            (("none", "NONE", "None"), None),
            (("true", "TRUE", "True"), True),
        ],
    )
    def test_normal_values(
        self, values: tuple[bool | str | None, ...], expected: bool | None
    ) -> None:
        """Boolean values and case variants normalise consistently.

        Args:
            values:
                Values supplied to the test.
            expected:
                Expected value for this test case.
        """
        for value in values:
            assert normalise_bool_value(value) is expected


class TestRaiseOnCollision:
    """Tests for raise_on_collision."""

    def test_dataset_slash_and_underscore_collision_raises(self) -> None:
        """Dataset spellings that sanitise identically must be rejected."""
        first: ResultIdentity = ("org/model", "foo/bar", False, True)
        second: ResultIdentity = ("org/model", "foo_bar", False, True)

        with pytest.raises(ValueError, match="Identity collision"):
            raise_on_collision(first, second)

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

    def test_same_identity_no_raise(self) -> None:
        """Same identity should not raise."""
        identity: ResultIdentity = ("org/model", "test_dataset", False, True)
        # Should not raise
        raise_on_collision(identity, identity)


class TestRecordFilename:
    """Tests for record_filename."""

    @pytest.mark.parametrize(
        ("dataset", "validation_split", "few_shot", "expected"),
        [
            ("test_dataset", False, True, "test_dataset__test__fewshot.json"),
            ("org/dataset", False, True, "org_dataset__test__fewshot.json"),
            ("test_dataset", None, None, "test_dataset__none__none.json"),
            ("test_dataset", True, True, "test_dataset__val__fewshot.json"),
            ("test_dataset", False, False, "test_dataset__test__zeroshot.json"),
        ],
    )
    def test_filename_labels(
        self,
        dataset: str,
        validation_split: bool | None,
        few_shot: bool | None,
        expected: str,
    ) -> None:
        """Record filenames combine sanitised names and shot/split labels.

        Args:
            dataset:
                Dataset supplied to the test.
            validation_split:
                Validation split supplied to the test.
            few_shot:
                Few shot supplied to the test.
            expected:
                Expected value for this test case.
        """
        assert (
            record_filename(
                dataset, validation_split=validation_split, few_shot=few_shot
            )
            == expected
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

    def test_with_param_variant(self) -> None:
        """Model with #param should preserve #."""
        path = record_relative_path(
            model_id="Qwen/Qwen3-30B-A3B#no-thinking",
            dataset="test_dataset",
            validation_split=False,
            few_shot=True,
        )
        expected = "Qwen_Qwen3-30B-A3B#no-thinking/test_dataset__test__fewshot.json"
        assert str(path) == expected

    def test_with_revision(self) -> None:
        """Model with @revision should preserve @."""
        path = record_relative_path(
            model_id="org/model@refs_pr_6",
            dataset="test_dataset",
            validation_split=False,
            few_shot=True,
        )
        assert str(path) == "org_model@refs_pr_6/test_dataset__test__fewshot.json"


class TestSanitiseDatasetName:
    """Tests for sanitise_dataset_name."""

    @pytest.mark.parametrize(
        ("dataset", "expected"),
        [("test_dataset", "test_dataset"), ("org/dataset", "org_dataset")],
    )
    def test_names(self, dataset: str, expected: str) -> None:
        """Dataset slashes are replaced while plain names are unchanged.

        Args:
            dataset:
                Dataset supplied to the test.
            expected:
                Expected value for this test case.
        """
        assert sanitise_dataset_name(dataset) == expected


class TestSanitiseModelDirName:
    """Tests for sanitise_model_dir_name."""

    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("org/model@refs_pr_6", "org_model@refs_pr_6"),
            ("Qwen/Qwen3-30B-A3B#no-thinking", "Qwen_Qwen3-30B-A3B#no-thinking"),
            ("model_name", "model_name"),
            ("org/model", "org_model"),
        ],
    )
    def test_names(self, model_id: str, expected: str) -> None:
        """Model slashes are replaced without altering suffix markers.

        Args:
            model_id:
                Model identifier used by the test.
            expected:
                Expected value for this test case.
        """
        assert sanitise_model_dir_name(model_id) == expected


class TestShotLabel:
    """Tests for shot_label."""

    @pytest.mark.parametrize(
        ("few_shot", "expected"),
        [(False, "zeroshot"), (None, "none"), (True, "fewshot")],
    )
    def test_labels(self, few_shot: bool | None, expected: str) -> None:
        """Shot values map to their filename labels.

        Args:
            few_shot:
                Few shot supplied to the test.
            expected:
                Expected value for this test case.
        """
        assert shot_label(few_shot) == expected


class TestSplitLabel:
    """Tests for split_label."""

    @pytest.mark.parametrize(
        ("validation_split", "expected"),
        [(False, "test"), (None, "none"), (True, "val")],
    )
    def test_labels(self, validation_split: bool | None, expected: str) -> None:
        """Split values map to their filename labels.

        Args:
            validation_split:
                Validation split supplied to the test.
            expected:
                Expected value for this test case.
        """
        assert split_label(validation_split) == expected
