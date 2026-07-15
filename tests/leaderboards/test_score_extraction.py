"""Tests for the `leaderboards.score_extraction` module."""

from __future__ import annotations

from leaderboards.score_extraction import _is_better_metadata, extract_model_metadata


class TestIsBetterMetadata:
    """Tests for the `_is_better_metadata` helper function."""

    def test_prefers_non_none_over_none(self) -> None:
        """Non-None values are preferred over None."""
        assert _is_better_metadata(new_value=True, old_value=None, field="open") is True
        assert (
            _is_better_metadata(new_value=None, old_value=True, field="open") is False
        )

    def test_prefers_non_none_over_none_commercial(self) -> None:
        """Non-None values are preferred over None for commercial field."""
        assert (
            _is_better_metadata(new_value=True, old_value=None, field="commercial")
            is True
        )

    def test_prefers_true_over_false_for_booleans(self) -> None:
        """For boolean fields, True is preferred over False."""
        assert (
            _is_better_metadata(new_value=True, old_value=False, field="commercial")
            is True
        )
        assert (
            _is_better_metadata(new_value=False, old_value=True, field="commercial")
            is False
        )

    def test_prefers_true_over_false_for_merge(self) -> None:
        """For merge field, True is preferred over False."""
        assert (
            _is_better_metadata(new_value=True, old_value=False, field="merge") is True
        )
        assert (
            _is_better_metadata(new_value=False, old_value=True, field="merge") is False
        )

    def test_prefers_true_over_false_for_open(self) -> None:
        """For open field, True is preferred over False."""
        assert (
            _is_better_metadata(new_value=True, old_value=False, field="open") is True
        )
        assert (
            _is_better_metadata(new_value=False, old_value=True, field="open") is False
        )

    def test_prefers_true_over_false_for_trained_from_scratch(self) -> None:
        """For trained_from_scratch field, True is preferred over False."""
        assert (
            _is_better_metadata(
                new_value=True, old_value=False, field="trained_from_scratch"
            )
            is True
        )

    def test_prefers_non_nan_over_nan_for_floats(self) -> None:
        """For float fields, non-NaN is preferred over NaN."""
        assert (
            _is_better_metadata(
                new_value=1000000.0, old_value=float("nan"), field="parameters"
            )
            is True
        )
        assert (
            _is_better_metadata(
                new_value=float("nan"), old_value=1000000.0, field="parameters"
            )
            is False
        )

    def test_prefers_non_empty_generative_type(self) -> None:
        """For generative_type, non-empty is preferred over empty."""
        assert (
            _is_better_metadata(
                new_value="instruction_tuned", old_value=None, field="generative_type"
            )
            is True
        )
        assert (
            _is_better_metadata(
                new_value=None, old_value="instruction_tuned", field="generative_type"
            )
            is False
        )

    def test_prefers_non_empty_model_url(self) -> None:
        """For model_url, non-empty is preferred over empty."""
        assert (
            _is_better_metadata(
                new_value="https://example.com", old_value=None, field="model_url"
            )
            is True
        )
        assert (
            _is_better_metadata(
                new_value=None, old_value="https://example.com", field="model_url"
            )
            is False
        )


class TestExtractModelMetadata:
    """Tests for the `extract_model_metadata` function."""

    def _record(
        self,
        name: str = "org/model",
        dataset: str = "angry-tweets",
        version: str = "17.6.0",
        additional_details: dict | None = None,
    ) -> dict:
        """Build a minimal EEE-style record.

        Args:
            name:
                The model name.
            dataset:
                The dataset name.
            version:
                The EuroEval version.
            additional_details:
                Additional details to include.

        Returns:
            A minimal EEE-style record.
        """
        additional: dict = {"dataset": dataset}
        if additional_details:
            additional.update(additional_details)
        return {
            "model_info": {"name": name, "additional_details": additional},
            "eval_library": {"version": version},
        }

    def test_metadata_not_overwritten_by_stale_record(self) -> None:
        """Enriched metadata should not be overwritten by a stale record.

        Regression test for issue where stale records from unknown.jsonl or
        misfiled results would override enriched metadata during extraction.
        """
        # Enriched record with full metadata
        enriched = self._record(
            name="Qwen/Qwen3.6-27B-FP8 (val)",
            additional_details={
                "commercially_licensed": True,
                "open": True,
                "merge": "false",
                "trained_from_scratch": True,
                "generative_type": "instruction_tuned",
                "model_url": "https://huggingface.co/Qwen/Qwen3.6-27B-FP8",
            },
        )

        # Stale record with missing/default metadata
        stale = self._record(
            name="Qwen/Qwen3.6-27B-FP8 (val)",
            additional_details={
                # No metadata fields - all defaults
                "dataset": "angry-tweets"
            },
        )

        # Process enriched first, then stale
        metadata = extract_model_metadata(results=[enriched, stale])

        # Model ID will be normalized by extract_model_ids_from_record
        model_key = "Qwen/Qwen3.6-27B-FP8 (val)"

        # Metadata should preserve enriched values
        assert metadata[model_key].get("commercial") is True
        assert metadata[model_key].get("open") is True
        assert metadata[model_key].get("merge") is False
        assert metadata[model_key].get("trained_from_scratch") is True
        assert metadata[model_key].get("generative_type") == "instruction_tuned"
        # model_url is preserved (exact format depends on generate_model_url)
        assert metadata[model_key].get("model_url") is not None

    def test_metadata_upgraded_when_better_data_arrives(self) -> None:
        """Metadata should be upgraded when a better record arrives later."""
        # Poor record first - needs model_url to avoid API call
        poor = self._record(
            name="org/model",
            additional_details={"model_url": "https://huggingface.co/org/model"},
        )

        # Rich record second
        rich = self._record(
            name="org/model",
            additional_details={
                "commercially_licensed": True,
                "open": True,
                "model_url": "https://huggingface.co/org/model",
            },
        )

        metadata = extract_model_metadata(results=[poor, rich])
        model_key = "org/model"

        assert metadata[model_key].get("commercial") is True
        assert metadata[model_key].get("open") is True

    def test_stale_unknown_jsonl_style_record_does_not_override(self) -> None:
        """A stale unknown.jsonl-style record should not override enriched metadata.

        This simulates the case where misfiled records for other models end up
        in the same file, or records from unknown.jsonl have null metadata.
        """
        enriched = self._record(
            name="Qwen/Qwen3.6-27B-FP8 (val)",
            additional_details={
                "commercially_licensed": True,
                "open": True,
                "merge": "false",
                "trained_from_scratch": True,
                "model_url": "https://huggingface.co/Qwen/Qwen3.6-27B-FP8",
            },
        )

        # Simulate a stale record like from unknown.jsonl with null/false defaults
        stale = self._record(
            name="Qwen/Qwen3.6-27B-FP8 (val)",
            additional_details={
                "commercially_licensed": False,  # Default
                "open": None,  # Missing
                "merge": "false",  # Default
                "trained_from_scratch": None,  # Missing
                "model_url": "https://huggingface.co/Qwen/Qwen3.6-27B-FP8",
            },
        )

        metadata = extract_model_metadata(results=[enriched, stale])
        model_key = "Qwen/Qwen3.6-27B-FP8 (val)"

        # Should preserve enriched values, not be downgraded
        assert metadata[model_key].get("commercial") is True
        assert metadata[model_key].get("open") is True
        assert metadata[model_key].get("trained_from_scratch") is True
