"""Tests for the `leaderboards.score_extraction` module."""

from __future__ import annotations

import math

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

    def test_prefers_present_over_absent_for_booleans(self) -> None:
        """For boolean fields, present (non-None) is preferred over absent (None).

        Explicit False is legitimate metadata and should be preserved.
        """
        # Present value preferred over absent
        assert (
            _is_better_metadata(new_value=True, old_value=None, field="commercial")
            is True
        )
        assert (
            _is_better_metadata(new_value=False, old_value=None, field="commercial")
            is True
        )
        # Absent value not preferred over present
        assert (
            _is_better_metadata(new_value=None, old_value=True, field="commercial")
            is False
        )
        assert (
            _is_better_metadata(new_value=None, old_value=False, field="commercial")
            is False
        )
        # Equal presence: neither is "better" (don't overwrite existing)
        assert (
            _is_better_metadata(new_value=False, old_value=False, field="commercial")
            is False
        )

    def test_prefers_present_over_absent_for_merge(self) -> None:
        """For merge field, present (non-None) is preferred over absent (None).

        Explicit False is legitimate metadata and should be preserved.
        """
        assert (
            _is_better_metadata(new_value=False, old_value=None, field="merge") is True
        )
        assert (
            _is_better_metadata(new_value=None, old_value=False, field="merge") is False
        )

    def test_prefers_present_over_absent_for_open(self) -> None:
        """For open field, present (non-None) is preferred over absent (None).

        Explicit False is legitimate metadata and should be preserved.
        """
        assert (
            _is_better_metadata(new_value=False, old_value=None, field="open") is True
        )
        assert (
            _is_better_metadata(new_value=None, old_value=False, field="open") is False
        )

    def test_prefers_present_over_absent_for_trained_from_scratch(self) -> None:
        """For trained_from_scratch field, present (non-None) is preferred over absent.

        Explicit False is legitimate metadata and should be preserved.
        """
        assert (
            _is_better_metadata(
                new_value=False, old_value=None, field="trained_from_scratch"
            )
            is True
        )
        assert (
            _is_better_metadata(
                new_value=None, old_value=False, field="trained_from_scratch"
            )
            is False
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

    def test_false_metadata_preserved_against_stale_none(self) -> None:
        """Legitimate False metadata should be preserved against stale None values.

        Regression test: models with merge=False, open=False, etc. should not
        have their explicit False values overwritten by later stale records
        with None/missing metadata.
        """
        # Record with explicit False values (legitimate metadata)
        record_with_false = self._record(
            name="org/model-false",
            additional_details={
                "commercially_licensed": False,
                "open": False,
                "merge": "false",
                "trained_from_scratch": False,
                "model_url": "https://huggingface.co/org/model-false",
            },
        )

        # Stale record with None/missing metadata
        stale = self._record(
            name="org/model-false",
            additional_details={
                "commercially_licensed": None,
                "open": None,
                "merge": None,
                "trained_from_scratch": None,
                "model_url": "https://huggingface.co/org/model-false",
            },
        )

        metadata = extract_model_metadata(results=[record_with_false, stale])
        model_key = "org/model-false"

        # Explicit False values should be preserved
        assert metadata[model_key].get("commercial") is False
        assert metadata[model_key].get("open") is False
        assert metadata[model_key].get("merge") is False
        assert metadata[model_key].get("trained_from_scratch") is False

    def test_false_not_overwritten_by_later_true(self) -> None:
        """Regression test: explicit False should not be overwritten by later True.

        Before the fix, truthiness checks like `if old_value := existing.get(...)`
        treated False as absent, allowing later True values to overwrite explicit False.
        """
        # First record with explicit False values
        record_false = self._record(
            name="org/model-bool-test",
            additional_details={
                "commercially_licensed": False,
                "open": False,
                "merge": "false",
                "trained_from_scratch": False,
                "model_url": "https://huggingface.co/org/model-bool-test",
            },
        )

        # Second record with True values (later in processing order)
        record_true = self._record(
            name="org/model-bool-test",
            additional_details={
                "commercially_licensed": True,
                "open": True,
                "merge": "true",
                "trained_from_scratch": True,
                "model_url": "https://huggingface.co/org/model-bool-test",
            },
        )

        metadata = extract_model_metadata(results=[record_false, record_true])
        model_key = "org/model-bool-test"

        # Explicit False values should be preserved (not overwritten by later True)
        assert metadata[model_key].get("commercial") is False
        assert metadata[model_key].get("open") is False
        assert metadata[model_key].get("merge") is False
        assert metadata[model_key].get("trained_from_scratch") is False

    def test_fallback_model_url_stores_when_missing(self) -> None:
        """Regression test: generated fallback URL should be stored when missing.

        Before the fix, model_url_present remained False after generating fallback,
        so the fallback URL was never stored.
        """
        # Record without explicit model_url (fallback will be generated)
        # Use ollama/ prefix so generate_ollama_url can generate URL without API calls
        record_no_url = self._record(
            name="ollama/test-model", additional_details={"commercially_licensed": True}
        )

        metadata = extract_model_metadata(results=[record_no_url])
        model_key = "ollama/test-model"

        # Fallback URL should be stored
        assert metadata[model_key].get("model_url") is not None
        assert "ollama.com" in metadata[model_key]["model_url"]

    def test_fallback_model_url_does_not_overwrite_explicit(self) -> None:
        """Fallback URL should fill missing but not overwrite explicit URLs."""
        # First record with explicit URL. Use ollama/ prefix to avoid API calls
        record_with_url = self._record(
            name="ollama/model-explicit-url",
            additional_details={"model_url": "https://explicit.example.com/model"},
        )

        # Second record without URL (would generate fallback if first wasn't present)
        record_no_url = self._record(
            name="ollama/model-explicit-url", additional_details={}
        )

        metadata = extract_model_metadata(results=[record_with_url, record_no_url])
        model_key = "ollama/model-explicit-url"

        # Explicit URL should be preserved (not overwritten by generated fallback)
        assert (
            metadata[model_key].get("model_url") == "https://explicit.example.com/model"
        )

    def test_explicit_model_url_replaces_fallback(self) -> None:
        """Regression test: explicit URL should replace generated fallback.

        Before the fix, if a record without model_url was processed before one
        with an explicit URL for the same model, the generated fallback blocked
        the explicit URL from being stored.
        """
        # First record without URL (fallback will be generated and stored)
        # Use ollama/ prefix so generate_ollama_url can generate URL without API calls
        record_no_url = self._record(
            name="ollama/model-fallback-first",
            additional_details={"commercially_licensed": True},
        )

        # Second record with explicit URL
        record_with_url = self._record(
            name="ollama/model-fallback-first",
            additional_details={"model_url": "https://explicit.example.com/model"},
        )

        metadata = extract_model_metadata(results=[record_no_url, record_with_url])
        model_key = "ollama/model-fallback-first"

        # Explicit URL should replace the generated fallback
        assert (
            metadata[model_key].get("model_url") == "https://explicit.example.com/model"
        )

    def test_generated_fallback_preserved_when_no_explicit_url(self) -> None:
        """Regression test: generated fallback kept if no explicit URL arrives.

        When multiple records without explicit URLs are processed for the same model,
        the first generated fallback should be preserved.
        """
        # Two records without explicit URLs for the same model
        record_no_url_1 = self._record(
            name="ollama/model-multiple-fallbacks",
            additional_details={"commercially_licensed": True},
        )

        record_no_url_2 = self._record(
            name="ollama/model-multiple-fallbacks", additional_details={"open": True}
        )

        metadata = extract_model_metadata(results=[record_no_url_1, record_no_url_2])
        model_key = "ollama/model-multiple-fallbacks"

        # URL should be present (first generated fallback preserved)
        assert metadata[model_key].get("model_url") is not None
        assert "ollama.com" in metadata[model_key]["model_url"]

    def test_all_standard_metadata_keys_present_for_all_models(self) -> None:
        """Regression test: all models have all standard metadata keys.

        Before the fix, extract_model_metadata() only set metadata fields when
        explicitly present in records, causing KeyError in generate_dataframe()
        which expects all standard columns to exist for every model.

        This test verifies that models with missing metadata fields still get
        all standard keys with appropriate defaults.
        """
        # Record with full metadata (use ollama/ prefix to avoid API calls)
        record_full = self._record(
            name="ollama/model-full",
            additional_details={
                "generative_type": "instruction_tuned",
                "commercially_licensed": True,
                "open": True,
                "merge": "false",
                "trained_from_scratch": True,
                "num_model_parameters": "7000000000",
                "vocabulary_size": "32000",
                "max_sequence_length": "4096",
                "model_url": "https://huggingface.co/ollama/model-full",
            },
        )

        # Record lacking generative_type and other metadata
        record_missing = self._record(
            name="ollama/model-missing-generative-type",
            additional_details={
                # No generative_type field
                "commercially_licensed": False  # Explicit False should be preserved
            },
        )

        metadata = extract_model_metadata(results=[record_full, record_missing])

        # Both models should have all standard keys
        standard_keys = [
            "parameters",
            "vocabulary_size",
            "context",
            "generative_type",
            "commercial",
            "merge",
            "open",
            "trained_from_scratch",
            "model_url",
        ]

        model_full_key = "ollama/model-full"
        model_missing_key = "ollama/model-missing-generative-type"

        # Model with full metadata should have all keys
        for key in standard_keys:
            assert key in metadata[model_full_key], (
                f"Model {model_full_key!r} missing key {key!r}"
            )

        # Model with missing metadata should also have all keys (with defaults)
        for key in standard_keys:
            assert key in metadata[model_missing_key], (
                f"Model {model_missing_key!r} missing key {key!r}"
            )

        # Verify specific default values for the model with missing metadata
        assert metadata[model_missing_key]["generative_type"] is None
        assert metadata[model_missing_key]["commercial"] is False  # Explicit False
        assert metadata[model_missing_key]["merge"] is False  # Default
        assert metadata[model_missing_key]["open"] is None  # Default
        assert metadata[model_missing_key]["trained_from_scratch"] is None  # Default
        assert math.isnan(metadata[model_missing_key]["parameters"])
        assert math.isnan(metadata[model_missing_key]["vocabulary_size"])
        assert math.isnan(metadata[model_missing_key]["context"])
        # model_url should have generated fallback for ollama model
        assert metadata[model_missing_key]["model_url"] is not None
        assert "ollama.com" in metadata[model_missing_key]["model_url"]

        # Verify full metadata model has correct values
        assert metadata[model_full_key]["generative_type"] == "instruction_tuned"
        assert metadata[model_full_key]["commercial"] is True
        assert metadata[model_full_key]["merge"] is False
        assert metadata[model_full_key]["open"] is True
        assert metadata[model_full_key]["trained_from_scratch"] is True
        assert metadata[model_full_key]["parameters"] == 7_000_000_000.0
        assert metadata[model_full_key]["vocabulary_size"] == 32_000.0
        assert metadata[model_full_key]["context"] == 4_096.0
        assert (
            metadata[model_full_key]["model_url"]
            == "https://huggingface.co/ollama/model-full"
        )
