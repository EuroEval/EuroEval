"""Tests for the `model_cache` module."""

import json
from pathlib import Path

from datasets import Dataset

from euroeval.data_models import GenerativeModelOutput
from euroeval.model_cache import ModelCache, load_cached_model_outputs


class TestBPCCaching:
    """Tests that BPC scores are correctly cached and loaded."""

    def test_bpc_score_saved_and_loaded(self, tmp_path: Path) -> None:
        """BPC scores are persisted to disk and reconstructed on load."""
        cache = ModelCache(
            model_cache_dir=tmp_path,
            cache_name="test-bpc-model-outputs.json",
            max_generated_tokens=64,
            progress_bar=False,
            store_metadata=False,
            indent_json_when_saving=True,
        )

        # Create model output with BPC scores
        model_output = GenerativeModelOutput(
            sequences=["test sequence 1", "test sequence 2"], bpc_scores=[0.75, 0.80]
        )

        # Add to cache
        model_inputs = {"text": ["input 1", "input 2"]}
        cache.add_to_cache(model_inputs=model_inputs, model_output=model_output)

        # Save to disk
        cache.save()

        # Create a new cache instance and load from disk
        cache2 = ModelCache(
            model_cache_dir=tmp_path,
            cache_name="test-bpc-model-outputs.json",
            max_generated_tokens=64,
            progress_bar=False,
            store_metadata=False,
            indent_json_when_saving=True,
        )
        cache2.load()

        # Verify bpc_score is in the cached items
        for key, value in cache2.cache.items():
            assert value.bpc_score is not None

    def test_bpc_score_roundtrip_via_json(self, tmp_path: Path) -> None:
        """Verify bpc_score field survives JSON serialization roundtrip."""
        cache = ModelCache(
            model_cache_dir=tmp_path,
            cache_name="test-roundtrip.json",
            max_generated_tokens=64,
            progress_bar=False,
            store_metadata=False,
            indent_json_when_saving=True,
        )

        model_output = GenerativeModelOutput(sequences=["test"], bpc_scores=[1.25])

        model_inputs = {"text": ["input"]}
        cache.add_to_cache(model_inputs=model_inputs, model_output=model_output)
        cache.save()

        # Read raw JSON to verify bpc_score is present
        cache_file = tmp_path / "test-roundtrip.json"
        raw_json = json.loads(cache_file.read_text())
        first_entry = list(raw_json.values())[0]
        assert "bpc_score" in first_entry
        assert first_entry["bpc_score"] == 1.25

    def test_load_cached_model_outputs_reconstructs_bpc_scores(
        self, tmp_path: Path
    ) -> None:
        """load_cached_model_outputs reconstructs bpc_scores from cache."""
        cache = ModelCache(
            model_cache_dir=tmp_path,
            cache_name="test-reconstruct.json",
            max_generated_tokens=64,
            progress_bar=False,
            store_metadata=False,
            indent_json_when_saving=True,
        )

        # Create and cache model outputs with BPC scores
        model_output = GenerativeModelOutput(
            sequences=["seq1", "seq2", "seq3"], bpc_scores=[0.5, 0.6, 0.7]
        )

        model_inputs = {"text": ["input1", "input2", "input3"]}
        cache.add_to_cache(model_inputs=model_inputs, model_output=model_output)
        cache.save()

        # Load the cache
        cache.load()

        # Create a cached dataset
        cached_dataset = Dataset.from_dict({"text": ["input1", "input2", "input3"]})

        # Load cached model outputs
        loaded_output = load_cached_model_outputs(
            cached_dataset=cached_dataset, cache=cache
        )

        # Verify bpc_scores are reconstructed
        assert loaded_output.bpc_scores is not None
        assert len(loaded_output.bpc_scores) == 3
        assert loaded_output.bpc_scores == [0.5, 0.6, 0.7]

    def test_cache_key_distinguishes_bpc_runs(self, tmp_path: Path) -> None:
        """Cache stores bpc_score separately from non-bpc data."""
        cache = ModelCache(
            model_cache_dir=tmp_path,
            cache_name="test-bpc-distinction.json",
            max_generated_tokens=64,
            progress_bar=False,
            store_metadata=False,
            indent_json_when_saving=True,
        )

        # Cache with BPC scores
        model_output_with_bpc = GenerativeModelOutput(
            sequences=["test"], bpc_scores=[0.9]
        )
        cache.add_to_cache(
            model_inputs={"text": ["input"]}, model_output=model_output_with_bpc
        )

        # Save and reload
        cache.save()
        cache.load()

        # Verify the cached entry has bpc_score
        cached_entry = list(cache.cache.values())[0]
        assert cached_entry.bpc_score == 0.9
