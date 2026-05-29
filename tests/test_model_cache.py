"""Tests for the `model_cache` module."""

from pathlib import Path

from datasets import Dataset

from euroeval.data_models import (
    GenerativeModelOutput,
    HashableDict,
    SingleGenerativeModelOutput,
)
from euroeval.model_cache import ModelCache, load_cached_model_outputs


def _make_cache(tmp_path: Path, name: str = "round-trip-test.json") -> ModelCache:
    """Build a minimal ModelCache backed by `tmp_path`.

    Args:
        tmp_path: Directory to store the cache file in.
        name: Cache filename (relative to `tmp_path`).

    Returns:
        A `ModelCache` instance configured for tests.
    """
    return ModelCache(
        model_cache_dir=tmp_path,
        cache_name=name,
        max_generated_tokens=1,
        progress_bar=False,
        store_metadata=False,
        indent_json_when_saving=False,
    )


class TestCFScoresRoundTrip:
    """Tests that `cf_scores` survive the JSON cache save/load cycle.

    The Cloze Formulation evaluation path stores per-choice scores on the
    cached output; if these are dropped on save or load, a CF run cannot
    resume from cache and would silently re-score every example.
    """

    def test_save_and_load_preserves_cf_scores(self, tmp_path: Path) -> None:
        """A `cf_scores` field set on a single output survives save → load."""
        cache = _make_cache(tmp_path)
        cache["sample-prompt"] = SingleGenerativeModelOutput(
            sequence="",
            predicted_label=None,
            scores=None,
            cf_scores=[-1.5, -2.0, -0.5, -3.0],
            metadata=HashableDict({}),
        )
        cache.save()

        fresh = _make_cache(tmp_path)
        fresh.load()
        assert fresh["sample-prompt"].cf_scores == [-1.5, -2.0, -0.5, -3.0]

    def test_load_with_no_cf_scores_field_yields_none(self, tmp_path: Path) -> None:
        """Loading a legacy cache file without a `cf_scores` field returns None.

        Older caches written before CF support was added do not include the
        `cf_scores` key. The loader must default it to None rather than raise.
        """
        cache_path = tmp_path / "legacy.json"
        cache_path.write_text(
            '{"some-hash": {"sequence": "hi", "predicted_label": "a", "scores": null}}'
        )
        cache = _make_cache(tmp_path, name="legacy.json")
        cache.load()
        # The single entry round-trips back without a cf_scores field.
        loaded = next(iter(cache.cache.values()))
        assert loaded.cf_scores is None
        assert loaded.sequence == "hi"

    def test_add_to_cache_carries_cf_scores_per_sample(self, tmp_path: Path) -> None:
        """`add_to_cache` slices `cf_scores` per sample and persists each row."""
        cache = _make_cache(tmp_path)
        model_inputs = {"text": ["prompt-1", "prompt-2"]}
        model_output = GenerativeModelOutput(
            sequences=["", ""], cf_scores=[[-1.0, -2.0], [-3.0, -4.0]]
        )
        cache.add_to_cache(model_inputs=model_inputs, model_output=model_output)
        cache.save()

        fresh = _make_cache(tmp_path)
        fresh.load()
        assert fresh["prompt-1"].cf_scores == [-1.0, -2.0]
        assert fresh["prompt-2"].cf_scores == [-3.0, -4.0]


class TestLoadCachedModelOutputsWithCFScores:
    """Tests for `load_cached_model_outputs` rebuilding `cf_scores` on the batch."""

    def test_cf_scores_are_aggregated_into_batch_output(self, tmp_path: Path) -> None:
        """Per-sample cached `cf_scores` reappear as a 2D batch matrix."""
        cache = _make_cache(tmp_path)
        cache["prompt-1"] = SingleGenerativeModelOutput(
            sequence="", cf_scores=[-1.0, -2.0]
        )
        cache["prompt-2"] = SingleGenerativeModelOutput(
            sequence="", cf_scores=[-0.5, -3.0]
        )
        dataset = Dataset.from_dict({"text": ["prompt-1", "prompt-2"]})
        output = load_cached_model_outputs(cached_dataset=dataset, cache=cache)
        assert output.cf_scores == [[-1.0, -2.0], [-0.5, -3.0]]
        # MCF-only fields stay None when no scores were cached.
        assert output.scores is None

    def test_cf_scores_none_when_not_cached(self, tmp_path: Path) -> None:
        """If no cached entry has `cf_scores`, the batch output has `cf_scores=None`."""
        cache = _make_cache(tmp_path)
        cache["prompt-1"] = SingleGenerativeModelOutput(sequence="hello")
        dataset = Dataset.from_dict({"text": ["prompt-1"]})
        output = load_cached_model_outputs(cached_dataset=dataset, cache=cache)
        assert output.cf_scores is None
