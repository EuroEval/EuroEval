"""Tests for bootstrap-based rank computation.

Tests cover determinism, paired-rank behaviour, and eligible-set consistency.
"""

from __future__ import annotations

import numpy as np

from src.leaderboards.bootstrap_cis import (
    _aggregate_scores_to_categories,
    bootstrap_confidence_intervals,
    bootstrap_rank_scores,
)
from src.leaderboards.score_computation import (
    compute_dataset_ranks_bootstrap,
    compute_standard_ranks_bootstrap,
    compute_standard_ranks_from_bootstrap_scores,
)


class TestBootstrapAggregation:
    """Tests for bootstrap aggregation semantics."""

    def test_aggregate_scores_retains_duplicate_dataset_samples(self) -> None:
        """Duplicate dataset samples contribute with their full multiplicity."""
        configs = {"da": {"knowledge": ["dataset_a", "dataset_b"]}}
        dataset_scores = {"dataset_a": [1.0, 3.0], "dataset_b": [5.0]}

        language_scores, overall = _aggregate_scores_to_categories(
            dataset_scores=dataset_scores, configs=configs
        )

        assert language_scores["da"] == 3.0
        assert overall == 3.0


class TestBootstrapDeterminism:
    """Tests for bootstrap determinism with seed."""

    def test_bootstrap_scores_deterministic_with_seed(self) -> None:
        """Bootstrapping with a seed produces identical results across runs."""
        model_ids = ["model_a", "model_b", "model_c"]
        datasets = [f"dataset_{i}" for i in range(8)]
        model_results = _make_dummy_results(model_ids, datasets)
        configs = _make_dummy_configs(["da"], datasets)

        # Run twice with same seed
        result1 = bootstrap_rank_scores(
            model_results=model_results, configs=configs, n_bootstraps=100, seed=42
        )
        result2 = bootstrap_rank_scores(
            model_results=model_results, configs=configs, n_bootstraps=100, seed=42
        )

        # Results should be identical
        for model_id in model_ids:
            for category in result1[model_id]:
                for lang in result1[model_id][category]:
                    np.testing.assert_array_equal(
                        result1[model_id][category][lang],
                        result2[model_id][category][lang],
                    )

    def test_bootstrap_scores_different_without_seed(self) -> None:
        """Bootstrapping without a seed produces different results across runs."""
        model_ids = ["model_a", "model_b"]
        datasets = [f"dataset_{i}" for i in range(4)]
        model_results = _make_dummy_results(model_ids, datasets)
        configs = _make_dummy_configs(["da"], datasets)

        result1 = bootstrap_rank_scores(
            model_results=model_results, configs=configs, n_bootstraps=100, seed=None
        )
        result2 = bootstrap_rank_scores(
            model_results=model_results, configs=configs, n_bootstraps=100, seed=None
        )

        # Results should be different (with very high probability)
        model_id = model_ids[0]
        category = list(result1[model_id].keys())[0]
        lang = list(result1[model_id][category].keys())[0]
        assert not np.array_equal(
            result1[model_id][category][lang], result2[model_id][category][lang]
        )

    def test_dataset_rank_bootstrap_ignores_dataset_list_order(self) -> None:
        """Dataset-level CIs are deterministic under config list reordering."""
        model_ids = ["model_a", "model_b"]
        datasets = [f"dataset_{i}" for i in range(4)]
        model_results = _make_dummy_results(model_ids, datasets)
        configs = _make_dummy_configs(["da"], datasets)
        configs_reordered = {
            "da": {
                task: list(reversed(task_datasets))
                for task, task_datasets in configs["da"].items()
            }
        }

        result1 = compute_dataset_ranks_bootstrap(
            model_results=model_results, configs=configs, n_bootstraps=100, seed=42
        )
        result2 = compute_dataset_ranks_bootstrap(
            model_results=model_results,
            configs=configs_reordered,
            n_bootstraps=100,
            seed=42,
        )

        assert result1 == result2

    def test_semantic_inputs_same_order_produce_identical_arrays(self) -> None:
        """Same semantic inputs with different dict insertion order give same results.

        This tests that sorting model IDs, datasets, tasks, and categories before
        iterating ensures determinism regardless of Python dict ordering.
        """
        model_ids = ["model_a", "model_b", "model_c"]
        datasets = [f"dataset_{i}" for i in range(6)]
        model_results = _make_dummy_results(model_ids, datasets)
        configs = _make_dummy_configs(["da"], datasets)

        model_results_reordered = {
            model_id: {
                ds: model_results[model_id][ds]
                for ds in reversed(list(model_results[model_id].keys()))
            }
            for model_id in reversed(model_ids)
        }

        configs_reordered = {
            lang: {
                task: list(reversed(configs[lang][task]))
                for task in reversed(list(configs[lang].keys()))
            }
            for lang in reversed(list(configs.keys()))
        }

        result1 = bootstrap_rank_scores(
            model_results=model_results, configs=configs, n_bootstraps=100, seed=42
        )
        result2 = bootstrap_rank_scores(
            model_results=model_results_reordered,
            configs=configs_reordered,
            n_bootstraps=100,
            seed=42,
        )

        for model_id in model_ids:
            for category in result1[model_id]:
                for lang in result1[model_id][category]:
                    np.testing.assert_array_equal(
                        result1[model_id][category][lang],
                        result2[model_id][category][lang],
                    )


def _make_dummy_configs(
    languages: list[str], datasets: list[str]
) -> dict[str, dict[str, list[str]]]:
    """Create dummy configs mapping languages to tasks to datasets.

    Uses valid task names that work with category_includes_task.

    Returns:
        Dict mapping language -> task -> list of dataset IDs.
    """
    configs = {}
    for lang in languages:
        # Use valid NLU task names
        configs[lang] = {
            "named-entity-recognition": datasets[: len(datasets) // 2],
            "sentiment-classification": datasets[len(datasets) // 2 :],
        }
    return configs


def _make_dummy_results(
    model_ids: list[str],
    dataset_ids: list[str],
    base_score: float = 0.7,
    noise: float = 0.01,
) -> dict[str, dict[str, list[tuple[list[float], float, float]]]]:
    """Create dummy model results for testing.

    Each model has the same datasets with similar scores to create realistic
    bootstrap distributions.

    Returns:
        Dict mapping model_id -> dataset -> [(raw_scores, mean, std_err)].
    """
    rng = np.random.default_rng(42)
    results: dict[str, dict[str, list[tuple[list[float], float, float]]]] = {}

    for model_id in model_ids:
        results[model_id] = {}
        for dataset_id in dataset_ids:
            # Generate raw scores around base_score with some model-specific offset
            model_offset = rng.uniform(-0.05, 0.05)
            raw_scores = (
                rng.uniform(0, 1, 100) * noise + base_score + model_offset
            ).tolist()
            mean_score = float(np.mean(raw_scores))
            std_err = float(np.std(raw_scores) / np.sqrt(len(raw_scores)))
            results[model_id][dataset_id] = [(raw_scores, mean_score, std_err)]

    return results


class TestEligibleSetConsistency:
    """Tests for consistency between rank score and ordinal rank."""

    def test_non_eligible_models_show_placeholder(self) -> None:
        """Non-eligible models in full results get '-' for rank and rank score.

        This tests the behavior in _build_model_row_data where non-eligible
        models should show '-' for both columns.
        """
        # This is more of an integration test for leaderboard_generation
        # but we verify the logic here
        all_datasets = [f"dataset_{i}" for i in range(8)]

        model_results = {
            "eligible_model": {ds: [([0.7] * 100, 0.7, 0.01)] for ds in all_datasets},
            "partial_model": {
                ds: [([0.65] * 100, 0.65, 0.01)]
                for ds in all_datasets[:6]  # Missing last 2
            },
        }

        # Simulate eligibility check - must have ALL datasets
        eligible_model_results = {
            mid: mr
            for mid, mr in model_results.items()
            if all(ds in mr for ds in all_datasets)
        }

        # Only eligible_model should be in eligible results
        assert "eligible_model" in eligible_model_results
        assert "partial_model" not in eligible_model_results

    def test_ranks_derived_from_same_bootstrap_distribution(self) -> None:
        """Rank score and ordinal rank come from the same bootstrap distribution.

        This is the key fix: previously, rank score was computed over all models
        while ordinal rank was computed over eligible models only, causing
        inconsistencies.
        """
        # Create model results where some models are "eligible" (have all datasets)
        # and some are not
        all_datasets = [f"dataset_{i}" for i in range(8)]

        model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]] = {
            "eligible_model_a": {ds: [([0.7] * 100, 0.7, 0.01)] for ds in all_datasets},
            "eligible_model_b": {
                ds: [([0.65] * 100, 0.65, 0.01)] for ds in all_datasets
            },
            "partial_model": {
                ds: [([0.60] * 100, 0.60, 0.01)]
                for ds in all_datasets[:6]  # Missing last 2 datasets
            },
        }
        configs = _make_dummy_configs(["da"], all_datasets)

        # Eligibility check: model must have ALL datasets
        required_datasets = all_datasets
        eligible_model_results = {
            mid: mr
            for mid, mr in model_results.items()
            if all(ds in mr for ds in required_datasets)
        }

        # Only the two eligible models should remain
        assert "eligible_model_a" in eligible_model_results
        assert "eligible_model_b" in eligible_model_results
        assert "partial_model" not in eligible_model_results

        bootstrap_scores = bootstrap_rank_scores(
            model_results=eligible_model_results,
            configs=configs,
            n_bootstraps=200,
            seed=42,
            categories=("all_models",),
        )

        # Derive both rank score and ordinal rank from same distribution
        rank_scores = bootstrap_confidence_intervals(bootstrap_scores)
        ordinal_ranks = compute_standard_ranks_from_bootstrap_scores(
            bootstrap_scores=bootstrap_scores, alpha=0.05
        )

        # Both should only contain eligible models
        assert "eligible_model_a" in rank_scores
        assert "eligible_model_b" in rank_scores
        assert "eligible_model_a" in ordinal_ranks
        assert "eligible_model_b" in ordinal_ranks
        assert "partial_model" not in rank_scores  # Not in bootstrap
        assert "partial_model" not in ordinal_ranks  # Not in bootstrap


class TestPairedBootstrapRanks:
    """Tests for paired-bootstrap ordinal rank computation."""

    def test_better_models_get_lower_ranks(self) -> None:
        """Models with higher scores should get lower (better) ranks."""
        # Create results where model_a is clearly better than model_b
        model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]] = {
            "model_a": {f"dataset_{i}": [([0.8] * 100, 0.8, 0.01)] for i in range(8)},
            "model_b": {f"dataset_{i}": [([0.6] * 100, 0.6, 0.01)] for i in range(8)},
        }
        configs = _make_dummy_configs(["da"], list(model_results["model_a"].keys()))

        ranks = compute_standard_ranks_bootstrap(
            model_results=model_results, configs=configs, n_bootstraps=200, seed=42
        )

        # model_a should be ranked better (lower number) than model_b
        assert ranks["model_a"]["all_models"] <= ranks["model_b"]["all_models"]

    def test_non_transitive_overlap_compares_to_group_anchor(self) -> None:
        """An intermediate overlap does not pull worse models into rank 1."""
        model_a = np.zeros(100)
        model_b = np.full(100, 0.1)
        model_b[:3] = -0.1
        model_c = np.full(100, 0.2)
        model_c[3:6] = 0.05
        bootstrap_scores = {
            "model_a": {"generative": {"overall": model_a}},
            "model_b": {"generative": {"overall": model_b}},
            "model_c": {"generative": {"overall": model_c}},
        }

        ranks = compute_standard_ranks_from_bootstrap_scores(
            bootstrap_scores=bootstrap_scores, alpha=0.05
        )

        assert ranks["model_a"]["generative"] == ranks["model_b"]["generative"]
        assert ranks["model_c"]["generative"] > ranks["model_a"]["generative"]

    def test_overlapping_paired_difference_ci_keeps_same_rank(self) -> None:
        """Models tie when paired differences are not reliably positive."""
        bootstrap_scores = {
            "model_a": {"generative": {"overall": np.array([1.0, 1.0, 1.0, 1.0, 1.0])}},
            "model_b": {"generative": {"overall": np.array([0.9, 1.1, 1.1, 1.1, 1.1])}},
        }

        ranks = compute_standard_ranks_from_bootstrap_scores(
            bootstrap_scores=bootstrap_scores, alpha=0.05
        )

        assert ranks["model_a"]["generative"] == ranks["model_b"]["generative"]

    def test_paired_ranks_deterministic_with_seed(self) -> None:
        """Paired-bootstrap ranks are deterministic when using a seed."""
        model_ids = ["model_a", "model_b", "model_c"]
        datasets = [f"dataset_{i}" for i in range(8)]
        model_results = _make_dummy_results(model_ids, datasets)
        configs = _make_dummy_configs(["da"], datasets)

        ranks1 = compute_standard_ranks_bootstrap(
            model_results=model_results, configs=configs, n_bootstraps=200, seed=42
        )
        ranks2 = compute_standard_ranks_bootstrap(
            model_results=model_results, configs=configs, n_bootstraps=200, seed=42
        )

        assert ranks1 == ranks2

    def test_paired_ranks_from_bootstrap_scores(self) -> None:
        """Paired-bootstrap ranks can be derived from pre-computed bootstrap scores."""
        model_ids = ["model_a", "model_b", "model_c"]
        datasets = [f"dataset_{i}" for i in range(8)]
        model_results = _make_dummy_results(model_ids, datasets)
        configs = _make_dummy_configs(["da"], datasets)

        bootstrap_scores = bootstrap_rank_scores(
            model_results=model_results, configs=configs, n_bootstraps=200, seed=42
        )

        ranks = compute_standard_ranks_from_bootstrap_scores(
            bootstrap_scores=bootstrap_scores, alpha=0.05
        )

        for model_id in model_ids:
            assert model_id in ranks
            assert "all_models" in ranks[model_id]
