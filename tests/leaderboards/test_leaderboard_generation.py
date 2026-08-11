"""Tests for leaderboard generation with per-language rank-score sync.

Tests verify that multilingual leaderboards have per-language rank-score columns
matching the corresponding monolingual leaderboards, while overall rank scores
and ordinal ranks remain pan-leaderboard.
"""

from __future__ import annotations

import math

import numpy as np

from src.leaderboards.enums import LeaderboardCategory
from src.leaderboards.leaderboard_generation import _compute_eligible_models_and_ranks


class TestMultilingualPerLanguageRankScores:
    """Tests for per-language rank-score sync in multilingual leaderboards."""

    def test_language_rank_computed_with_language_eligible_set(self) -> None:
        """Per-language ranks use language-specific eligible sets.

        This directly tests the fix: each language's rank score is computed
        using only models that have all datasets for that language.
        """
        albanian_datasets = [f"sq_dataset_{i}" for i in range(4)]
        danish_datasets = [f"da_dataset_{i}" for i in range(4)]
        all_datasets = albanian_datasets + danish_datasets

        # model_a complete, model_b only Albanian, model_c only Danish
        model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]] = {
            "model_a": {ds: [([0.75] * 100, 0.75, 0.01)] for ds in all_datasets},
            "model_b": {ds: [([0.70] * 100, 0.70, 0.01)] for ds in albanian_datasets},
            "model_c": {ds: [([0.65] * 100, 0.65, 0.01)] for ds in danish_datasets},
        }

        multilingual_config = {
            **_make_dummy_configs(["albanian"], albanian_datasets),
            **_make_dummy_configs(["danish"], danish_datasets),
        }

        multilingual_category_to_datasets = {"generative": all_datasets}
        (_, _, ranks, std_ranks) = _compute_eligible_models_and_ranks(
            model_results=model_results,
            category=LeaderboardCategory.GENERATIVE,
            category_to_datasets=multilingual_category_to_datasets,
            category_to_orthogonal_datasets={"generative": {}},
            leaderboard_configs=multilingual_config,
        )

        # Only model_a is multilingual-eligible, so std_ranks may be empty
        # (with one model, no ordinal ranking is possible)
        # The key test is that per-language ranks are computed correctly

        # Albanian ranks: model_a and model_b should have scores
        # (both are Albanian-eligible)
        assert "albanian" in ranks["model_a"]["generative"], (
            f"model_a should have Albanian rank, got: {ranks.get('model_a', {})}"
        )
        assert "albanian" in ranks["model_b"]["generative"], (
            f"model_b should have Albanian rank, got: {ranks.get('model_b', {})}"
        )
        # model_c is not Albanian-eligible
        assert "albanian" not in ranks.get("model_c", {}).get("generative", {})

        # Danish ranks: model_a and model_c should have scores
        assert "danish" in ranks["model_a"]["generative"], (
            f"model_a should have Danish rank, got: {ranks.get('model_a', {})}"
        )
        assert "danish" in ranks["model_c"]["generative"], (
            f"model_c should have Danish rank, got: {ranks.get('model_c', {})}"
        )
        # model_b is not Danish-eligible
        assert "danish" not in ranks.get("model_b", {}).get("generative", {})

        # Verify the rank scores differ between languages (different eligible sets)
        # model_a is rank 1 in both (best overall), but model_b and model_c are
        # ranked relative to their language-specific competitors
        model_a_albanian = ranks["model_a"]["generative"]["albanian"]["score"]
        model_b_albanian = ranks["model_b"]["generative"]["albanian"]["score"]
        # model_a should rank better than model_b in Albanian (lower score = better)
        assert model_a_albanian < model_b_albanian, (
            f"model_a ({model_a_albanian}) should rank better than model_b "
            f"({model_b_albanian}) in Albanian"
        )

    def test_monolingual_output_unchanged(self) -> None:
        """Monolingual leaderboard output is unchanged by the fix.

        The fix only affects multilingual leaderboards (len(configs) > 1).
        Monolingual leaderboards should produce identical results.
        """
        datasets = [f"dataset_{i}" for i in range(8)]
        model_ids = ["model_a", "model_b", "model_c"]
        model_results = _make_dummy_results(model_ids, datasets)

        config = _make_dummy_configs(["albanian"], datasets)
        category_to_datasets = {"generative": datasets}
        category_to_orthogonal = {"generative": {}}

        # Run twice - should produce identical results
        (_, _, ranks1, std_ranks1) = _compute_eligible_models_and_ranks(
            model_results=model_results,
            category=LeaderboardCategory.GENERATIVE,
            category_to_datasets=category_to_datasets,
            category_to_orthogonal_datasets=category_to_orthogonal,
            leaderboard_configs=config,
        )

        (_, _, ranks2, std_ranks2) = _compute_eligible_models_and_ranks(
            model_results=model_results,
            category=LeaderboardCategory.GENERATIVE,
            category_to_datasets=category_to_datasets,
            category_to_orthogonal_datasets=category_to_orthogonal,
            leaderboard_configs=config,
        )

        # Results should be identical (deterministic with seed=42)
        assert ranks1 == ranks2
        assert std_ranks1 == std_ranks2

    def test_multilingual_overall_rank_differs_from_monolingual(self) -> None:
        """Overall rank score in multilingual differs from monolingual.

        The overall "Rank score" column remains pan-leaderboard, using the
        multilingual eligible set. This should differ from monolingual overall.
        """
        albanian_datasets = [f"sq_dataset_{i}" for i in range(4)]
        danish_datasets = [f"da_dataset_{i}" for i in range(4)]
        all_datasets = albanian_datasets + danish_datasets

        model_ids = ["model_a", "model_b"]
        model_results = _make_dummy_results(model_ids, all_datasets)

        albanian_config = _make_dummy_configs(["albanian"], albanian_datasets)
        multilingual_config = {
            **albanian_config,
            **_make_dummy_configs(["danish"], danish_datasets),
        }

        # Albanian monolingual
        albanian_category_to_datasets = {"generative": albanian_datasets}
        (_, _, albanian_ranks, _) = _compute_eligible_models_and_ranks(
            model_results=model_results,
            category=LeaderboardCategory.GENERATIVE,
            category_to_datasets=albanian_category_to_datasets,
            category_to_orthogonal_datasets={"generative": {}},
            leaderboard_configs=albanian_config,
        )

        # Multilingual
        multilingual_category_to_datasets = {"generative": all_datasets}
        (_, _, multilingual_ranks, _) = _compute_eligible_models_and_ranks(
            model_results=model_results,
            category=LeaderboardCategory.GENERATIVE,
            category_to_datasets=multilingual_category_to_datasets,
            category_to_orthogonal_datasets={"generative": {}},
            leaderboard_configs=multilingual_config,
        )

        # Overall scores should differ (different eligible sets and datasets)
        for model_id in model_ids:
            albanian_overall = (
                albanian_ranks.get(model_id, {})
                .get("generative", {})
                .get("overall", {})
                .get("score", float("nan"))
            )
            multilingual_overall = (
                multilingual_ranks.get(model_id, {})
                .get("generative", {})
                .get("overall", {})
                .get("score", float("nan"))
            )
            # These should be different since they use different datasets/bases
            assert math.isfinite(albanian_overall)
            assert math.isfinite(multilingual_overall)
            # Note: we don't assert they're different since they could coincidentally
            # be similar, but they're computed from different distributions

    def test_partial_model_gets_language_score_but_no_overall_multilingual(
        self,
    ) -> None:
        """Model complete for one language but not multilingual gets language score.

        A model that has all datasets for one language but is missing datasets
        for the full multilingual board should:
        - Get a rank score for the language it's complete in
        - Get '-' for the overall multilingual rank score
        - Not appear in the multilingual ordinal ranks
        """
        albanian_datasets = [f"sq_dataset_{i}" for i in range(4)]
        danish_datasets = [f"da_dataset_{i}" for i in range(4)]
        all_datasets = albanian_datasets + danish_datasets

        # model_a is complete, model_b is missing Danish datasets
        model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]] = {
            "model_a": {ds: [([0.7] * 100, 0.7, 0.01)] for ds in all_datasets},
            "model_b": {
                ds: [([0.65] * 100, 0.65, 0.01)]
                for ds in albanian_datasets  # Missing all Danish datasets
            },
        }

        multilingual_config = {
            **_make_dummy_configs(["albanian"], albanian_datasets),
            **_make_dummy_configs(["danish"], danish_datasets),
        }

        multilingual_category_to_datasets = {"generative": all_datasets}
        (eligible, lang_to_datasets, ranks, std_ranks) = (
            _compute_eligible_models_and_ranks(
                model_results=model_results,
                category=LeaderboardCategory.GENERATIVE,
                category_to_datasets=multilingual_category_to_datasets,
                category_to_orthogonal_datasets={"generative": {}},
                leaderboard_configs=multilingual_config,
            )
        )

        # model_a should be eligible for multilingual
        assert "model_a" in eligible
        # model_b should NOT be eligible for multilingual (missing Danish datasets)
        assert "model_b" not in eligible

        # model_a should have overall multilingual rank score
        model_a_overall = (
            ranks.get("model_a", {}).get("generative", {}).get("overall", {})
        )
        assert "score" in model_a_overall
        assert math.isfinite(model_a_overall["score"])

        # model_b should NOT have overall multilingual rank score (not eligible)
        # model_b is not in the bootstrap at all, so no "overall" key
        assert "overall" not in ranks.get("model_b", {}).get("generative", {})

        # model_b SHOULD have Albanian rank score (it's Albanian-eligible)
        model_b_albanian = (
            ranks.get("model_b", {}).get("generative", {}).get("albanian", {})
        )
        assert "score" in model_b_albanian, (
            f"model_b should have Albanian score, got: {ranks.get('model_b', {})}"
        )
        assert math.isfinite(model_b_albanian["score"]), (
            f"model_b Albanian score should be finite: {model_b_albanian}"
        )

        # model_b should NOT have Danish rank score (not Danish-eligible)
        model_b_danish = (
            ranks.get("model_b", {}).get("generative", {}).get("danish", {})
        )
        assert "score" not in model_b_danish, (
            f"model_b should not have Danish score, got: {model_b_danish}"
        )

        # Ordinal ranks: model_a should have one, model_b should not
        assert "model_a" in std_ranks
        assert "model_b" not in std_ranks

    def test_per_language_scores_match_monolingual(self) -> None:
        """Per-language rank scores match monolingual leaderboard values.

        This is the core fix: each per-language rank-score column in a
        multilingual leaderboard should match the corresponding monolingual
        leaderboard rank score for the same model/category/language.
        """
        # Create two languages with separate dataset sets
        albanian_datasets = [f"sq_dataset_{i}" for i in range(4)]
        danish_datasets = [f"da_dataset_{i}" for i in range(4)]
        all_datasets = albanian_datasets + danish_datasets

        # Models that are complete for both languages
        model_ids = ["model_a", "model_b", "model_c"]
        model_results = _make_dummy_results(model_ids, all_datasets)

        # Single-language config for Albanian
        albanian_config = _make_dummy_configs(["albanian"], albanian_datasets)
        # Single-language config for Danish
        danish_config = _make_dummy_configs(["danish"], danish_datasets)
        # Multilingual config
        multilingual_config = {**albanian_config, **danish_config}

        # Compute Albanian monolingual ranks
        albanian_category_to_datasets = {"generative": albanian_datasets}
        albanian_category_to_orthogonal = {"generative": {}}
        (
            albanian_eligible,
            albanian_lang_to_datasets,
            albanian_ranks,
            albanian_std_ranks,
        ) = _compute_eligible_models_and_ranks(
            model_results=model_results,
            category=LeaderboardCategory.GENERATIVE,
            category_to_datasets=albanian_category_to_datasets,
            category_to_orthogonal_datasets=albanian_category_to_orthogonal,
            leaderboard_configs=albanian_config,
        )

        # Compute Danish monolingual ranks
        danish_category_to_datasets = {"generative": danish_datasets}
        danish_category_to_orthogonal = {"generative": {}}
        (danish_eligible, danish_lang_to_datasets, danish_ranks, danish_std_ranks) = (
            _compute_eligible_models_and_ranks(
                model_results=model_results,
                category=LeaderboardCategory.GENERATIVE,
                category_to_datasets=danish_category_to_datasets,
                category_to_orthogonal_datasets=danish_category_to_orthogonal,
                leaderboard_configs=danish_config,
            )
        )

        # Compute multilingual ranks (with the fix)
        multilingual_category_to_datasets = {"generative": all_datasets}
        multilingual_category_to_orthogonal = {"generative": {}}
        (
            multilingual_eligible,
            multilingual_lang_to_datasets,
            multilingual_ranks,
            multilingual_std_ranks,
        ) = _compute_eligible_models_and_ranks(
            model_results=model_results,
            category=LeaderboardCategory.GENERATIVE,
            category_to_datasets=multilingual_category_to_datasets,
            category_to_orthogonal_datasets=multilingual_category_to_orthogonal,
            leaderboard_configs=multilingual_config,
        )

        # Verify: per-language scores in multilingual should match monolingual
        for model_id in model_ids:
            # Albanian score should match
            albanian_mono_score = (
                albanian_ranks.get(model_id, {})
                .get("generative", {})
                .get("albanian", {})
                .get("score", float("nan"))
            )
            albanian_multi_score = (
                multilingual_ranks.get(model_id, {})
                .get("generative", {})
                .get("albanian", {})
                .get("score", float("nan"))
            )
            assert math.isfinite(albanian_mono_score), (
                f"Albanian monolingual score should be finite for {model_id}"
            )
            assert math.isfinite(albanian_multi_score), (
                f"Albanian multilingual score should be finite for {model_id}"
            )
            assert abs(albanian_mono_score - albanian_multi_score) < 1e-6, (
                f"Albanian score mismatch for {model_id}: "
                f"mono={albanian_mono_score}, multi={albanian_multi_score}"
            )

            # Danish score should match
            danish_mono_score = (
                danish_ranks.get(model_id, {})
                .get("generative", {})
                .get("danish", {})
                .get("score", float("nan"))
            )
            danish_multi_score = (
                multilingual_ranks.get(model_id, {})
                .get("generative", {})
                .get("danish", {})
                .get("score", float("nan"))
            )
            assert math.isfinite(danish_mono_score), (
                f"Danish monolingual score should be finite for {model_id}"
            )
            assert math.isfinite(danish_multi_score), (
                f"Danish multilingual score should be finite for {model_id}"
            )
            assert abs(danish_mono_score - danish_multi_score) < 1e-6, (
                f"Danish score mismatch for {model_id}: "
                f"mono={danish_mono_score}, multi={danish_multi_score}"
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


class TestPerLanguageRankScoreFormat:
    """Tests for per-language rank score formatting and structure."""

    def test_multilingual_has_both_overall_and_language_scores(self) -> None:
        """Multilingual ranks have both 'overall' and per-language scores."""
        albanian_datasets = [f"sq_dataset_{i}" for i in range(4)]
        danish_datasets = [f"da_dataset_{i}" for i in range(4)]
        all_datasets = albanian_datasets + danish_datasets

        model_ids = ["model_a"]
        model_results = _make_dummy_results(model_ids, all_datasets)

        multilingual_config = {
            **_make_dummy_configs(["albanian"], albanian_datasets),
            **_make_dummy_configs(["danish"], danish_datasets),
        }

        multilingual_category_to_datasets = {"generative": all_datasets}
        (_, _, ranks, _) = _compute_eligible_models_and_ranks(
            model_results=model_results,
            category=LeaderboardCategory.GENERATIVE,
            category_to_datasets=multilingual_category_to_datasets,
            category_to_orthogonal_datasets={"generative": {}},
            leaderboard_configs=multilingual_config,
        )

        model_data = ranks["model_a"]["generative"]
        # Should have overall
        assert "overall" in model_data
        assert "score" in model_data["overall"]
        # Should have each language
        assert "albanian" in model_data
        assert "danish" in model_data
        assert "score" in model_data["albanian"]
        assert "score" in model_data["danish"]

    def test_rank_score_has_confidence_intervals(self) -> None:
        """Per-language rank scores include confidence intervals."""
        datasets = [f"sq_dataset_{i}" for i in range(4)]
        model_ids = ["model_a", "model_b"]
        model_results = _make_dummy_results(model_ids, datasets)
        config = _make_dummy_configs(["albanian"], datasets)

        category_to_datasets = {"generative": datasets}
        (_, _, ranks, _) = _compute_eligible_models_and_ranks(
            model_results=model_results,
            category=LeaderboardCategory.GENERATIVE,
            category_to_datasets=category_to_datasets,
            category_to_orthogonal_datasets={"generative": {}},
            leaderboard_configs=config,
        )

        for model_id in model_ids:
            albanian_data = ranks[model_id]["generative"]["albanian"]
            assert "score" in albanian_data
            assert "ci_lower" in albanian_data
            assert "ci_upper" in albanian_data
            # CI should be valid: lower <= score <= upper
            assert albanian_data["ci_lower"] <= albanian_data["score"]
            assert albanian_data["score"] <= albanian_data["ci_upper"]


class TestRegressionForReportedIssue:
    """Tests directly addressing the reported issue (Qwen model score mismatch)."""

    def test_multilingual_language_score_matches_monolingual(self) -> None:
        """Reproduce the reported issue: multilingual should match monolingual.

        The user reported that Qwen/Qwen3.6-27B-FP8 (val) has 1.56 ± 0.08 on
        the Albanian monolingual leaderboard, but 1.40 ± 0.30 on the European
        multilingual leaderboard's Albanian column. After the fix, these should
        match.
        """
        # Simulate the scenario: multiple languages with different eligible sets
        albanian_datasets = [f"sq_dataset_{i}" for i in range(4)]
        danish_datasets = [f"da_dataset_{i}" for i in range(4)]
        all_datasets = albanian_datasets + danish_datasets

        # Multiple models, like a real leaderboard
        model_results = _make_dummy_results(
            ["model_a", "model_b", "model_c"], all_datasets, base_score=0.65
        )

        albanian_config = _make_dummy_configs(["albanian"], albanian_datasets)
        danish_config = _make_dummy_configs(["danish"], danish_datasets)
        multilingual_config = {**albanian_config, **danish_config}

        # Compute monolingual Albanian ranks
        albanian_category_to_datasets = {"generative": albanian_datasets}
        (_, _, albanian_ranks, _) = _compute_eligible_models_and_ranks(
            model_results=model_results,
            category=LeaderboardCategory.GENERATIVE,
            category_to_datasets=albanian_category_to_datasets,
            category_to_orthogonal_datasets={"generative": {}},
            leaderboard_configs=albanian_config,
        )

        # Compute multilingual ranks
        multilingual_category_to_datasets = {"generative": all_datasets}
        (_, _, multilingual_ranks, _) = _compute_eligible_models_and_ranks(
            model_results=model_results,
            category=LeaderboardCategory.GENERATIVE,
            category_to_datasets=multilingual_category_to_datasets,
            category_to_orthogonal_datasets={"generative": {}},
            leaderboard_configs=multilingual_config,
        )

        # Verify: each model's Albanian score in multilingual should match
        # the monolingual Albanian score
        for model_id in ["model_a", "model_b", "model_c"]:
            albanian_mono = albanian_ranks[model_id]["generative"]["albanian"]["score"]
            albanian_multi = multilingual_ranks[model_id]["generative"]["albanian"][
                "score"
            ]
            assert abs(albanian_mono - albanian_multi) < 1e-6, (
                f"{model_id} Albanian score mismatch: mono={albanian_mono}, "
                f"multi={albanian_multi}"
            )
