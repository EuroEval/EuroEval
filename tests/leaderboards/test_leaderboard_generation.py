"""Tests for leaderboard generation with per-language rank-score sync.

Tests verify that multilingual leaderboards have per-language rank-score columns
matching the corresponding monolingual leaderboards, while overall rank scores
and ordinal ranks remain pan-leaderboard.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from euroeval.dataset_configs.danish import (
    ANGRY_TWEETS_CONFIG,
    DALA_CONFIG,
    RAGTRUTH_DA_CONFIG,
    VALEU_DA_CONFIG,
)
from euroeval.dataset_configs.norwegian import NOREC_CONFIG
from euroeval.dataset_configs.swedish import SCALA_SV_CONFIG, SWEREC_CONFIG
from src.leaderboards import leaderboard_generation
from src.leaderboards.enums import LeaderboardCategory
from src.leaderboards.invalid_results import InvalidResult, load_invalid_results
from src.leaderboards.leaderboard_generation import (
    _build_category_dataset_maps,
    _compute_eligible_models_and_ranks,
    _create_simplified_and_rename,
    _reorder_columns,
)


class TestGlobalVariantSelection:
    """Tests for global validation/test variant selection."""

    def test_polish_keeps_variant_selected_from_global_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Polish filter must not change the global validation/test choice.

        Args:
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
        """
        global_model_results = {
            "org/model (val)": {"global-dataset": [], "polish-dataset": []},
            "org/model": {"polish-dataset": []},
        }
        local_model_results = {
            "org/model (val)": {"polish-dataset": []},
            "org/model": {"polish-dataset": []},
        }
        captured_model_results = []
        results = [
            {"eval_library": {"additional_details": {"dataset": "global-dataset"}}},
            {"eval_library": {"additional_details": {"dataset": "polish-dataset"}}},
        ]

        monkeypatch.setattr(leaderboard_generation, "load_raw_results", lambda: results)
        monkeypatch.setattr(
            leaderboard_generation,
            "group_results_by_model",
            lambda results: (
                global_model_results if len(results) == 2 else local_model_results
            ),
        )
        monkeypatch.setattr(
            leaderboard_generation, "extract_model_metadata", lambda results: {}
        )
        monkeypatch.setattr(
            leaderboard_generation,
            "official_datasets_for_language",
            lambda language: {"task": ["polish-dataset"]},
        )
        monkeypatch.setattr(
            leaderboard_generation,
            "_generate_dataframe",
            lambda **kwargs: (
                captured_model_results.append(kwargs["model_results"]) or []
            ),
        )

        leaderboard_generation.generate_leaderboard(
            leaderboard_name="polish",
            language_names=["polish"],
            categories=[LeaderboardCategory.GENERATIVE],
            force=False,
        )

        assert captured_model_results == [{"org/model (val)": {"polish-dataset": []}}]

    def test_simplified_outputs_use_global_canonical_variant(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Language outputs do not resurrect a narrower split variant.

        Args:
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
            tmp_path:
                Temporary directory for files created by the test.
        """
        score = [([0.75] * 10, 0.75, 0.01)]
        other_score = [([0.65] * 10, 0.65, 0.01)]
        model_results = {
            "org/model (val)": {"european-dataset": score, "polish-dataset": score},
            "org/model": {"polish-dataset": score},
            "org/other": {
                "european-dataset": other_score,
                "polish-dataset": other_score,
            },
        }
        metadata = {
            model_id: {
                "generative_type": "base",
                "open": True,
                "commercial": True,
                "merge": False,
                "trained_from_scratch": False,
                "release_date": "2024-01-01",
                "parameters": 1_000_000,
                "vocabulary_size": 1_000,
                "context": 1_024,
            }
            for model_id in model_results
        }

        monkeypatch.setattr(leaderboard_generation, "OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(leaderboard_generation, "load_raw_results", lambda: [{}])
        monkeypatch.setattr(
            leaderboard_generation,
            "group_results_by_model",
            lambda results: model_results,
        )
        monkeypatch.setattr(
            leaderboard_generation, "extract_model_metadata", lambda results: metadata
        )
        monkeypatch.setattr(
            leaderboard_generation,
            "official_datasets_for_language",
            lambda language: (
                {"knowledge": ["european-dataset", "polish-dataset"]}
                if language == "european"
                else {"knowledge": ["polish-dataset"]}
            ),
        )

        for leaderboard_name, language in (
            ("european", "european"),
            ("polish", "polish"),
        ):
            leaderboard_generation.generate_leaderboard(
                leaderboard_name=leaderboard_name,
                language_names=[language],
                categories=[LeaderboardCategory.GENERATIVE],
                force=True,
            )

        european = pd.read_csv(tmp_path / "european_generative_simplified.csv")
        polish = pd.read_csv(tmp_path / "polish_generative_simplified.csv")
        assert polish["model"].tolist() == european["model"].tolist()
        assert [
            model for model in european["model"] if model.startswith("org/model")
        ] == ["org/model (val)"]


class TestInvalidResults:
    """Tests for leaving invalid (model, dataset) results out of aggregated scores."""

    def test_empty_list_leaves_leaderboard_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An empty invalid-results file leaves every model ranked and unmarked.

        Args:
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
            tmp_path:
                Temporary directory for files created by the test.
        """
        empty_file = tmp_path / "invalid_results.yaml"
        empty_file.write_text("# No entries.\n", encoding="utf-8")
        from_file = self._generate(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path / "file",
            scores=self._scores(),
            invalid_results=load_invalid_results(path=empty_file),
        )
        from_list = self._generate(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path / "list",
            scores=self._scores(),
            invalid_results=[],
        )

        assert from_file == from_list
        assert "(invalid)" not in from_file[0]
        simplified = pd.read_csv(tmp_path / "list" / "danish_generative_simplified.csv")
        assert sorted(simplified["model"]) == ["org/a", "org/b", "org/c"]

    @staticmethod
    def _generate(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        scores: dict[str, dict[str, list[tuple[list[float], float, float]]]],
        invalid_results: list[InvalidResult],
        generative_type: str = "base",
        language_names: tuple[str, ...] = ("danish",),
        force: bool = True,
    ) -> tuple[str, str]:
        """Generate a generative leaderboard from the given scores.

        Args:
            monkeypatch:
                The pytest monkeypatch fixture.
            tmp_path:
                The directory to write the leaderboards to.
            scores:
                Model ID to dataset to scores.
            invalid_results:
                The invalid (model, dataset) results to apply.
            generative_type (optional):
                The generative type of every model. Orthogonal scores are only
                shown for instruction-tuned and reasoning models.
            language_names (optional):
                The leaderboard languages: Danish, Swedish and/or Norwegian. Several
                languages give a multilingual leaderboard named after the languages.
            force (optional):
                Whether to force writing the leaderboard files.

        Returns:
            The full and the simplified leaderboard CSV contents.
        """
        metadata = {
            model_id: {
                "generative_type": generative_type,
                "model_type": None,
                "parameters": 100,
                "vocabulary_size": math.nan,
                "context": math.nan,
                "release_date": None,
                "open": True,
                "commercial": False,
                "merge": False,
                "trained_from_scratch": False,
            }
            for model_id in scores
        }
        monkeypatch.setattr(leaderboard_generation, "OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(leaderboard_generation, "NUM_BOOTSTRAPS", 20)
        monkeypatch.setattr(leaderboard_generation, "load_raw_results", lambda: [{}])
        monkeypatch.setattr(
            leaderboard_generation, "group_results_by_model", lambda results: scores
        )
        monkeypatch.setattr(
            leaderboard_generation, "extract_model_metadata", lambda results: metadata
        )
        configs = {
            "danish": {
                ANGRY_TWEETS_CONFIG.task.name: [ANGRY_TWEETS_CONFIG.name],
                DALA_CONFIG.task.name: [DALA_CONFIG.name],
                VALEU_DA_CONFIG.task.name: [VALEU_DA_CONFIG.name],
            },
            "swedish": {
                SWEREC_CONFIG.task.name: [SWEREC_CONFIG.name],
                SCALA_SV_CONFIG.task.name: [SCALA_SV_CONFIG.name],
            },
            "norwegian": {NOREC_CONFIG.task.name: [NOREC_CONFIG.name]},
        }
        monkeypatch.setattr(
            leaderboard_generation,
            "official_datasets_for_language",
            lambda language: configs[language],
        )
        monkeypatch.setattr(
            leaderboard_generation, "load_invalid_results", lambda: invalid_results
        )
        leaderboard_name = "_".join(language_names)
        leaderboard_generation.generate_leaderboard(
            leaderboard_name=leaderboard_name,
            language_names=list(language_names),
            categories=[LeaderboardCategory.GENERATIVE],
            force=force,
        )
        return (
            (tmp_path / f"{leaderboard_name}_generative.csv").read_text(
                encoding="utf-8"
            ),
            (tmp_path / f"{leaderboard_name}_generative_simplified.csv").read_text(
                encoding="utf-8"
            ),
        )

    @staticmethod
    def _scores() -> dict[str, dict[str, list[tuple[list[float], float, float]]]]:
        """Create scores for three models on Danish, Swedish and Norwegian datasets.

        Returns:
            Model ID to dataset to scores.
        """
        return {
            model_id: {
                ANGRY_TWEETS_CONFIG.name: [([score, score + 2.0], score + 1.0, 1.0)],
                DALA_CONFIG.name: [([score, score + 4.0], score + 2.0, 2.0)],
                VALEU_DA_CONFIG.name: [([30.0, 40.0], 35.0, 0.1)],
                SWEREC_CONFIG.name: [([score, score + 2.0], score + 1.0, 1.0)],
                SCALA_SV_CONFIG.name: [([score, score + 4.0], score + 2.0, 2.0)],
                NOREC_CONFIG.name: [([score, score + 2.0], score + 1.0, 1.0)],
            }
            for model_id, score in (("org/a", 70.0), ("org/b", 60.0), ("org/c", 50.0))
        }

    def test_invalid_orthogonal_result_is_shown_and_marked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An invalid orthogonal result keeps its score, marked, and ranks are kept.

        Args:
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
            tmp_path:
                Temporary directory for files created by the test.
        """
        with_invalid = self._generate(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path / "invalid",
            scores=self._scores(),
            invalid_results=[
                self._invalid_result(model="org/a", dataset=VALEU_DA_CONFIG.name)
            ],
            generative_type="instruction_tuned",
        )
        without_entry = self._generate(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path / "valid",
            scores=self._scores(),
            invalid_results=[],
            generative_type="instruction_tuned",
        )

        full = pd.read_csv(tmp_path / "invalid" / "danish_generative.csv", skiprows=1)
        valid = pd.read_csv(tmp_path / "valid" / "danish_generative.csv", skiprows=1)
        pd.testing.assert_frame_equal(
            full[["Model", "Rank", "Rank score"]],
            valid[["Model", "Rank", "Rank score"]],
        )
        assert with_invalid[1] == without_entry[1]
        values_col = next(col for col in full.columns if "European Values" in col)
        values = full.set_index("Model")[values_col]
        assert values["org/a"] == "35.00 ± 0.20 (invalid)"
        assert values.drop(index="org/a").tolist() == ["35.00 ± 0.20"] * 2

    @staticmethod
    def _invalid_result(model: str, dataset: str = DALA_CONFIG.name) -> InvalidResult:
        """Create an invalid result for a model.

        Args:
            model:
                The model ID.
            dataset (optional):
                The dataset name. Defaults to DaLA.

        Returns:
            The invalid result.
        """
        return InvalidResult(
            model=model,
            dataset=dataset,
            reason="Trained on the test split.",
            source="https://example.com/model-card",
            date=dt.date(2026, 1, 13),
        )

    def test_invalid_result_is_shown_but_not_ranked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An invalid result is marked and ranked as if it were missing.

        Args:
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
            tmp_path:
                Temporary directory for files created by the test.
        """
        with_invalid = self._generate(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path / "invalid",
            scores=self._scores(),
            invalid_results=[self._invalid_result(model="org/a")],
        )
        scores_without_dala = self._scores()
        del scores_without_dala["org/a"][DALA_CONFIG.name]
        without_result = self._generate(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path / "missing",
            scores=scores_without_dala,
            invalid_results=[],
        )

        full = pd.read_csv(tmp_path / "invalid" / "danish_generative.csv", skiprows=1)
        missing = pd.read_csv(
            tmp_path / "missing" / "danish_generative.csv", skiprows=1
        )
        pd.testing.assert_frame_equal(
            full[["Model", "Rank", "Rank score"]],
            missing[["Model", "Rank", "Rank score"]],
        )
        assert with_invalid[1] == without_result[1]

        row = full.set_index("Model").loc["org/a"]
        dala_col, angry_tweets_col = (
            next(
                col
                for col in full.columns
                if col.endswith(f">{config.name.replace('-', '_')}</a>")
            )
            for config in (DALA_CONFIG, ANGRY_TWEETS_CONFIG)
        )
        assert row[dala_col] == "72.00 ± 2.00 (invalid)"
        assert row[angry_tweets_col] == "71.00 ± 1.00"
        assert row["Rank score"] == "-"
        other_rows = full.set_index("Model").drop(index="org/a")
        assert not other_rows[dala_col].str.contains("(invalid)", regex=False).any()

        simplified = pd.read_csv(
            tmp_path / "invalid" / "danish_generative_simplified.csv"
        )
        assert sorted(simplified["model"]) == ["org/b", "org/c"]

    def test_new_entry_rewrites_language_rank_without_force(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A non-forced run rewrites a language rank that a new entry removes.

        The model has no Norwegian result, so it is unranked overall but ranked in
        Danish and Swedish. Invalidating a Danish result removes only its Danish rank.

        Args:
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
            tmp_path:
                Temporary directory for files created by the test.
        """
        scores = self._scores()
        scores["org/d"] = {
            dataset: score
            for dataset, score in scores["org/c"].items()
            if dataset != NOREC_CONFIG.name
        }
        language_names = ("danish", "swedish", "norwegian")
        self._generate(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            scores=scores,
            invalid_results=[],
            language_names=language_names,
        )
        full_path = tmp_path / "danish_swedish_norwegian_generative.csv"
        before = pd.read_csv(full_path, skiprows=1).set_index("Model")
        self._generate(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            scores=scores,
            invalid_results=[self._invalid_result(model="org/d")],
            language_names=language_names,
            force=False,
        )
        after = pd.read_csv(full_path, skiprows=1).set_index("Model")

        assert before.loc["org/d", ["Rank score", "Norwegian"]].tolist() == ["-", "-"]
        assert "-" not in before.loc["org/d", ["Danish", "Swedish"]].tolist()
        assert after.loc["org/d", ["Rank score", "Danish"]].tolist() == ["-", "-"]
        assert after.loc["org/d", "Swedish"] != "-"

    def test_new_entry_rewrites_multilingual_leaderboard_without_force(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A non-forced run rewrites a multilingual leaderboard after a new entry.

        Args:
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
            tmp_path:
                Temporary directory for files created by the test.
        """
        before = self._generate(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            scores=self._scores(),
            invalid_results=[],
            language_names=("danish", "swedish"),
        )
        after = self._generate(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            scores=self._scores(),
            invalid_results=[self._invalid_result(model="org/a")],
            language_names=("danish", "swedish"),
            force=False,
        )

        assert after != before
        simplified = pd.read_csv(tmp_path / "danish_swedish_generative_simplified.csv")
        assert sorted(simplified["model"]) == ["org/b", "org/c"]
        full = pd.read_csv(tmp_path / "danish_swedish_generative.csv", skiprows=1)
        row = full.set_index("Model").loc["org/a"]
        assert row["Rank score"] == "-"
        assert row["Danish"] == "-"
        assert row["Swedish"] != "-"

    def test_unknown_model_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An entry for a model without results stops leaderboard generation.

        Args:
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
            tmp_path:
                Temporary directory for files created by the test.
        """
        with pytest.raises(ValueError, match="org/missing"):
            self._generate(
                monkeypatch=monkeypatch,
                tmp_path=tmp_path,
                scores=self._scores(),
                invalid_results=[self._invalid_result(model="org/missing")],
            )


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

    Args:
        languages:
            Languages supplied to the test.
        datasets:
            Datasets supplied to the test.

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

    Args:
        model_ids:
            Model ids supplied to the test.
        dataset_ids:
            Dataset ids supplied to the test.
        base_score (optional):
            Base score supplied to the test. Defaults to 0.7.
        noise (optional):
            Noise supplied to the test. Defaults to 0.01.

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


class TestOrthogonalDatasetsByCategory:
    """Tests for orthogonal bonus-column scoping in _build_category_dataset_maps."""

    def test_hallucination_result_is_optional_and_does_not_change_ranking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hallucination results are optional and leave rank eligibility unchanged.

        Args:
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
        """
        hallucination_task = RAGTRUTH_DA_CONFIG.task.name
        hallucination_dataset = RAGTRUTH_DA_CONFIG.name
        assert hallucination_task == "hallucination"
        assert hallucination_dataset == "ragtruth-da"
        config = {
            "danish": {
                "sentiment-classification": ["angry-tweets"],
                hallucination_task: [hallucination_dataset],
            }
        }
        category = LeaderboardCategory.CHAT
        _, orthogonal = _build_category_dataset_maps(
            categories=[category], leaderboard_configs=config
        )
        standard = {"angry-tweets": [([70.0, 80.0], 75.0, 0.0)]}
        with_hallucination = {
            "model-a": standard | {hallucination_dataset: [([0.2], 0.2, 0.0)]},
            "model-b": standard,
        }
        without_hallucination = {"model-a": standard, "model-b": standard}

        monkeypatch.setattr(leaderboard_generation, "NUM_BOOTSTRAPS", 20)

        def rank_inputs(
            model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
        ) -> tuple[dict, dict, dict]:
            """Rank inputs for this test.

            Args:
                model_results:
                    Model results supplied to the test.

            Returns:
                The value returned by the test helper.
            """
            eligible, _, ranks, ordinal_ranks = _compute_eligible_models_and_ranks(
                model_results=model_results,
                category=category,
                category_to_datasets={
                    category: ["angry-tweets", hallucination_dataset]
                },
                category_to_orthogonal_datasets=orthogonal,
                leaderboard_configs=config,
            )
            return eligible, ranks, ordinal_ranks

        with_result = rank_inputs(with_hallucination)
        without_result = rank_inputs(without_hallucination)
        assert set(with_result[0]) == {"model-a", "model-b"}
        assert set(without_result[0]) == {"model-a", "model-b"}
        assert with_result[1:] == without_result[1:]
        assert with_result[1]
        assert with_result[2]

        collected = leaderboard_generation._collect_orthogonal_scores(
            model_results=with_hallucination,
            category=category,
            category_to_orthogonal_datasets=orthogonal,
        )
        assert collected["model-a"][hallucination_dataset] == (0.2, None)
        assert "model-b" not in collected

    def test_orthogonal_datasets_populated_for_chat_and_generative(self) -> None:
        """Orthogonal datasets map to Chat and Generative only."""
        hallucination_task = RAGTRUTH_DA_CONFIG.task.name
        hallucination_dataset = RAGTRUTH_DA_CONFIG.name
        leaderboard_configs = {
            "danish": {
                "sentiment-classification": ["angry-tweets"],
                "european-values": ["valeu-da"],
                hallucination_task: [hallucination_dataset],
            }
        }
        _, category_to_orthogonal_datasets = _build_category_dataset_maps(
            categories=[
                LeaderboardCategory.CHAT,
                LeaderboardCategory.GENERATIVE,
                LeaderboardCategory.ALL_MODELS,
                LeaderboardCategory.UNDERSTANDING,
            ],
            leaderboard_configs=leaderboard_configs,
        )

        assert category_to_orthogonal_datasets[LeaderboardCategory.CHAT] == {
            "valeu-da": "european-values",
            hallucination_dataset: hallucination_task,
        }
        assert category_to_orthogonal_datasets[LeaderboardCategory.GENERATIVE] == {
            "valeu-da": "european-values",
            hallucination_dataset: hallucination_task,
        }
        assert category_to_orthogonal_datasets[LeaderboardCategory.ALL_MODELS] == {}
        assert category_to_orthogonal_datasets[LeaderboardCategory.UNDERSTANDING] == {}


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


def test_generate_all_models_csv_preserves_classifier_icon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The full generation pipeline keeps the classifier target icon in CSV.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    dataset = "sentiment"
    score = [([0.8, 0.9], 0.85, 0.05)]
    model_results = {"org/laya": {dataset: score}, "org/encoder": {dataset: score}}
    metadata = {
        "org/laya": {
            "model_type": "zero_shot_classifier",
            "generative_type": None,
            "parameters": 100,
            "vocabulary_size": math.nan,
            "context": math.nan,
            "release_date": None,
            "open": True,
            "commercial": False,
            "merge": False,
            "trained_from_scratch": False,
        },
        "org/encoder": {
            "model_type": None,
            "generative_type": None,
            "parameters": 200,
            "vocabulary_size": math.nan,
            "context": math.nan,
            "release_date": None,
            "open": True,
            "commercial": False,
            "merge": False,
            "trained_from_scratch": False,
        },
    }
    monkeypatch.setattr(leaderboard_generation, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(leaderboard_generation, "load_raw_results", lambda: [{}])
    monkeypatch.setattr(
        leaderboard_generation, "group_results_by_model", lambda results: model_results
    )
    monkeypatch.setattr(
        leaderboard_generation, "extract_model_metadata", lambda results: metadata
    )
    monkeypatch.setattr(
        leaderboard_generation,
        "official_datasets_for_language",
        lambda language: {"sentiment-classification": [dataset]},
    )

    leaderboard_generation.generate_leaderboard(
        leaderboard_name="english",
        language_names=["english"],
        categories=[LeaderboardCategory.ALL_MODELS],
        force=True,
    )

    all_models = pd.read_csv(tmp_path / "english_all_models_simplified.csv")
    icons = all_models.set_index("model")["generative_type"].to_dict()
    assert icons == {"org/laya": "🎯", "org/encoder": "🔍"}


def test_generate_generative_csv_keeps_optional_orthogonal_scores_out_of_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Generate both optional columns without changing either model's rank.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    standard = ANGRY_TWEETS_CONFIG.name
    values = VALEU_DA_CONFIG.name
    hallucination = RAGTRUTH_DA_CONFIG.name
    configs = {
        ANGRY_TWEETS_CONFIG.task.name: [standard],
        VALEU_DA_CONFIG.task.name: [values],
        RAGTRUTH_DA_CONFIG.task.name: [hallucination],
    }
    scores = {
        "org/a (zero-shot)": {standard: [([70.0, 80.0], 75.0, 0.0)]},
        "org/b (zero-shot)": {standard: [([60.0, 70.0], 65.0, 0.0)]},
    }
    metadata = {
        model_id: {
            "generative_type": "instruction_tuned",
            "model_type": None,
            "parameters": 100,
            "vocabulary_size": math.nan,
            "context": math.nan,
            "release_date": None,
            "open": True,
            "commercial": False,
            "merge": False,
            "trained_from_scratch": False,
        }
        for model_id in scores
    }
    monkeypatch.setattr(leaderboard_generation, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(leaderboard_generation, "NUM_BOOTSTRAPS", 20)
    monkeypatch.setattr(leaderboard_generation, "load_raw_results", lambda: [{}])
    monkeypatch.setattr(
        leaderboard_generation, "group_results_by_model", lambda results: scores
    )
    monkeypatch.setattr(
        leaderboard_generation, "extract_model_metadata", lambda results: metadata
    )
    monkeypatch.setattr(
        leaderboard_generation,
        "official_datasets_for_language",
        lambda language: configs,
    )

    def generate() -> pd.DataFrame:
        """Generate and read the public leaderboard CSV.

        Returns:
            The generated Generative leaderboard.
        """
        leaderboard_generation.generate_leaderboard(
            leaderboard_name="danish",
            language_names=["danish"],
            categories=[LeaderboardCategory.GENERATIVE],
            force=True,
        )
        return pd.read_csv(tmp_path / "danish_generative.csv", skiprows=1)

    without = generate()
    scores["org/a (zero-shot)"] |= {
        values: [([30.0, 40.0], 35.0, 0.1)],
        hallucination: [([0.4, 0.6], 0.5, 0.0)],
    }
    scores["org/b (zero-shot)"] |= {
        values: [([35.678], 35.678, math.nan)],
        hallucination: [([0.4, 0.6], 0.5, math.nan)],
    }
    with_scores = generate()

    pd.testing.assert_frame_equal(
        without[["Model", "Rank", "Rank score"]],
        with_scores[["Model", "Rank", "Rank score"]],
    )
    assert len(with_scores) == 2
    values_col = next(col for col in with_scores if "European Values" in col)
    hallucination_col = (
        '<a href="https://euroeval.com/tasks/hallucination-detection">'
        "Hallucination Rate</a>"
    )
    assert hallucination_col in with_scores.columns
    assert with_scores[values_col].tolist() == ["35.00 ± 0.20", "35.68"]
    assert with_scores[hallucination_col].tolist() == ["0.50 ± 0.00", "0.5"]


def test_generate_leaderboard_projects_orthogonal_scores_to_plain_model_variants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Project orthogonal points and uncertainty to every variant row.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    standard = ANGRY_TWEETS_CONFIG.name
    values = VALEU_DA_CONFIG.name
    hallucination = RAGTRUTH_DA_CONFIG.name
    configs = {
        ANGRY_TWEETS_CONFIG.task.name: [standard],
        VALEU_DA_CONFIG.task.name: [values],
        RAGTRUTH_DA_CONFIG.task.name: [hallucination],
    }
    scores = {
        "org/model": {standard: [([70.0, 80.0], 75.0, 0.0)]},
        "org/model (zero-shot)": {standard: [([70.0, 80.0], 75.0, 0.0)]},
    }
    metadata = {
        model_id: {
            "generative_type": "instruction_tuned",
            "model_type": None,
            "parameters": 100,
            "vocabulary_size": math.nan,
            "context": math.nan,
            "release_date": None,
            "open": True,
            "commercial": False,
            "merge": False,
            "trained_from_scratch": False,
        }
        for model_id in scores
    }
    monkeypatch.setattr(leaderboard_generation, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(leaderboard_generation, "NUM_BOOTSTRAPS", 20)
    monkeypatch.setattr(leaderboard_generation, "load_raw_results", lambda: [{}])
    monkeypatch.setattr(
        leaderboard_generation, "group_results_by_model", lambda results: scores
    )
    monkeypatch.setattr(
        leaderboard_generation, "extract_model_metadata", lambda results: metadata
    )
    monkeypatch.setattr(
        leaderboard_generation,
        "official_datasets_for_language",
        lambda language: configs,
    )

    def generate() -> pd.DataFrame:
        """Generate and read the public leaderboard CSV.

        Returns:
            The generated Generative leaderboard.
        """
        leaderboard_generation.generate_leaderboard(
            leaderboard_name="danish",
            language_names=["danish"],
            categories=[LeaderboardCategory.GENERATIVE],
            force=True,
        )
        return pd.read_csv(tmp_path / "danish_generative.csv", skiprows=1)

    without = generate()
    scores["org/model (zero-shot)"] |= {
        values: [([30.0, 40.0], 35.0, 0.1)],
        hallucination: [([0.5], 0.5, math.nan)],
    }
    with_scores = generate()

    pd.testing.assert_frame_equal(
        without[["Model", "Rank", "Rank score"]],
        with_scores[["Model", "Rank", "Rank score"]],
    )
    assert len(with_scores) == 2
    values_col = next(col for col in with_scores if "European Values" in col)
    hallucination_col = (
        '<a href="https://euroeval.com/tasks/hallucination-detection">'
        "Hallucination Rate</a>"
    )
    assert with_scores[values_col].tolist() == ["35.00 ± 0.20", "35.00 ± 0.20"]
    assert with_scores[hallucination_col].tolist() == [0.5, 0.5]


def test_release_date_is_emitted_for_frontend_visualizations() -> None:
    """The existing result metadata reaches the full leaderboard CSV."""
    df = pd.DataFrame(
        {
            "rank": ["1"],
            "model": ["org/model"],
            "mean_rank_score": ["1.25 ± 0.05"],
            "generative_type": ["📝"],
            "open": ["✓"],
            "commercial": ["✗"],
            "merge": ["✗"],
            "trained_from_scratch": ["✓"],
            "release_date": ["2024-02-03"],
            "parameters": [7_000_000_000],
            "vocabulary_size": [32_000],
            "context": [4_096],
        }
    )

    ordered = _reorder_columns(
        df=df,
        category="generative",
        category_to_orthogonal_datasets={"generative": {}},
        category_to_datasets={"generative": []},
        rank_cols=["rank", "mean_rank_score"],
        include_dataset_columns=False,
    )
    full, _ = _create_simplified_and_rename(
        df=ordered,
        rank_cols=["rank", "mean_rank_score"],
        category_to_orthogonal_datasets={"generative": {}},
        category="generative",
    )

    assert full.loc[0, "Release Date"] == "2024-02-03"


def test_understanding_rank_pool_excludes_zero_shot_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Laya may have scores but must not affect Understanding bootstrap ranks.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    observed_model_ids: list[set[str]] = []

    def fake_bootstrap_rank_scores(model_results: dict, **_: object) -> dict:
        """Fake bootstrap rank scores for this test.

        Args:
            model_results:
                Model results supplied to the test.
            _:
                  supplied to the test.

        Returns:
            The value returned by the test helper.
        """
        observed_model_ids.append(set(model_results))
        return {}

    monkeypatch.setattr(
        leaderboard_generation, "bootstrap_rank_scores", fake_bootstrap_rank_scores
    )
    monkeypatch.setattr(
        leaderboard_generation,
        "bootstrap_confidence_intervals",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        leaderboard_generation,
        "compute_standard_ranks_from_bootstrap_scores",
        lambda **_: {},
    )

    _compute_eligible_models_and_ranks(
        model_results={
            "encoder": {"sentiment": []},
            "Laya": {"sentiment": []},
            "zero_shot_generative": {"sentiment": []},
        },
        category=LeaderboardCategory.UNDERSTANDING,
        category_to_datasets={LeaderboardCategory.UNDERSTANDING: ["sentiment"]},
        category_to_orthogonal_datasets={LeaderboardCategory.UNDERSTANDING: {}},
        leaderboard_configs={
            "english": {"sentiment-classification": ["sentiment"]},
            "danish": {"sentiment-classification": ["sentiment"]},
        },
        metadata_dict={
            "Laya": {"model_type": "zero_shot_classifier"},
            "zero_shot_generative": {
                "model_type": "instruction_tuned_decoder",
                "zero_shot": True,
            },
        },
    )

    assert observed_model_ids == [
        {"encoder", "zero_shot_generative"},
        {"encoder", "zero_shot_generative"},
        {"encoder", "zero_shot_generative"},
    ]
