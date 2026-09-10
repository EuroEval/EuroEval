"""Focused tests for aggregate core-model selection."""

from __future__ import annotations

import math
from unittest.mock import Mock

import numpy as np
import pytest

import leaderboards.core_models as core_models
from leaderboards.core_models import (
    CoreModel,
    ModelType,
    SizeBucket,
    _pareto_categories_per_model,
    build_core_model_list,
)
from leaderboards.enums import LeaderboardCategory


def _model_results(*datasets: str) -> dict[str, list[tuple[list[float], float, float]]]:
    return {dataset: [([1.0], 1.0, 1.0)] for dataset in datasets}


def test_aggregate_pareto_requires_complete_coverage_and_unions_categories() -> None:
    """Require complete datasets and retain qualifying decoder categories."""
    configs = {
        "europe": {
            "sentiment-classification": ["sentiment"],
            "summarization": ["summary"],
            "european-values": ["orthogonal"],
        }
    }
    results = {
        "encoder": _model_results("sentiment"),
        "small": _model_results("sentiment", "summary"),
        "large": _model_results("sentiment", "summary"),
        "partial": _model_results("summary"),
    }
    metadata = {
        "encoder": {"parameters": 1.0},
        "small": {"parameters": 1.0},
        "large": {"parameters": 2.0},
        "partial": {"parameters": 3.0},
    }
    model_types = {
        "encoder": ModelType.ENCODER,
        "small": ModelType.INSTRUCTION_TUNED_DECODER,
        "large": ModelType.INSTRUCTION_TUNED_DECODER,
        "partial": ModelType.INSTRUCTION_TUNED_DECODER,
    }
    scores = {
        "encoder": {
            LeaderboardCategory.ALL_MODELS.value: {"overall": np.array([1.0, 1.0])}
        },
        "small": {
            LeaderboardCategory.GENERATIVE.value: {"overall": np.array([1.0, 1.0])},
            LeaderboardCategory.ALL_MODELS.value: {"overall": np.array([1.0, 1.0])},
        },
        "large": {
            LeaderboardCategory.GENERATIVE.value: {"overall": np.array([2.0, 2.0])},
            LeaderboardCategory.ALL_MODELS.value: {"overall": np.array([2.0, 2.0])},
        },
        "partial": {
            LeaderboardCategory.GENERATIVE.value: {"overall": np.array([0.0, 0.0])},
            LeaderboardCategory.ALL_MODELS.value: {"overall": np.array([0.0, 0.0])},
        },
    }

    pareto = _pareto_categories_per_model(
        bootstrap_scores=scores,
        model_results=results,
        configs=configs,
        metadata=metadata,
        model_types=model_types,
    )

    assert pareto["encoder"] == {LeaderboardCategory.ALL_MODELS.value}
    assert pareto["small"] == {
        LeaderboardCategory.GENERATIVE.value,
        LeaderboardCategory.ALL_MODELS.value,
    }
    assert "large" not in pareto
    assert "partial" not in pareto


def test_pipeline_excludes_partial_models_before_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial models cannot change category distributions or the frontier."""
    configs = {
        "language-a": {"sentiment-classification": ["dataset-a"]},
        "language-b": {"sentiment-classification": ["dataset-b"]},
    }
    results = {
        "strong": _model_results("dataset-a", "dataset-b"),
        "peer": _model_results("dataset-a", "dataset-b"),
        "partial": _model_results("dataset-a"),
    }
    metadata = {model_id: {"parameters": 1.0} for model_id in results}
    model_types = {
        model_id: ModelType.INSTRUCTION_TUNED_DECODER for model_id in results
    }

    def fake_bootstrap_rank_scores(
        *,
        model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
        configs: dict[str, dict[str, list[str]]],
        n_bootstraps: int,
        seed: int,
        categories: tuple[LeaderboardCategory, ...],
    ) -> dict[str, dict[str, dict[str, np.ndarray]]]:
        del configs, n_bootstraps, seed
        scores = {"strong": 1.0, "peer": 2.0, "partial": 0.0}
        return {
            model_id: {
                category.value: {"overall": np.full(4, scores[model_id])}
                for category in categories
            }
            for model_id in model_results
        }

    bootstrap = Mock(side_effect=fake_bootstrap_rank_scores)
    monkeypatch.setattr(core_models, "bootstrap_rank_scores", bootstrap)

    pareto = _pareto_categories_per_model(
        model_results=results,
        configs=configs,
        metadata=metadata,
        model_types=model_types,
    )

    assert [set(call.kwargs["model_results"]) for call in bootstrap.call_args_list] == [
        {"strong", "peer"},
        {"strong", "peer"},
    ]
    assert all(call.kwargs["configs"] == configs for call in bootstrap.call_args_list)
    assert pareto == {
        "strong": {
            LeaderboardCategory.GENERATIVE.value,
            LeaderboardCategory.ALL_MODELS.value,
        }
    }


def test_pipeline_bootstrap_matches_european_language_weighting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Core-model scores use the same per-language hierarchy as leaderboards."""
    configs = {
        "language-a": {"sentiment-classification": ["dataset-a"]},
        "language-b": {"sentiment-classification": ["dataset-b"]},
    }
    results = {"model": _model_results("dataset-a", "dataset-b")}
    metadata = {"model": {"parameters": 1.0}}
    model_types = {"model": ModelType.ENCODER}
    original_bootstrap = core_models.bootstrap_rank_scores
    bootstrap = Mock(wraps=original_bootstrap)
    monkeypatch.setattr(core_models, "bootstrap_rank_scores", bootstrap)

    _pareto_categories_per_model(
        model_results=results,
        configs=configs,
        metadata=metadata,
        model_types=model_types,
    )

    all_models_call = next(
        call
        for call in bootstrap.call_args_list
        if call.kwargs["categories"] == (LeaderboardCategory.ALL_MODELS,)
    )
    expected = original_bootstrap(
        model_results=results,
        configs=configs,
        n_bootstraps=core_models.NUM_BOOTSTRAPS,
        seed=0,
        categories=(LeaderboardCategory.ALL_MODELS,),
    )
    actual = original_bootstrap(**all_models_call.kwargs)
    np.testing.assert_array_equal(
        actual["model"][LeaderboardCategory.ALL_MODELS]["language-a"],
        expected["model"][LeaderboardCategory.ALL_MODELS]["language-a"],
    )
    np.testing.assert_array_equal(
        actual["model"][LeaderboardCategory.ALL_MODELS]["language-b"],
        expected["model"][LeaderboardCategory.ALL_MODELS]["language-b"],
    )
    np.testing.assert_array_equal(
        actual["model"][LeaderboardCategory.ALL_MODELS]["overall"],
        expected["model"][LeaderboardCategory.ALL_MODELS]["overall"],
    )
    assert all_models_call.kwargs["configs"] == configs


def test_statistical_ties_remain_on_the_frontier() -> None:
    """Do not remove a model when paired bootstrap samples tie."""
    configs = {"europe": {"sentiment-classification": ["sentiment"]}}
    results = {"a": _model_results("sentiment"), "b": _model_results("sentiment")}
    metadata = {"a": {"parameters": 1.0}, "b": {"parameters": 2.0}}
    model_types = {"a": ModelType.ENCODER, "b": ModelType.ENCODER}
    scores = {
        model: {LeaderboardCategory.ALL_MODELS.value: {"overall": np.array([1.0, 1.0])}}
        for model in results
    }

    pareto = _pareto_categories_per_model(
        bootstrap_scores=scores,
        model_results=results,
        configs=configs,
        metadata=metadata,
        model_types=model_types,
    )

    assert set(pareto) == {"a", "b"}


def test_build_retains_osai_and_api_but_not_eu_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain OSAI/API entries without accepting an EU source input."""
    monkeypatch.setattr(
        "leaderboards.core_models.languages_with_official_datasets", lambda: ["english"]
    )
    monkeypatch.setattr(
        "leaderboards.core_models.official_datasets_for_language",
        lambda language: {"sentiment-classification": ["sentiment"]},
    )
    monkeypatch.setattr("leaderboards.core_models.load_raw_results", lambda: [])
    monkeypatch.setattr(
        "leaderboards.core_models.osai_top_models",
        lambda limit, overrides: [("osai/model", 1)],
    )
    monkeypatch.setattr(
        "leaderboards.core_models.params_from_model_id", lambda model_id: math.nan
    )
    monkeypatch.setattr(
        "leaderboards.core_models.params_from_hf_safetensors", lambda model_id: math.nan
    )

    models = build_core_model_list(api_model_ids=["openai/gpt-5"])
    by_id = {model.model_id: model for model in models}

    assert set(by_id) == {"openai/gpt-5", "osai/model"}
    assert by_id["openai/gpt-5"].api
    assert by_id["osai/model"].osai_rank == 1


def test_core_model_schema_has_aggregate_pareto_categories() -> None:
    """Expose aggregate category data rather than language or EU fields."""
    model = CoreModel(
        model_id="org/model",
        model_type=ModelType.ENCODER,
        size_bucket=SizeBucket.ENCODER,
        parameters=math.nan,
        pareto_categories=(LeaderboardCategory.ALL_MODELS.value,),
        osai_rank=None,
        api=False,
    )

    assert model.pareto_categories == ("all_models",)
    assert not hasattr(model, "eu")
    assert not hasattr(model, "pareto_languages")
