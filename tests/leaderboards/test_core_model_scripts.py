"""Tests for core-model serialisation and scheduling."""

from __future__ import annotations

import math

import pytest

from leaderboards.core_models import CoreModel, ModelType, SizeBucket
from scripts import run_core_model_evaluations as runner
from scripts.update_core_models import _format_models_yaml, _reasoning_flags


def _core_model(**overrides: object) -> CoreModel:
    values: dict[str, object] = {
        "model_id": "org/model",
        "model_type": ModelType.ENCODER,
        "size_bucket": SizeBucket.ENCODER,
        "parameters": math.nan,
        "pareto_categories": (),
        "osai_rank": None,
        "api": False,
    }
    values.update(overrides)
    return CoreModel(**values)  # type: ignore[arg-type]


def test_yaml_and_issue_flags_use_aggregate_schema() -> None:
    """Serialisation contains categories and source flags only."""
    model = _core_model(pareto_categories=("all_models",), osai_rank=1, api=True)

    rendered = _format_models_yaml([model])

    assert "pareto_categories: [all_models]" in rendered
    assert "pareto_languages" not in rendered
    assert "eu:" not in rendered
    assert _reasoning_flags(model) == "⭐💜👾"


@pytest.mark.parametrize(
    "model", [_core_model(), _core_model(pareto_categories=("generative",))]
)
def test_core_models_schedule_every_european_language(
    monkeypatch: pytest.MonkeyPatch, model: CoreModel
) -> None:
    """Every core model is scheduled for every official language."""
    monkeypatch.setattr(
        runner,
        "official_dataset_language_pairs",
        lambda: [("dataset-a", "en"), ("dataset-b", "fr"), ("dataset-a", "en")],
    )

    assert runner.codes_for_model(model) == {"en", "fr"}
