"""Compute the list of 'core models' to re-evaluate when datasets change.

The leaderboards grow whenever we add a new model, and re-evaluating every
historical entrant every time a dataset changes is unsustainable. This
module derives a maintained 'core' set from aggregate Pareto, OSAI and API sources.

  1. Complete-coverage aggregate Pareto frontier. For each European
     leaderboard category and model type, a model qualifies only when it has
     every applicable non-orthogonal dataset. An equal-or-smaller model of
     the same type removes it only when the aligned paired-bootstrap score
     difference is significant.
  2. Top-10 'truly open' models from osai-index.eu (filters: text,
     basemodel weights / training code / data sources all open). The site is
     a Nuxt SPA and exposes the database via a JS bundle; we locate that
     bundle from the homepage, parse the model entries, rank them by
     openness count, and pick the top 10. If the scrape fails we fall back
     to `osai_overrides` in the YAML config.

`build_core_model_list` is the public entry point. It returns a list of
`CoreModel` records; the updater script renders them into the GitHub
issue and writes the same list back into `core_models.yaml` (alongside
`last_updated`).
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import math
from collections import defaultdict

import numpy as np

from euroeval.constants import ORTHOGONAL_TASKS

from .bootstrap_cis import bootstrap_rank_scores
from .constants import (
    API_MODEL_PATTERNS,
    EXCLUDED_MODEL_PATTERNS,
    GENERATIVE_TYPE_TO_MODEL_TYPE,
    NUM_BOOTSTRAPS,
    PARAM_SIZE_BUCKET_ORDER,
)
from .enums import LeaderboardCategory
from .model_sources import params_from_hf_safetensors, params_from_model_id
from .osai import osai_top_models
from .records import drop_val_duplicates, get_dataset, plain_model_id
from .result_loading import load_raw_results
from .score_extraction import extract_model_metadata, group_results_by_model
from .task_metadata import (
    category_includes_task,
    languages_with_official_datasets,
    official_datasets_for_language,
)

logger = logging.getLogger(__name__)


class SizeBucket(enum.StrEnum):
    """Bucket label used to group models in the GitHub issue."""

    ENCODER = "encoder"
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"
    API = "api"


class ModelType(enum.StrEnum):
    """The architectural / training-stage category of a core model."""

    ENCODER = "encoder"
    BASE_DECODER = "base_decoder"
    INSTRUCTION_TUNED_DECODER = "instruction_tuned_decoder"
    REASONING_DECODER = "reasoning_decoder"
    API = "api"


@dataclasses.dataclass(frozen=True)
class CoreModel:
    """A model that should be re-evaluated when datasets change.

    Attributes:
        model_id:
            The HuggingFace-style model identifier.
        model_type:
            Which architectural/training category the model belongs to.
        size_bucket:
            The size bucket used for grouping in the GitHub issue.
        parameters:
            Number of parameters (NaN for API models / unknown).
        pareto_categories:
            Sorted leaderboard categories in which the model is on the
            aggregate Pareto frontier. Empty if it qualifies only via OSAI.
        osai_rank:
            1-based rank in the OSAI top-10 list, or None if not in the list.
        api:
            Whether the model is in the hardcoded litellm API list from
            `core_models.yaml::api_models`. Always evaluated on every
            language.
    """

    model_id: str
    model_type: ModelType
    size_bucket: SizeBucket
    parameters: float
    pareto_categories: tuple[str, ...]
    osai_rank: int | None
    api: bool


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_core_model_list(
    api_model_ids: list[str] | None = None,
    osai_overrides: list[str] | None = None,
    osai_limit: int = 10,
) -> list[CoreModel]:
    """Build the combined core-model list.

    Args:
        api_model_ids (optional):
            Hardcoded list of litellm-style API model identifiers from
            `core_models.yaml::api_models`. Always emitted with the API
            flag and "All languages". Defaults to None.
        osai_overrides (optional):
            Override list used when the OSAI scrape fails. Defaults to None.
        osai_limit (optional):
            How many OSAI top models to keep. Defaults to 10.

    Returns:
        Sorted list of `CoreModel` records.
    """
    api_set = set(api_model_ids or [])
    languages = languages_with_official_datasets()
    configs: dict[str, dict[str, list[str]]] = {
        language: dict(official_datasets_for_language(language))
        for language in languages
    }
    datasets = {
        dataset
        for config in configs.values()
        for task_datasets in config.values()
        for dataset in task_datasets
    }

    results = [r for r in load_raw_results() if get_dataset(r) in datasets]
    model_results = group_results_by_model(results=results)
    model_results = drop_val_duplicates(model_results=model_results)
    metadata = extract_model_metadata(results=results)

    model_types: dict[str, ModelType] = {
        anchored_id: _classify_model(anchored_id, metadata.get(anchored_id, {}))
        for anchored_id in model_results
    }

    pareto = _pareto_categories_per_model(
        model_results=model_results,
        configs=configs,
        metadata=metadata,
        model_types=model_types,
    )

    # Collapse anchored variants ("X (zero-shot)", "X (zero-shot, val)", ...)
    # down to the plain `org/repo` slug. Pareto categories for the plain id
    # are the union of its variants.
    by_plain: dict[str, list[str]] = defaultdict(list)
    for anchored_id in model_results:
        by_plain[plain_model_id(anchored_id)].append(anchored_id)

    osai_ranked = osai_top_models(limit=osai_limit, overrides=osai_overrides)
    osai_rank_by_id = {model_id: rank for model_id, rank in osai_ranked}

    # OSAI / API-list may name models we haven't evaluated yet. Include
    # them as placeholders so the issue surfaces them as TODO
    # targets.
    all_plain_ids = set(by_plain) | set(osai_rank_by_id) | api_set

    # Drop entire serving-backend families we don't want in the core list.
    all_plain_ids = {
        pid
        for pid in all_plain_ids
        if not any(p.match(pid.split("#")[0]) for p in EXCLUDED_MODEL_PATTERNS)
    }

    core: list[CoreModel] = []
    for plain_id in all_plain_ids:
        variants = by_plain.get(plain_id, [])
        pareto_categories = sorted(
            {category for v in variants for category in pareto.get(v, [])}
        )
        osai_rank = osai_rank_by_id.get(plain_id)
        is_api = plain_id in api_set
        if not (pareto_categories or osai_rank or is_api):
            continue

        # Pick the variant with the most params info / a known type. The
        # base anchored_id (without zero-shot suffix) typically sorts first.
        rep = sorted(variants)[0] if variants else plain_id
        meta = metadata.get(rep, {})
        model_type = model_types.get(rep) or _classify_model(plain_id, meta)
        parameters = meta.get("parameters", float("nan"))
        if not math.isfinite(parameters):
            parameters = params_from_model_id(model_id=plain_id)

        # API models don't live on HuggingFace, so hitting the HF
        # safetensors endpoint for them just guarantees a 404.
        if not math.isfinite(parameters) and model_type != ModelType.API:
            parameters = params_from_hf_safetensors(model_id=plain_id.split("#")[0])
        bucket = _size_bucket(model_type, parameters)
        core.append(
            CoreModel(
                model_id=plain_id,
                model_type=model_type,
                size_bucket=bucket,
                parameters=parameters,
                pareto_categories=tuple(pareto_categories),
                osai_rank=osai_rank,
                api=is_api,
            )
        )

    core.sort(
        key=lambda m: (PARAM_SIZE_BUCKET_ORDER[m.size_bucket], m.model_id.lower())
    )
    return core


def _classify_model(model_id: str, metadata: dict) -> ModelType:
    """Return the architectural/training type for a model.

    Args:
        model_id:
            The HuggingFace-style model identifier.
        metadata:
            The metadata entry from `extract_model_metadata`.

    Returns:
        One of the `ModelType` literals.
    """
    plain = plain_model_id(model_id).split("#")[0]
    if any(p.fullmatch(plain) for p in API_MODEL_PATTERNS):
        return ModelType.API
    generative_type = metadata.get("generative_type")
    if generative_type is None:
        return ModelType.ENCODER
    model_type = GENERATIVE_TYPE_TO_MODEL_TYPE.get(generative_type)
    return ModelType(model_type) if model_type is not None else ModelType.BASE_DECODER


def _aggregate_bootstrap_scores(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    configs: dict[str, dict[str, list[str]]],
    category: LeaderboardCategory,
) -> dict[str, dict[str | LeaderboardCategory, dict[str, np.ndarray]]]:
    """Compute aligned bootstrap score distributions for one category.

    Args:
        model_results:
            Complete-coverage model results for ``category``.
        configs:
            The original per-language leaderboard configurations.
        category:
            The single category to rank.

    Returns:
        Model/category/aggregate score distributions with aligned samples.
    """
    return bootstrap_rank_scores(
        model_results=model_results,
        configs=configs,
        n_bootstraps=NUM_BOOTSTRAPS,
        seed=0,
        categories=(category,),
    )


def _pareto_categories_per_model(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    configs: dict[str, dict[str, list[str]]],
    metadata: dict[str, dict],
    model_types: dict[str, ModelType],
    bootstrap_scores: dict[str, dict[str | LeaderboardCategory, dict[str, np.ndarray]]]
    | None = None,
) -> dict[str, set[str]]:
    """Return aggregate Pareto categories for completely evaluated models.

    A model is eligible in a category only when it has a result for every
    applicable non-orthogonal dataset. Decoder types are eligible in either
    aggregate category; encoders are eligible only in ``all_models``. When
    scores are not supplied, each category is bootstrapped independently from
    its complete-coverage model set using the original per-language configs.

    Args:
        model_results:
            Results grouped by model and dataset.
        configs:
            The original per-language leaderboard configurations.
        metadata:
            Model metadata, including parameter counts.
        model_types:
            Architectural type for each model.
        bootstrap_scores (optional):
            Precomputed scores, retained for focused callers that provide
            deterministic distributions. Defaults to None.
    """
    if bootstrap_scores is None:
        bootstrap_scores = {}
        for category in (
            LeaderboardCategory.GENERATIVE,
            LeaderboardCategory.ALL_MODELS,
        ):
            eligible_results = _complete_coverage_model_results(
                model_results=model_results, configs=configs, category=category
            )
            category_scores = _aggregate_bootstrap_scores(
                model_results=eligible_results, configs=configs, category=category
            )
            for model_id, model_scores in category_scores.items():
                bootstrap_scores.setdefault(model_id, {}).update(model_scores)
    categories_for_type: dict[ModelType, tuple[LeaderboardCategory, ...]] = {
        ModelType.ENCODER: (LeaderboardCategory.ALL_MODELS,),
        ModelType.BASE_DECODER: (
            LeaderboardCategory.GENERATIVE,
            LeaderboardCategory.ALL_MODELS,
        ),
        ModelType.INSTRUCTION_TUNED_DECODER: (
            LeaderboardCategory.GENERATIVE,
            LeaderboardCategory.ALL_MODELS,
        ),
        ModelType.REASONING_DECODER: (
            LeaderboardCategory.GENERATIVE,
            LeaderboardCategory.ALL_MODELS,
        ),
    }
    required = {
        category: _required_datasets(configs=configs, category=category)
        for category in {
            category
            for categories in categories_for_type.values()
            for category in categories
        }
    }
    eligible: dict[tuple[LeaderboardCategory, ModelType], list[tuple[str, float]]] = (
        defaultdict(list)
    )
    for model_id, model_type in model_types.items():
        if model_type not in categories_for_type:
            continue
        params = metadata.get(model_id, {}).get("parameters", float("nan"))
        if not math.isfinite(params):
            continue
        for category in categories_for_type[model_type]:
            if not all(
                model_results.get(model_id, {}).get(dataset)
                for dataset in required[category]
            ):
                continue
            distribution = (
                bootstrap_scores.get(model_id, {}).get(category, {}).get("overall")
            )
            if distribution is not None:
                eligible[(category, model_type)].append((model_id, params))

    pareto: dict[str, set[str]] = defaultdict(set)
    for (category, _model_type), members in eligible.items():
        for model_id, params in members:
            distribution = bootstrap_scores[model_id][category]["overall"]
            dominated = any(
                other_id != model_id
                and other_params <= params
                and _is_significantly_worse(
                    candidate=distribution,
                    competitor=bootstrap_scores[other_id][category]["overall"],
                    alpha=0.05,
                )
                for other_id, other_params in members
            )
            if not dominated:
                pareto[model_id].add(category.value)
    return pareto


def _required_datasets(
    configs: dict[str, dict[str, list[str]]], category: LeaderboardCategory
) -> set[str]:
    """Return all non-orthogonal datasets applicable to a category."""
    return {
        dataset
        for config in configs.values()
        for task, datasets in config.items()
        if task not in ORTHOGONAL_TASKS
        and category_includes_task(category=category, task=task)
        for dataset in datasets
    }


def _complete_coverage_model_results(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    configs: dict[str, dict[str, list[str]]],
    category: LeaderboardCategory,
) -> dict[str, dict[str, list[tuple[list[float], float, float]]]]:
    """Return models with every non-orthogonal dataset for a category."""
    required_datasets = _required_datasets(configs=configs, category=category)
    return {
        model_id: model_results[model_id]
        for model_id in sorted(model_results)
        if all(dataset in model_results[model_id] for dataset in required_datasets)
    }


def _is_significantly_worse(
    candidate: np.ndarray, competitor: np.ndarray, alpha: float = 0.05
) -> bool:
    """Return whether candidate loses to competitor in a paired bootstrap.

    Rank scores are minimised. Thus a positive lower percentile of
    ``candidate - competitor`` means the candidate is significantly worse.
    """
    candidate_array = np.asarray(candidate)
    competitor_array = np.asarray(competitor)
    if candidate_array.shape != competitor_array.shape:
        return False
    difference = candidate_array - competitor_array
    return bool(np.percentile(difference, 100 * alpha / 2) > 0)


# ---------------------------------------------------------------------------
# Model classification helpers
# ---------------------------------------------------------------------------
def _size_bucket(model_type: ModelType, parameters: float) -> SizeBucket:
    """Map a model's type and parameter count to a bucket for the issue.

    Args:
        model_type:
            The classification from `_classify_model`.
        parameters:
            Number of parameters; NaN for API/unknown.

    Returns:
        The bucket label used to group models in the issue body.
    """
    if model_type == ModelType.ENCODER:
        return SizeBucket.ENCODER
    if model_type == ModelType.API:
        return SizeBucket.API
    if not math.isfinite(parameters):
        return SizeBucket.XLARGE
    if parameters < 2_000_000_000:
        return SizeBucket.TINY
    if parameters < 10_000_000_000:
        return SizeBucket.SMALL
    if parameters < 40_000_000_000:
        return SizeBucket.MEDIUM
    if parameters < 80_000_000_000:
        return SizeBucket.LARGE
    return SizeBucket.XLARGE
