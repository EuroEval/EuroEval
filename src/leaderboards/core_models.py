"""Compute the list of 'core models' to re-evaluate when datasets change.

The leaderboards grow whenever we add a new model, and re-evaluating every
historical entrant every time a dataset changes is unsustainable. This
module derives a maintained 'core' set: the union of three sources.

  1. Per-language Pareto frontier. For each language leaderboard and each
     model type (encoder, base decoder, instruction-tuned decoder,
     reasoning decoder), a model qualifies if no other model of the same
     type with equal-or-smaller parameter count has a strictly better
     rank score in that language. A model that qualifies in any one
     language is included; the languages it qualifies in are recorded.
  2. EU-built models. Hardcoded regex list in `core_models.yaml`, seeded
     from issue #1186 (orgs like utter-project/, PleIAs/, EuroBERT/,
     LiquidAI/, occiglot/, swiss-ai/, mistralai/, ...).
  3. Top-10 'truly open' models from osai-index.eu (filters: text,
     basemodel weights / training code / data sources all open). The
     site is a Nuxt SPA and exposes the database via a JS bundle; we
     locate that bundle from the homepage, parse the model entries, rank
     them by openness count, and pick the top 10. If the scrape fails we
     fall back to `osai_overrides` in the YAML config.

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

from .constants import (
    API_MODEL_PATTERNS,
    EXCLUDED_MODEL_PATTERNS,
    GENERATIVE_TYPE_TO_MODEL_TYPE,
    NUM_BOOTSTRAPS,
    PARAM_SIZE_BUCKET_ORDER,
)
from .model_sources import eu_models, params_from_hf_safetensors, params_from_model_id
from .osai import osai_top_models
from .records import drop_val_duplicates, get_dataset, plain_model_id
from .result_loading import load_raw_results
from .score_computation import compute_ranks
from .score_extraction import extract_model_metadata, group_results_by_model
from .task_metadata import (
    languages_with_official_datasets,
    official_datasets_for_language,
)

logger = logging.getLogger(__name__)


class ModelType(enum.StrEnum):
    """The architectural / training-stage category of a core model."""

    ENCODER = "encoder"
    BASE_DECODER = "base_decoder"
    INSTRUCTION_TUNED_DECODER = "instruction_tuned_decoder"
    REASONING_DECODER = "reasoning_decoder"
    API = "api"


class SizeBucket(enum.StrEnum):
    """Bucket label used to group models in the GitHub issue."""

    ENCODER = "encoder"
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"
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
        pareto_languages:
            Sorted list of languages in which the model is on the Pareto
            frontier within its model type. Empty if it qualifies only
            via the EU or OSAI source.
        eu:
            Whether the model matches the EU-trained regex list.
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
    pareto_languages: tuple[str, ...]
    eu: bool
    osai_rank: int | None
    api: bool


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_core_model_list(
    eu_patterns: list[str],
    api_model_ids: list[str] | None = None,
    osai_overrides: list[str] | None = None,
    osai_limit: int = 10,
) -> list[CoreModel]:
    """Build the combined core-model list.

    Args:
        eu_patterns:
            Regex patterns for EU-built models (from `core_models.yaml`).
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
    ranks = compute_ranks(
        model_results=model_results, configs=configs, n_bootstraps=NUM_BOOTSTRAPS
    )
    metadata = extract_model_metadata(results=results)

    # Restrict per-language ranking to languages that actually appear as
    # keys in the rank dict (compute_ranks elides single-language scenarios).
    available_languages: set[str] = set()
    for per_category in ranks.values():
        for per_language in per_category.values():
            available_languages.update(per_language.keys())
    available_languages.discard("overall")
    language_list = sorted(available_languages)

    model_types: dict[str, ModelType] = {
        anchored_id: _classify_model(anchored_id, metadata.get(anchored_id, {}))
        for anchored_id in model_results
    }

    pareto = _pareto_languages_per_model(
        ranks=ranks, metadata=metadata, model_types=model_types, languages=language_list
    )

    # Collapse anchored variants ("X (zero-shot)", "X (zero-shot, val)", ...)
    # down to the plain `org/repo` slug. The Pareto languages and metadata
    # for the plain id are the union/best of its variants.
    by_plain: dict[str, list[str]] = defaultdict(list)
    for anchored_id in model_results:
        by_plain[plain_model_id(anchored_id)].append(anchored_id)

    eu_set = eu_models(model_ids=by_plain.keys(), eu_patterns=eu_patterns)
    osai_ranked = osai_top_models(limit=osai_limit, overrides=osai_overrides)
    osai_rank_by_id = {model_id: rank for model_id, rank in osai_ranked}

    # OSAI / EU / API-list may name models we haven't evaluated yet.
    # Include them as placeholders so the issue surfaces them as TODO
    # targets.
    all_plain_ids = set(by_plain) | set(osai_rank_by_id) | eu_set | api_set

    # Drop entire serving-backend families we don't want in the core list.
    all_plain_ids = {
        pid
        for pid in all_plain_ids
        if not any(p.match(pid.split("#")[0]) for p in EXCLUDED_MODEL_PATTERNS)
    }

    core: list[CoreModel] = []
    for plain_id in all_plain_ids:
        variants = by_plain.get(plain_id, [])
        pareto_langs = sorted({lang for v in variants for lang in pareto.get(v, [])})
        is_eu = plain_id in eu_set
        osai_rank = osai_rank_by_id.get(plain_id)
        is_api = plain_id in api_set
        if not (pareto_langs or is_eu or osai_rank or is_api):
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
                pareto_languages=tuple(pareto_langs),
                eu=is_eu,
                osai_rank=osai_rank,
                api=is_api,
            )
        )

    core.sort(
        key=lambda m: (PARAM_SIZE_BUCKET_ORDER[m.size_bucket], m.model_id.lower())
    )
    return core


# ---------------------------------------------------------------------------
# Model classification helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------


def _pareto_languages_per_model(
    ranks: dict[str, dict[str, dict[str, dict[str, float]]]],
    metadata: dict[str, dict],
    model_types: dict[str, ModelType],
    languages: list[str],
) -> dict[str, list[str]]:
    """For each model, return the languages where it is on its Pareto frontier.

    A model M with type T and parameter count P is on the Pareto frontier
    for language L iff no other model with type T and parameters <= P has
    a strictly better (smaller) rank score in L. Models with unknown
    parameter counts are skipped: we can't compare them on the (size, rank)
    plane.

    Args:
        ranks:
            Output of `compute_ranks`: model -> category -> language ->
            {"score", "ci_lower", "ci_upper"}.
        metadata:
            Output of `extract_model_metadata`.
        model_types:
            Mapping of model_id to its `ModelType`.
        languages:
            Languages to consider (each must appear as a key in the inner
            dicts of `ranks`).

    Returns:
        model_id -> sorted list of languages where the model qualifies.
    """
    logger.info("Fetching the Pareto frontier languages for each model...")

    # Encoders only get scored on the NLU-restricted `all_models`
    # category. Generative models live on both leaderboards: `generative`
    # spans every task and `all_models` restricts to NLU — a generative
    # model that's Pareto-optimal in either category counts, so we
    # consider both and union the languages.
    categories_for_type: dict[ModelType, tuple[str, ...]] = {
        ModelType.ENCODER: ("all_models",),
        ModelType.BASE_DECODER: ("generative", "all_models"),
        ModelType.INSTRUCTION_TUNED_DECODER: ("generative", "all_models"),
        ModelType.REASONING_DECODER: ("generative", "all_models"),
    }

    # Group candidate models by type, dropping anything we can't size.
    by_type: dict[ModelType, list[tuple[str, float]]] = defaultdict(list)
    for model_id, model_type in model_types.items():
        if model_type == ModelType.API:
            continue
        params = metadata.get(model_id, {}).get("parameters", float("nan"))
        if not math.isfinite(params):
            continue
        by_type[model_type].append((model_id, params))

    pareto: dict[str, set[str]] = defaultdict(set)
    for model_type, members in by_type.items():
        for category in categories_for_type[model_type]:
            for language in languages:
                # Cache (model_id, params, rank) for this (type, category,
                # language) so the inner check is O(n^2) within the type
                # rather than O(n^2) globally.
                sized_ranked: list[tuple[str, float, float]] = []
                for model_id, params in members:
                    rank_entry = (
                        ranks.get(model_id, {})
                        .get(category, {})
                        .get(language, {})
                        .get("score")
                    )
                    if rank_entry is None or not math.isfinite(rank_entry):
                        continue
                    sized_ranked.append((model_id, params, rank_entry))

                for model_id, params, rank in sized_ranked:
                    dominated = any(
                        other_params <= params and other_rank < rank
                        for other_id, other_params, other_rank in sized_ranked
                        if other_id != model_id
                    )
                    if not dominated:
                        pareto[model_id].add(language)

    logger.info("Fetched the Pareto frontier languages for each model.")
    return {model_id: sorted(langs) for model_id, langs in pareto.items()}
