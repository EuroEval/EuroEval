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

import collections.abc as c
import dataclasses
import enum
import logging
import math
import re
import urllib.error
import urllib.request
from collections import defaultdict
from functools import cache

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.hf_api import RepositoryNotFoundError

from .result_loading import load_processed_results
from .result_processing import extract_model_metadata, group_results_by_model
from .score_computation import compute_ranks
from .task_metadata import (
    languages_with_official_datasets,
    official_datasets_for_language,
)
from .utils import drop_val_duplicates

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


# Matches the labels used by `GenerativeType` in the euroeval lib (snake_case
# auto-enum). The lib enum has BASE / INSTRUCTION_TUNED / REASONING.
_GENERATIVE_TYPE_TO_MODEL_TYPE: dict[str, ModelType] = {
    "base": ModelType.BASE_DECODER,
    "instruction_tuned": ModelType.INSTRUCTION_TUNED_DECODER,
    "reasoning": ModelType.REASONING_DECODER,
}

# Single source of truth for API model identification — also imported by
# `scripts/generate_leaderboards.py` so the leaderboard pipeline and the
# core-model updater agree on which ids are API models.
API_MODEL_PATTERNS: list[re.Pattern] = [
    re.compile(r"gemini/.*"),
    re.compile(r"(openai/)?gpt-[456789].*"),
    re.compile(r"(anthropic/)?claude.*"),
    re.compile(r"(xai/)?grok.*"),
]

# Models matching these patterns are excluded from the core-model list
# entirely. We currently drop `ollama_chat/*` — those are hard to
# re-evaluate in CI (Ollama-based local serving), so including them in
# the "must re-run when datasets change" list just creates churn.
EXCLUDED_MODEL_PATTERNS: list[re.Pattern] = [
    re.compile(r"^ollama_chat/.*"),
    re.compile(r"^bigscience/bloom.*"),
]


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
# Model classification helpers
# ---------------------------------------------------------------------------


_ANCHOR_RE = re.compile(r"<a [^>]*>(?P<inner>[^<]+)</a>")
# Strips trailing ``(zero-shot)``, ``(val)``, ``(zero-shot, val)`` etc.
# annotations that `extract_model_ids_from_record` appends to variants.
_VARIANT_SUFFIX_RE = re.compile(r"\s*\((?:zero-shot|val)(?:,\s*(?:zero-shot|val))*\)$")


def _plain_model_id(model_id: str) -> str:
    """Strip the HTML anchor and variant-suffix from a result-record model id.

    Records label few-shot vs zero-shot and test vs validation by appending
    ``(zero-shot)`` / ``(val)`` / ``(zero-shot, val)`` to the model id.
    For the core-model list we collapse all those variants down to the
    canonical ``org/repo`` slug — we don't want to list the same model
    several times.

    Args:
        model_id:
            The (possibly anchored, possibly variant-suffixed) identifier.

    Returns:
        The canonical ``org/repo`` slug.
    """
    m = _ANCHOR_RE.search(model_id)
    inner = m.group("inner").strip() if m else model_id
    return _VARIANT_SUFFIX_RE.sub("", inner)


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
    plain = _plain_model_id(model_id).split("#")[0]
    if any(p.fullmatch(plain) for p in API_MODEL_PATTERNS):
        return ModelType.API
    generative_type = metadata.get("generative_type")
    if generative_type is None:
        return ModelType.ENCODER
    return _GENERATIVE_TYPE_TO_MODEL_TYPE.get(generative_type, ModelType.BASE_DECODER)


# Match the first ``<N>B`` / ``<N>M`` token in a model id (e.g. ``22B`` in
# ``EuroLLM-22B-Instruct-2512``, ``270m`` in ``gemma-3-270m``). The lookbehind
# stops us from picking up tokens like ``A3B`` in ``80B-A3B``; the lookahead
# excludes things like ``multilingual``.
_PARAMS_FROM_ID_RE = re.compile(r"(?<![A-Za-z\d])(\d+(?:\.\d+)?)([BbMm])(?![A-Za-z])")


@cache
def _params_from_hf_safetensors(model_id: str) -> float:
    """Best-effort parameter count from a model's safetensors manifest.

    HF's model-info endpoint exposes a ``safetensors.total`` field for
    repos that ship safetensors weights; that's a sum across every
    parameter tensor in the manifest. We use it as a last-resort
    fallback for models where neither the EuroEval metadata nor the id
    itself encodes the size (`yulan-team/YuLan-Mini` and friends).

    Network failures, missing repos, and missing safetensors data all
    degrade to NaN — the caller will then bucket the model as ``xlarge``
    which is the existing behaviour.

    Args:
        model_id:
            HuggingFace ``org/repo`` slug, optionally with an
            ``@revision`` suffix. Anything that doesn't look like
            ``a/b`` is skipped without a network call.

    Returns:
        Total parameter count, or NaN.
    """
    repo_id, _, revision = model_id.partition("@")
    if "/" not in repo_id or repo_id.count("/") > 1:
        return float("nan")
    try:
        info = HfApi().model_info(
            repo_id=repo_id, revision=revision or None, files_metadata=False
        )
    except (RepositoryNotFoundError, HfHubHTTPError, OSError, ValueError) as exc:
        # 404s for IDs we synthesised locally (renamed repos, transient
        # variants, etc.) aren't actionable — log at debug level so the
        # standard run output stays clean.
        logger.debug(f"HF safetensors lookup failed for {model_id!r}: {exc}")
        return float("nan")
    safetensors = getattr(info, "safetensors", None)
    if safetensors is None:
        return float("nan")
    total = getattr(safetensors, "total", None)
    return float(total) if isinstance(total, int) and total > 0 else float("nan")


def _params_from_model_id(model_id: str) -> float:
    """Best-effort parameter count parsed from a model identifier.

    Some entries (typically ones we haven't evaluated yet, or with broken
    HF metadata) reach the core-model list with NaN parameters, which
    would otherwise default them to the ``xlarge`` bucket. When the id
    itself encodes the size (``EuroLLM-22B``, ``Ministral-3-14B``,
    ``SmolLM2-135M``, ...), use that as a fallback.

    Args:
        model_id:
            HuggingFace-style id, optionally with `org/` prefix.

    Returns:
        Parameter count, or NaN if no size token is present.
    """
    m = _PARAMS_FROM_ID_RE.search(model_id)
    if not m:
        return float("nan")
    value = float(m.group(1))
    unit = m.group(2).lower()
    return value * (1_000_000_000 if unit == "b" else 1_000_000)


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
                    try:
                        rank = ranks[model_id][category][language]["score"]
                    except (KeyError, TypeError):
                        continue
                    if not math.isfinite(rank):
                        continue
                    sized_ranked.append((model_id, params, rank))

                for model_id, params, rank in sized_ranked:
                    dominated = any(
                        other_params <= params and other_rank < rank
                        for other_id, other_params, other_rank in sized_ranked
                        if other_id != model_id
                    )
                    if not dominated:
                        pareto[model_id].add(language)

    return {model_id: sorted(langs) for model_id, langs in pareto.items()}


# ---------------------------------------------------------------------------
# EU-trained models
# ---------------------------------------------------------------------------


def eu_models(model_ids: c.Iterable[str], eu_patterns: list[str]) -> set[str]:
    """Return the subset of `model_ids` matched by an EU-org regex.

    Args:
        model_ids:
            Candidate model identifiers (HuggingFace-style).
        eu_patterns:
            Regex patterns from `core_models.yaml::eu_model_patterns`.

    Returns:
        Set of model_ids that match at least one pattern.
    """
    compiled = [re.compile(p) for p in eu_patterns]
    return {
        model_id
        for model_id in model_ids
        if any(p.search(_plain_model_id(model_id).split("#")[0]) for p in compiled)
    }


# ---------------------------------------------------------------------------
# OSAI top-10 truly-open models
# ---------------------------------------------------------------------------

_OSAI_BASE = "https://osai-index.eu"
_OSAI_DB_URL = (
    f"{_OSAI_BASE}/database/"
    "?type=text&weights_basemodel=1&trainingcode=1&datasources_basemodel=1"
)
# Required openness fields (all must be "open"). For weights, see
# _OSAI_WEIGHT_CRITERIA below — at least one of base/end must be open.
_OSAI_REQUIRED_OPEN = ("trainingcode", "datasources_basemodel")
_OSAI_WEIGHT_CRITERIA = ("weights_basemodel", "weights_endmodel")
# All openness fields we count to rank models. Drawn from the rendered
# OSAI methodology page. Order doesn't matter; only the count of "open"s.
_OSAI_RANKING_FIELDS = (
    "weights_basemodel",
    "weights_endmodel",
    "trainingcode",
    "code",
    "datasources_basemodel",
    "datasources_endmodel",
    "datasheet",
    "modelcard",
    "package",
    "license_basemodel",
    "license_endmodel",
    "hardware_architecture",
    "preprint",
    "paper",
)


def _fetch(url: str, timeout: float = 20.0) -> str:
    """Fetch a URL as text.

    Args:
        url:
            The URL to fetch.
        timeout (optional):
            Socket timeout in seconds. Defaults to 20.0.

    Returns:
        The response body as text.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "EuroEval-CoreModels/1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


_OSAI_BUNDLE_MARKER = "system:{name:"


def _osai_bundle() -> str | None:
    """Discover the current Nuxt JS bundle that contains the model database.

    OSAI ships a Nuxt SPA. The model entries are bundled into a single
    `/_nuxt/*.js` chunk whose filename rotates on every deploy. We
    enumerate the chunks referenced from the homepage, probe them in
    largest-first order, and return the first that contains the
    distinctive `system:{name:` marker that begins each model entry.

    Returns:
        Raw JS source of the model-data chunk, or None if discovery failed.
    """
    try:
        html = _fetch(_OSAI_BASE)
    except (urllib.error.URLError, OSError) as exc:
        logger.warning(f"OSAI: cannot reach homepage: {exc}")
        return None
    chunks = re.findall(r'href="(/_nuxt/[^"]+\.js)"', html)
    if not chunks:
        logger.warning("OSAI: no /_nuxt/*.js chunks found on homepage.")
        return None
    sized: list[tuple[int, str]] = []
    for chunk in set(chunks):
        url = _OSAI_BASE + chunk
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as resp:
                length = int(resp.headers.get("Content-Length") or 0)
        except (urllib.error.URLError, OSError, ValueError):
            continue
        sized.append((length, url))
    sized.sort(reverse=True)
    # Skip the obviously-too-small chunks; the model bundle is hundreds
    # of KB. Of the remaining candidates, return the first one whose
    # contents contain the model-entry marker.
    for length, url in sized:
        if length < 100_000:
            break
        try:
            body = _fetch(url, timeout=30.0)
        except (urllib.error.URLError, OSError):
            continue
        if _OSAI_BUNDLE_MARKER in body:
            logger.info(f"OSAI: model data found in {url}")
            return body
    logger.warning("OSAI: no chunk contained the model-entry marker.")
    return None


# Matches the start of each model entry in the bundle. Each model is a JS
# object literal opening with `system:{name:"...",link:"...",type:"...",...,
# endmodelname:"...",...}`. The openness-criteria fields follow that block.
_OSAI_SYSTEM_RE = re.compile(
    r"system:\{"
    r"name:\"(?P<name>[^\"]+)\","
    r"link:\"(?P<link>[^\"]*)\","
    r"type:\"(?P<type>[^\"]+)\","
    r"performanceclass:\"[^\"]+\","
    r"basemodelname:\"[^\"]+\","
    r"endmodelname:\"(?P<endmodelname>[^\"]+)\""
)
# Matches one openness criterion declaration: <field>:{class:"<cls>"...
_OSAI_FIELD_RE = re.compile(r"(\w+):\{class:\"(open|closed|partial)\"")
# Within an openness block, the link can be a single string or a JSON array
# of strings. We extract the first HF URL we find (string-form).
_OSAI_BLOCK_LINK_RE = re.compile(
    r"link:(?:\"(?P<single>[^\"]*)\"|\[\"(?P<first>[^\"]*)\")"
)
# Matches the whole weights_endmodel/basemodel block so we can scan inside
# it for an HF link.
_OSAI_WEIGHTS_BLOCK_RE = re.compile(
    r"(weights_endmodel|weights_basemodel):\{(?P<body>[^}]*)\}"
)


def _parse_osai_models(bundle: str) -> list[dict]:
    """Parse model entries out of an OSAI JS bundle.

    Each model entry begins with a `system:{...}` block and is followed by
    openness-criteria fields. We slice the bundle on `system:{name:` to
    delimit entries, parse each slice, and stop at the next `system:{`
    (or end of bundle).

    Args:
        bundle:
            Raw JavaScript source of the Nuxt chunk that holds the model
            database.

    Returns:
        List of dicts, one per model, with keys: name, endmodelname,
        type, open_count, required_open (bool — trainingcode and
        datasources_basemodel both open), weight_links (dict mapping
        the open weights field name to its HF URL).
    """
    matches = list(_OSAI_SYSTEM_RE.finditer(bundle))
    entries: list[dict] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(bundle)
        slice_ = bundle[start:end]

        classes = dict(_OSAI_FIELD_RE.findall(slice_))
        open_count = sum(
            1 for field in _OSAI_RANKING_FIELDS if classes.get(field) == "open"
        )
        required_open = all(
            classes.get(field) == "open" for field in _OSAI_REQUIRED_OPEN
        )

        # Collect an HF URL for each weights field that is open. If both
        # base and end weights are open we'll emit both downstream so the
        # eval covers them individually.
        weight_links: dict[str, str] = {}
        for block in _OSAI_WEIGHTS_BLOCK_RE.finditer(slice_):
            field = block.group(1)
            if classes.get(field) != "open":
                continue
            link_match = _OSAI_BLOCK_LINK_RE.search(block.group("body"))
            if not link_match:
                continue
            link = link_match.group("single") or link_match.group("first") or ""
            if "huggingface.co/" in link:
                weight_links[field] = link

        entries.append(
            dict(
                name=m.group("name"),
                endmodelname=m.group("endmodelname"),
                type=m.group("type"),
                open_count=open_count,
                required_open=required_open,
                weight_links=weight_links,
            )
        )
    return entries


def _hf_url_to_model_id(url: str) -> str | None:
    """Convert an HF URL to a `org/repo` model identifier.

    Args:
        url:
            A URL like ``https://huggingface.co/org/repo`` (with optional
            trailing path, blob link, etc.).

    Returns:
        The ``org/repo`` slug, or None if the URL doesn't match that shape.
    """
    m = re.match(r"https://huggingface\.co/([^/]+)/([^/?#]+)", url)
    if not m:
        return None
    org, repo = m.group(1), m.group(2)
    if org in {"datasets", "spaces", "docs", "blog", "collections"}:
        return None
    return f"{org}/{repo}"


def osai_top_models(
    limit: int = 10, overrides: list[str] | None = None
) -> list[tuple[str, int]]:
    """Return the top OSAI 'truly open' text models as (model_id, rank) pairs.

    The OSAI index is filtered to text models with open base weights,
    training code, and data sources. Models are ranked by total open-field
    count (descending). We scrape the live JS bundle; on any failure (no
    bundle reachable, regex match yields zero entries, etc.) we return
    the override list from the YAML config so the pipeline stays
    deterministic.

    Args:
        limit (optional):
            Number of top models to return. Defaults to 10.
        overrides (optional):
            Fallback list of `org/repo` model IDs in rank order, used
            verbatim when the scrape fails. Defaults to None.

    Returns:
        List of `(model_id, rank)` pairs in 1-based rank order. Empty
        list if both the scrape and overrides yield nothing.
    """
    bundle = _osai_bundle()
    if bundle is None:
        logger.warning("OSAI: bundle not found; using overrides.")
        return _overrides_to_ranked(overrides, limit)

    entries = _parse_osai_models(bundle)
    if not entries:
        logger.warning("OSAI: parsed zero model entries; using overrides.")
        return _overrides_to_ranked(overrides, limit)

    # A model qualifies if training code and base-model data sources are
    # open and at least one of base/end-model weights is open. When both
    # weights are open, both are added to the eval target list — they're
    # different checkpoints worth scoring independently.
    qualifying = [
        e
        for e in entries
        if e["type"] == "text" and e["required_open"] and e["weight_links"]
    ]
    qualifying.sort(key=lambda e: (-e["open_count"], e["endmodelname"].lower()))

    seen: set[str] = set()
    ranked: list[tuple[str, int]] = []
    for entry in qualifying:
        # Stable ordering: prefer the base-model checkpoint first, then end.
        for field in _OSAI_WEIGHT_CRITERIA:
            link = entry["weight_links"].get(field)
            if not link:
                continue
            model_id = _hf_url_to_model_id(link)
            if model_id is None:
                logger.info(
                    f"OSAI: skipping {entry['endmodelname']!r} ({field}): "
                    f"cannot map {link!r} to org/repo."
                )
                continue
            if model_id in seen:
                continue
            seen.add(model_id)
            ranked.append((model_id, len(ranked) + 1))
            if len(ranked) >= limit:
                return ranked
        if not entry["weight_links"]:
            logger.info(
                f"OSAI: skipping {entry['endmodelname']!r}: no open weights link."
            )
    return ranked


def _overrides_to_ranked(
    overrides: list[str] | None, limit: int
) -> list[tuple[str, int]]:
    """Convert a flat override list into ranked tuples.

    Args:
        overrides:
            Manually-curated list of model IDs in rank order.
        limit:
            Max length of the returned list.

    Returns:
        `[(model_id, rank), ...]` up to `limit` entries.
    """
    if not overrides:
        return []
    return [(model_id, rank) for rank, model_id in enumerate(overrides[:limit], 1)]


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
            `core_models.yaml::api_models`. Always emitted with the 👾
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

    results = [r for r in load_processed_results() if r["dataset"] in datasets]
    model_results = group_results_by_model(results=results)
    model_results = drop_val_duplicates(model_results=model_results)
    ranks = compute_ranks(model_results=model_results, configs=configs)
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
        by_plain[_plain_model_id(anchored_id)].append(anchored_id)

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
            parameters = _params_from_model_id(plain_id)
        # API models don't live on HuggingFace, so hitting the HF
        # safetensors endpoint for them just guarantees a 404.
        if not math.isfinite(parameters) and model_type != ModelType.API:
            parameters = _params_from_hf_safetensors(plain_id.split("#")[0])
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

    core.sort(key=lambda m: (_BUCKET_ORDER[m.size_bucket], m.model_id.lower()))
    return core


_BUCKET_ORDER: dict[SizeBucket, int] = {
    SizeBucket.ENCODER: 0,
    SizeBucket.TINY: 1,
    SizeBucket.SMALL: 2,
    SizeBucket.MEDIUM: 3,
    SizeBucket.LARGE: 4,
    SizeBucket.XLARGE: 5,
    SizeBucket.API: 6,
}


__all__ = [
    "CoreModel",
    "ModelType",
    "SizeBucket",
    "build_core_model_list",
    "eu_models",
    "osai_top_models",
]
