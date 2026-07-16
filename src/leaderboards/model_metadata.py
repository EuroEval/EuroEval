"""Model classification and metadata fixing for result records.

These helpers determine model attributes (generative type, licensing,
openness, merge/scratch status) and fill in or repair missing metadata
fields on records before they are processed into leaderboards.
"""

from __future__ import annotations

import logging
import re
import typing as t
import warnings
from copy import deepcopy

import httpx
from huggingface_hub import HfApi
from huggingface_hub.errors import (
    GatedRepoError,
    HFValidationError,
    LocalTokenNotFoundError,
)
from huggingface_hub.hf_api import RepositoryNotFoundError
from requests.exceptions import RequestException

from euroeval.string_utils import split_model_id

from .cache import Cache, _is_hf_url_for_model
from .constants import GENERATIVE_TYPE_KEYWORDS, PERMISSIVE_LICENSES, RESULTS_DIR
from .link_generation import ask_user_to_remove_model, generate_model_url
from .record_fields import get_few_shot, get_task, get_version
from .records import get_bool_field, get_model_name, plain_model_id

logger = logging.getLogger(__name__)


def add_missing_entries(
    record: dict, trained_from_scratch_patterns: list[re.Pattern], cache: Cache
) -> dict:
    """Adds missing entries to a record.

    Fields are stored in their appropriate nested locations within the EEE
    record (``model_info`` and ``eval_library``).

    Args:
        record:
            A record from the JSONL file.
        trained_from_scratch_patterns:
            A list of regex patterns for trained-from-scratch models.
        cache:
            The cache.

    Returns:
        The record with missing entries added.
    """
    model_info = record.setdefault("model_info", {})
    model_additional = model_info.setdefault("additional_details", {})
    eval_lib = record.setdefault("eval_library", {})
    eval_additional = eval_lib.setdefault("additional_details", {})

    if "validation_split" not in eval_additional:
        eval_additional["validation_split"] = False
    if "few_shot" not in eval_additional:
        eval_additional["few_shot"] = True
    if "generative" not in model_additional:
        model_additional["generative"] = False
    if "generative_type" not in model_additional:
        model_additional["generative_type"] = get_generative_type(
            record=record, cache=cache
        )
    if "merge" not in model_additional:
        model_additional["merge"] = is_merge(record=record, cache=cache)

    if "commercially_licensed" not in model_additional:
        model_additional["commercially_licensed"] = is_commercially_licensed(
            record=record, cache=cache
        )
    # Recheck/repair open when missing OR when stale False with HF URL
    if "open" not in model_additional or (
        model_additional["open"] is False
        and "model_url" in model_additional
        and model_additional["model_url"] is not None
        and _is_hf_url_for_model(
            model_additional["model_url"], get_model_name(record=record)
        )
    ):
        model_additional["open"] = is_open(record=record, cache=cache)
    if "trained_from_scratch" not in model_additional:
        model_additional["trained_from_scratch"] = is_trained_from_scratch(
            record=record,
            trained_from_scratch_patterns=trained_from_scratch_patterns,
            cache=cache,
        )
    if "model_url" not in model_additional or model_additional["model_url"] is None:
        model_additional["model_url"] = generate_model_url_with_cache(
            model_id=plain_model_id(get_model_name(record=record)), cache=cache
        )

    return record


def generate_model_url_with_cache(model_id: str, cache: Cache) -> str | None:
    """Generates a model URL using a cache.

    When no URL can be generated, the operator is asked whether to drop the
    model from the results. A "keep" decision is cached so the prompt isn't
    repeated; a "remove" decision deletes the model's result file. Both
    decisions are persisted by ``ask_user_to_remove_model``.

    Args:
        model_id:
            The model ID.
        cache:
            The cache.

    Returns:
        The model URL, or None if no URL could be generated.
    """
    model_id = split_model_id(model_id=plain_model_id(model_id)).model_id
    if model_id in cache.model_url and cache.model_url[model_id] is not None:
        return cache.model_url[model_id]

    model_url = generate_model_url(model_id=model_id)
    if model_url is None and ask_user_to_remove_model(model_id=model_id):
        _remove_model_results(model_id=model_id)
    cache.model_url[model_id] = model_url
    return model_url


def _remove_model_results(model_id: str) -> None:
    """Delete a model's result file from RESULTS_DIR.

    ``RESULTS_DIR`` is the source of truth for the leaderboard, so removing
    the file drops the model from future builds.

    Args:
        model_id:
            The model id whose result file should be removed.
    """
    result_file = RESULTS_DIR / f"{plain_model_id(model_id).replace('/', '_')}.jsonl"
    if result_file.exists():
        result_file.unlink()
        logger.info(f"Removed result file {result_file.name} for {model_id}.")


def fix_metadata(record: dict[str, t.Any]) -> dict[str, t.Any]:
    """Fixes metadata in a record.

    Args:
        record:
            A record from the JSONL file.

    Returns:
        The record with fixed metadata.
    """
    # Copy the record to avoid modifying the original
    record = deepcopy(record)

    task = get_task(record)
    if task == "question-answering":
        record["eval_library"]["additional_details"]["task"] = "reading-comprehension"
    if task == "european-values":
        record["eval_library"]["additional_details"]["validation_split"] = None
        record["eval_library"]["additional_details"]["few_shot"] = None

    return record


def record_is_valid(
    record: dict,
    min_version: str,
    banned_versions: list[str],
    banned_model_patterns: list[re.Pattern],
    api_model_patterns: list[re.Pattern],
) -> bool:
    """Determine if a record is valid.

    Args:
        record:
            The record to validate.
        min_version:
            The minimum EuroEval version to consider.
        banned_versions:
            The EuroEval versions to ban.
        banned_model_patterns:
            The model IDs to ban.
        api_model_patterns:
            Regex patterns identifying models accessed via API.

    Returns:
        True if the record is valid, False otherwise.
    """
    # Remove anchors from model ID, for logging purposes
    inner_anchor_match = re.search(pattern=r">(.+?)<", string=get_model_name(record))
    inner_model_id = (
        inner_anchor_match.group(1) if inner_anchor_match else get_model_name(record)
    )

    # Remove records with disallowed EuroEval versions
    version = get_version(record)
    if version is None or version in banned_versions or version < min_version:
        return False

    # Remove banned models
    if any(
        re.search(pattern=pattern, string=inner_model_id)
        for pattern in banned_model_patterns
    ):
        return False

    # Do not allow few-shot evaluation for API models
    few_shot = get_few_shot(record)
    if (
        any(
            re.fullmatch(pattern=pattern, string=inner_model_id)
            for pattern in api_model_patterns
        )
        and few_shot
    ):
        return False

    # Otherwise, the record is valid
    return True


def get_generative_type(record: dict, cache: Cache) -> str | None:
    """Asks for the generative type of a model.

    Args:
        record:
            A record from the JSONL file.
        cache:
            The cache.

    Returns:
        The generative type of the model.
    """
    raw_model_id = _model_id_from_record(record=record)

    if "#thinking" in raw_model_id:
        cache.generative_type[raw_model_id] = "reasoning"
        return "reasoning"
    elif "#no-thinking" in raw_model_id:
        cache.generative_type[raw_model_id] = "instruction_tuned"
        return "instruction_tuned"

    # Remove revisions and parameters from the model ID, and strip variant suffixes.
    model_id = split_model_id(model_id=plain_model_id(raw_model_id)).model_id

    while True:
        if model_id in cache.generative_type:
            return cache.generative_type[model_id]

        # Pre-fill the generative type from keyword matches in the model id.
        for keywords, gen_type in GENERATIVE_TYPE_KEYWORDS:
            if any(
                re.search(pattern=keyword, string=model_id, flags=re.IGNORECASE)
                for keyword in keywords
            ):
                cache.generative_type[model_id] = gen_type
                return gen_type

        msg = f"What is the generative type of {model_id!r}?"
        if "/" in model_id:
            msg += f" (https://hf.co/{model_id})"
        msg += " [0=null, 1=base, 2=instruction_tuned, 3=reasoning] "
        user_input = input(msg)
        if user_input.lower() in {"0", "null"}:
            cache.generative_type[model_id] = None
        elif user_input.lower() in {"1", "base"}:
            cache.generative_type[model_id] = "base"
        elif user_input.lower() in {"2", "instruction_tuned"}:
            cache.generative_type[model_id] = "instruction_tuned"
        elif user_input.lower() in {"3", "reasoning"}:
            cache.generative_type[model_id] = "reasoning"
        else:
            logger.error("Invalid input. Please try again.")


def _infer_commercial_from_hf_licence(
    model_id: str, licence_cache: dict[str, bool | None]
) -> bool | None:
    """Infer commercial licence status from HF model info.

    Best-effort function that checks the Hugging Face model page for a
    permissive licence tag. Returns ``True`` for permissive licences
    (MIT, Apache-2.0, BSD, etc.), ``None`` for unknown/non-permissive
    licences or on any HF/network/auth error. Never crashes.

    Caches lookups per model_id to avoid repeated API calls.

    Args:
        model_id:
            The Hugging Face model ID (e.g. ``BAAI/bge-m3``).
        licence_cache:
            Cache dict mapping model IDs to inferred licence status.

    Returns:
        ``True`` if licence is permissive, ``False`` if explicitly
        non-permissive (not possible with current logic), or ``None``
        if unknown or on error.
    """
    if model_id in licence_cache:
        return licence_cache[model_id]

    try:
        api = HfApi()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model_info = api.model_info(repo_id=model_id)
    except (
        GatedRepoError,
        LocalTokenNotFoundError,
        RepositoryNotFoundError,
        HFValidationError,
        RequestException,
        OSError,
        httpx.HTTPError,
        httpx.TransportError,
    ):
        licence_cache[model_id] = None
        return None

    # Extract licence from tags (format: "license:apache-2.0")
    licence: str | None = None
    if model_info.tags:
        for tag in model_info.tags:
            if tag.startswith("license:"):
                licence = tag.removeprefix("license:").lower()
                break

    result = licence in PERMISSIVE_LICENSES if licence else None
    licence_cache[model_id] = result
    return result


def is_commercially_licensed(record: dict, cache: Cache) -> bool:
    """Determine if a model is commercially licensed.

    First checks the cache, then tries best-effort inference from the
    Hugging Face model licence. Falls back to user prompt if licence
    cannot be inferred.

    Args:
        record:
            A record from the JSONL file.
        cache:
            The cache.

    Returns:
        Whether the model is commercially licensed.
    """
    model_id = split_model_id(
        model_id=plain_model_id(_model_id_from_record(record=record))
    ).model_id

    # Assume that non-generative models are always commercially licensed
    if not get_bool_field(record, "generative", True):
        cache.commercially_licensed[model_id] = True
        return True

    # Check cache first
    if model_id in cache.commercially_licensed:
        return cache.commercially_licensed[model_id]

    # Best-effort inference from HF licence
    # Use a separate cache dict since inference can return None on error
    licence_cache: dict[str, bool | None] = {}
    inferred = _infer_commercial_from_hf_licence(
        model_id=model_id, licence_cache=licence_cache
    )
    if inferred is not None:
        cache.commercially_licensed[model_id] = inferred
        return inferred

    # Fall back to user prompt
    while True:
        msg = f"Is {model_id!r} commercially licensed?"
        if "/" in model_id:
            msg += f" (https://hf.co/{model_id})"
        msg += " [y/n] "
        user_input = input(msg)
        if user_input.lower() in {"y", "yes"}:
            cache.commercially_licensed[model_id] = True
            return True
        if user_input.lower() in {"n", "no"}:
            cache.commercially_licensed[model_id] = False
            return False
        logger.error("Invalid input. Please try again.")


def is_trained_from_scratch(
    record: dict, trained_from_scratch_patterns: list[re.Pattern], cache: Cache
) -> bool:
    """Determine if a model was trained from scratch or fine-tuned.

    Args:
        record:
            A record from the JSONL file.
        trained_from_scratch_patterns:
            A list of regex patterns for trained-from-scratch models.
        cache:
            The cache.

    Returns:
        True if the model was trained from scratch.
    """
    model_id = split_model_id(
        model_id=plain_model_id(_model_id_from_record(record=record))
    ).model_id

    base_model_cache = {
        _base_model_id(m): value for m, value in cache.trained_from_scratch.items()
    }
    base_model_id = _base_model_id(model_id)
    if base_model_id in base_model_cache:
        value = base_model_cache[base_model_id]
        if model_id not in cache.trained_from_scratch:
            cache.trained_from_scratch[model_id] = value
        return value

    # Check if model is open or closed
    model_openness = cache.open.get(model_id)

    # For closed models, auto-return "scratch" without prompting
    if model_openness is False:
        cache.trained_from_scratch[model_id] = True
        return True

    # If it matches any of the trained-from-scratch patterns, set it automatically
    if any(
        pattern.match(model_id) is not None for pattern in trained_from_scratch_patterns
    ):
        return True

    # For open models, prompt user
    while True:
        msg = f"Was {model_id!r} trained from scratch? "
        if "/" in model_id:
            msg += f" (https://hf.co/{model_id})"
        msg += " [y/n] "
        user_input = input(msg)
        if user_input.lower() in {"y", "yes"}:
            cache.trained_from_scratch[model_id] = True
            return True
        if user_input.lower() in {"n", "no"}:
            cache.trained_from_scratch[model_id] = False
            return False
        logger.error("Invalid input. Please try again.")


def is_merge(record: dict, cache: Cache) -> bool:
    """Determines if a model is a merged model.

    Args:
        record:
            A record from the JSONL file.
        cache:
            The cache.

    Returns:
        Whether the model is a merged model.
    """
    model_id = split_model_id(
        model_id=plain_model_id(_model_id_from_record(record=record))
    ).model_id

    if model_id in cache.merge:
        return cache.merge[model_id]

    # Fresh models do not appear on the model hub, so we assume they are not merge
    # models
    if model_id.startswith("fresh"):
        cache.merge[model_id] = False
        return False

    # Fetch model info from the model hub, and assume that it is not a merged model if
    # the model is not found
    api = HfApi()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model_info = api.model_info(repo_id=model_id)
    except (RepositoryNotFoundError, HFValidationError):
        cache.merge[model_id] = False
        return False

    # A model is a merge model if it has merge-related tags
    merge_tags = ["merge", "mergekit"]
    has_merge_tag = any(tag in (model_info.tags or []) for tag in merge_tags)
    cache.merge[model_id] = has_merge_tag
    return has_merge_tag


def is_open(record: dict, cache: Cache) -> bool:
    """Determine if a model is open (open-weight) or closed.

    Args:
        record:
            A record from the JSONL file.
        cache:
            The cache.

    Returns:
        Whether the model is open (open-weight). Closed models return False.
    """
    model_id = split_model_id(
        model_id=plain_model_id(_model_id_from_record(record=record))
    ).model_id

    # Use exact model-id cache only (no base-model broadening)
    if model_id in cache.open:
        cached_value = cache.open[model_id]
        # Trust cached True; re-check HF for cached False if record has HF URL
        if cached_value is True:
            return True
        # Cached False: check if record has HF URL for this model
        model_url = (
            record.get("model_info", {}).get("additional_details", {}).get("model_url")
        )
        if model_url and _is_hf_url_for_model(model_url, model_id):
            # Re-check HF to repair potentially stale/corrupt False
            try:
                api = HfApi()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    api.model_info(repo_id=model_id)
            except (RepositoryNotFoundError, HFValidationError):
                return False
            # Model exists on HF: update cache and return True
            cache.open[model_id] = True
            return True
        return False

    # Not in cache: check HF
    try:
        api = HfApi()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            api.model_info(repo_id=model_id)
    except (RepositoryNotFoundError, HFValidationError):
        cache.open[model_id] = False
        return False

    # Models found on the HF Hub are open-weight
    cache.open[model_id] = True
    return True


def _model_id_from_record(record: dict) -> str:
    """Return the model id from a record, unwrapping an HTML anchor tag.

    Args:
        record:
            A record from the JSONL file.

    Returns:
        The model id, with any surrounding anchor tag stripped.
    """
    model_id = get_model_name(record)
    if model_id.startswith("<a href="):
        model_id_match = re.search(r">(.+?)<", model_id)
        if model_id_match:
            return model_id_match.group(1)
    return model_id


def _base_model_id(model_id: str) -> str:
    """Return the base-model slug (``org/repo-prefix``) for a model id.

    Args:
        model_id:
            The full model id (e.g. ``org/repo-instruct``).

    Returns:
        The base-model slug (e.g. ``org/repo``), or the id unchanged if it
        has no ``org/repo`` structure.
    """
    if "/" not in model_id:
        return model_id
    parts = model_id.split("/")
    return f"{parts[0]}/{parts[1].split('-')[0]}"
