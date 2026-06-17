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

from huggingface_hub import HfApi
from huggingface_hub.errors import HFValidationError
from huggingface_hub.hf_api import RepositoryNotFoundError

from .cache import Cache
from .constants import GENERATIVE_TYPE_KEYWORDS
from .link_generation import generate_anchor_tag, generate_model_url
from .record_fields import get_few_shot, get_task, get_version
from .records import get_model_name

logger = logging.getLogger(__name__)


def add_missing_entries(
    record: dict, trained_from_scratch_patterns: list[re.Pattern], cache: Cache
) -> dict:
    """Adds missing entries to a record.

    For EEE format records (with schema_version), fields are stored in their
    appropriate nested locations. For old format records, fields are added at
    top level.

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
    # Detect EEE format by presence of schema_version
    is_eee_format = "schema_version" in record

    if is_eee_format:
        # EEE format: store fields in appropriate nested locations
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

        # model_url goes in model_info.additional_details
        model_name = get_model_name(record=record)
        model_additional["model_url"] = generate_model_url(model_id=model_name)

        # Top-level precious metadata in EEE format. These values are only
        # copied from existing result sources/backups via the cache; they must
        # not be inferred during leaderboard processing.
        model_id = _model_id_from_record(record=record)
        if "commercially_licensed" not in record:
            if model_id in cache.commercially_licensed:
                record["commercially_licensed"] = cache.commercially_licensed[model_id]
        if "open" not in record:
            if model_id in cache.open:
                record["open"] = cache.open[model_id]
        if "trained_from_scratch" not in record:
            if model_id in cache.trained_from_scratch:
                record["trained_from_scratch"] = cache.trained_from_scratch[model_id]
    else:
        # Old format: store all fields at top level
        if "validation_split" not in record:
            record["validation_split"] = False
        if "few_shot" not in record:
            record["few_shot"] = True
        if "generative" not in record:
            record["generative"] = False
        if "generative_type" not in record:
            record["generative_type"] = get_generative_type(record=record, cache=cache)
        record["merge"] = is_merge(record=record, cache=cache)
        model_id = _model_id_from_record(record=record)
        if "commercially_licensed" not in record:
            if model_id in cache.commercially_licensed:
                record["commercially_licensed"] = cache.commercially_licensed[model_id]
        if "open" not in record:
            if model_id in cache.open:
                record["open"] = cache.open[model_id]
        if "trained_from_scratch" not in record:
            if model_id in cache.trained_from_scratch:
                record["trained_from_scratch"] = cache.trained_from_scratch[model_id]

        # Add model_url field
        model_name = get_model_name(record=record)
        record["model_url"] = generate_model_url(model_id=model_name)

    return record


def fix_metadata(record: dict[str, t.Any], cache: Cache) -> dict[str, t.Any] | None:
    """Fixes metadata in a record.

    Args:
        record:
            A record from the JSONL file.
        cache:
            Metadata cache used to fill in missing model fields.

    Returns:
        The record with fixed metadata, or None if the record should be removed.
    """
    # Copy the record to avoid modifying the original
    record = deepcopy(record)

    # Get task supporting both EEE and old formats
    task = get_task(record)
    if task == "question-answering":
        # Update in appropriate location based on format
        if "eval_library" in record:
            record["eval_library"]["additional_details"]["task"] = (
                "reading-comprehension"
            )
        else:
            record["task"] = "reading-comprehension"
    if task == "european-values":
        # For EEE format, store in eval_library.additional_details
        if "eval_library" in record:
            record["eval_library"]["additional_details"]["validation_split"] = None
            record["eval_library"]["additional_details"]["few_shot"] = None
        else:
            record["validation_split"] = None
            record["few_shot"] = None

    # Handle anchor tag assignment - need to modify the record in place
    model_name = get_model_name(record)
    if model_name in cache.anchor_tag:
        new_model_name = cache.anchor_tag[model_name]
    else:
        anchor_tag = generate_anchor_tag(model_id=model_name)
        if anchor_tag is None:
            return None
        cache.anchor_tag[model_name] = anchor_tag
        new_model_name = anchor_tag

    # Update the record with the anchor tag
    if "model_info" in record:
        record["model_info"]["name"] = new_model_name

        # Extract URL from anchor tag and store it
        url_match = re.search(r"<a href='([^']+)'>", new_model_name)
        if url_match:
            record["model_info"]["additional_details"]["url"] = url_match.group(1)
        elif "url" not in record["model_info"].get("additional_details", {}):
            record["model_info"]["additional_details"]["url"] = None
    else:
        record["model"] = new_model_name

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
    model_id = _model_id_from_record(record=record)

    if "#thinking" in model_id:
        cache.generative_type[model_id] = "reasoning"
        return "reasoning"
    elif "#no-thinking" in model_id:
        cache.generative_type[model_id] = "instruction_tuned"
        return "instruction_tuned"

    # Remove revisions and parameters from the model ID.
    model_id = model_id.split("@")[0].split("#")[0]

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


def is_commercially_licensed(record: dict, cache: Cache) -> bool:
    """Asks if a model is commercially licensed.

    Args:
        record:
            A record from the JSONL file.
        cache:
            The cache.

    Returns:
        Whether the model is commercially licensed.
    """
    model_id = _model_id_from_record(record=record).split("@")[0].split("#")[0]

    # Assume that non-generative models are always commercially licensed
    if not record.get("generative", True):
        cache.commercially_licensed[model_id] = True

    while True:
        if model_id in cache.commercially_licensed:
            return cache.commercially_licensed[model_id]

        msg = f"Is {model_id!r} commercially licensed?"
        if "/" in model_id:
            msg += f" (https://hf.co/{model_id})"
        msg += " [y/n] "
        user_input = input(msg)
        if user_input.lower() in {"y", "yes"}:
            cache.commercially_licensed[model_id] = True
        elif user_input.lower() in {"n", "no"}:
            cache.commercially_licensed[model_id] = False
        else:
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
    model_id = _model_id_from_record(record=record).split("@")[0].split("#")[0]

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
    model_id = _model_id_from_record(record=record).split("@")[0].split("#")[0]

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
    model_id = _model_id_from_record(record=record).split("@")[0].split("#")[0]

    base_model_cache = {_base_model_id(m): value for m, value in cache.open.items()}
    base_model_id = _base_model_id(model_id)
    if base_model_id in base_model_cache:
        value = base_model_cache[base_model_id]
        if model_id not in cache.open:
            cache.open[model_id] = value
        return value

    # Assume closed if not found on HF Hub
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
