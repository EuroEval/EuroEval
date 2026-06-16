"""Process EuroEval records from a JSONL file."""

from __future__ import annotations

import io
import json
import logging
import re
import statistics
import tarfile
import typing as t
import warnings
from collections import Counter, defaultdict
from copy import deepcopy

from huggingface_hub import HfApi
from huggingface_hub.errors import HFValidationError
from huggingface_hub.hf_api import RepositoryNotFoundError
from tqdm.auto import tqdm

from .cache import Cache
from .link_generation import generate_anchor_tag, generate_model_url
from .paths import RESULTS_DIR, RESULTS_PATH
from .result_loading import load_raw_results
from .task_metadata import task_metric_names
from .utils import (
    extract_model_ids_from_record,
    get_record_hash,
    log_once,
    plain_model_id,
)

logger = logging.getLogger(__name__)


def get_model_name(record: dict) -> str:
    """Get model name from record, supporting both EEE and old formats.

    Args:
        record: A result record in either EEE or old EuroEval format.

    Returns:
        The model name.
    """
    if "model_info" in record and "name" in record["model_info"]:
        return record["model_info"]["name"]
    return record.get("model", "unknown")


def get_dataset(record: dict) -> str | None:
    """Get dataset name from record, supporting both EEE and old formats.

    Args:
        record: A result record in either EEE or old EuroEval format.

    Returns:
        The dataset name, or None if not found.
    """
    # EEE format: in eval_library.additional_details or evaluation_results
    if "eval_library" in record:
        additional = record.get("eval_library", {}).get("additional_details", {})
        if "dataset" in additional:
            return additional["dataset"]
        # Also check evaluation_results
        eval_results = record.get("evaluation_results", [])
        if eval_results and isinstance(eval_results, list):
            source_data = eval_results[0].get("source_data", {})
            if "dataset_name" in source_data:
                return source_data["dataset_name"]
    # Old format: top-level dataset field
    return record.get("dataset")


def get_task(record: dict) -> str | None:
    """Get task name from record, supporting both EEE and old formats.

    Args:
        record: A result record in either EEE or old EuroEval format.

    Returns:
        The task name, or None if not found.
    """
    # EEE format: in eval_library.additional_details
    if "eval_library" in record:
        additional = record.get("eval_library", {}).get("additional_details", {})
        if "task" in additional:
            return additional["task"]
    # Old format: top-level task field
    return record.get("task")


def get_raw_results(record: dict) -> list[dict] | dict | None:
    """Get raw results from record, supporting both EEE and old formats.

    Args:
        record: A result record in either EEE or old EuroEval format.

    Returns:
        The raw results (list of dicts for EEE, dict or list for old format).
    """
    # EEE format: JSON string in eval_library.additional_details.raw_results
    if "eval_library" in record:
        additional = record.get("eval_library", {}).get("additional_details", {})
        raw_str = additional.get("raw_results")
        if raw_str and isinstance(raw_str, str):
            try:
                return json.loads(raw_str)
            except json.JSONDecodeError:
                return None
        if raw_str and isinstance(raw_str, list):
            return raw_str
    # Old format: record["results"]["raw"]
    results = record.get("results", {})
    if isinstance(results, dict):
        return results.get("raw")
    return None


def get_total_scores(record: dict) -> dict[str, float] | None:
    """Get total scores from record, supporting both EEE and old formats.

    Args:
        record: A result record in either EEE or old EuroEval format.

    Returns:
        Dict mapping score names to values, or None if not found.
    """
    # EEE format: aggregate from evaluation_results
    if "eval_library" in record and "evaluation_results" in record:
        scores = {}
        eval_results = record.get("evaluation_results", [])
        if isinstance(eval_results, list):
            for er in eval_results:
                if isinstance(er, dict):
                    name = er.get("evaluation_name", "")
                    score_details = er.get("score_details", {})
                    if isinstance(score_details, dict):
                        score = score_details.get("score")
                        if score is not None and isinstance(name, str) and name:
                            # Convert "test_mcc" -> {"test_mcc": 95.0}
                            scores[name] = float(score)
        return scores if scores else None
    # Old format: record["results"]["total"]
    results = record.get("results", {})
    if isinstance(results, dict):
        return results.get("total")
    return None


def process_results(
    min_version: str,
    min_number_of_model_records: int,
    banned_versions: list[str],
    banned_model_patterns: list[re.Pattern],
    api_model_patterns: list[re.Pattern],
    trained_from_scratch_patterns: list[re.Pattern],
) -> None:
    """Process EuroEval records from a JSONL file.

    Args:
        min_version:
            The minimum EuroEval version to include.
        min_number_of_model_records:
            The minimum number of records for a model to be included.
        banned_versions:
            A list of banned EuroEval versions to filter out.
        banned_model_patterns:
            A list of regex patterns to filter out models that should not be included.
        api_model_patterns:
            A list of regex patterns for API inference models.
        trained_from_scratch_patterns:
            A list of regex patterns for trained-from-scratch models.
    """
    results_path = RESULTS_PATH

    # Build the cache from the results directory if available,
    # otherwise fall back to the compressed results file
    cache = Cache.from_processed_records(
        compressed_results_path=results_path, results_dir=RESULTS_DIR
    )

    # Load all the raw records
    records = load_raw_results()
    num_raw_records = len(records)

    # Remove duplicates from the raw records
    all_hash_values = [get_record_hash(record=dct) for dct in records]
    unique_hash_values = sorted(set(all_hash_values))
    new_records = list()
    for unique_hash_value in tqdm(unique_hash_values, desc="Processing records"):
        matches = [
            record
            for record, hash_value in zip(records, all_hash_values)
            if hash_value == unique_hash_value
        ]
        versions = [
            list(
                map(
                    int,
                    re.sub(
                        pattern=r"\.dev[0-9]+",
                        repl="",
                        string=match.get(
                            "euroeval_version", match.get("scandeval_version")
                        )
                        or "0.0.0",
                    ).split("."),
                )
            )
            for match in matches
        ]
        newest_version = max(versions)
        matches_with_newest_version = [
            match
            for match, version in zip(matches, versions)
            if version == newest_version
        ]
        newest_match = matches_with_newest_version[-1]
        new_records.append(newest_match)
    records = new_records
    num_duplicates = num_raw_records - len(records)
    if num_duplicates:
        logger.info(f"Removed {num_duplicates:,} duplicates.")

    # Add missing metadata to records. If the metadata cannot be fixed, the record
    # will be replaced with None, which will be removed later.
    fixed_records: list[dict[str, t.Any] | None] = [
        fix_metadata(record=record, cache=cache)
        for record in tqdm(records, desc="Fixing metadata in records")
    ]

    # Remove regular records which has been removed during processing
    records = [
        record
        for record, fixed_record in zip(records, fixed_records)
        if fixed_record is not None
    ]

    # Remove invalid evaluation records
    processed_records = [
        record
        for record in fixed_records
        if record is not None
        and record_is_valid(
            record=record,
            min_version=min_version,
            banned_versions=banned_versions,
            banned_model_patterns=banned_model_patterns,
            api_model_patterns=api_model_patterns,
        )
    ]

    # Remove records for models with few records
    counter = Counter([get_model_name(record) for record in processed_records])
    processed_records = [
        record
        for record in processed_records
        if counter[get_model_name(record)] >= min_number_of_model_records
    ]

    num_invalid_records = num_raw_records - num_duplicates - len(processed_records)
    if num_invalid_records > 0:
        logger.info(f"Removed {num_invalid_records:,} invalid records.")

    processed_records = [
        add_missing_entries(
            record=record,
            trained_from_scratch_patterns=trained_from_scratch_patterns,
            cache=cache,
        )
        for record in tqdm(processed_records, desc="Adding missing entries")
    ]

    # Group processed records by model for bucket upload
    # Naming convention: replace slashes with underscores, preserve dots
    results_by_model: dict[str, list[dict]] = {}
    for record in processed_records:
        # EEE format has model_info.name, old format has model at top level
        model_id = record.get("model_info", {}).get("name") or record.get(
            "model", "unknown"
        )
        # Strip anchor tags before creating filename
        model_id_str = plain_model_id(model_id)
        filename = model_id_str.replace("/", "_") + ".jsonl"
        results_by_model.setdefault(filename, []).append(record)

    # Upload results to HF bucket as per-model files
    hf_results_bucket = "hf://buckets/EuroEval/results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Uploading results to HF bucket...")
    for filename, records in results_by_model.items():
        file_path = RESULTS_DIR / filename
        content = "\n".join(json.dumps(record) for record in records) + "\n"
        file_path.write_text(content, encoding="utf-8")

    # Sync to bucket using the Python API
    try:
        api = HfApi()
        api.sync_bucket(source=str(RESULTS_DIR), dest=hf_results_bucket)
        logger.info(
            f"Uploaded {len(results_by_model):,} model files to {hf_results_bucket}."
        )
    except Exception as e:
        logger.warning(f"Failed to sync results: {e}")

    # Store all records in the tar.gz archive as a single JSONL file
    # This contains all results (raw + processed with metadata)
    with tarfile.open(results_path, "w:gz") as tar:
        all_content_bytes = "\n".join(
            json.dumps(record) for record in processed_records
        ).encode(encoding="utf-8")
        tarinfo = tarfile.TarInfo(name="results/results.jsonl")
        tarinfo.size = len(all_content_bytes)
        fileobj = io.BytesIO(all_content_bytes)
        tar.addfile(tarinfo=tarinfo, fileobj=fileobj)


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
        # Top-level fields in EEE format
        if "commercially_licensed" not in record:
            record["commercially_licensed"] = is_commercially_licensed(
                record=record, cache=cache
            )
        if "open" not in record:
            record["open"] = is_open(record=record, cache=cache)
        if "trained_from_scratch" not in record:
            record["trained_from_scratch"] = is_trained_from_scratch(
                record=record,
                trained_from_scratch_patterns=trained_from_scratch_patterns,
                cache=cache,
            )
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
        record["commercially_licensed"] = is_commercially_licensed(
            record=record, cache=cache
        )
        record["open"] = is_open(record=record, cache=cache)
        record["trained_from_scratch"] = is_trained_from_scratch(
            record=record,
            trained_from_scratch_patterns=trained_from_scratch_patterns,
            cache=cache,
        )
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
    # If model ID is anchor tag, extract the actual model ID
    model_id = get_model_name(record)
    if get_model_name(record).startswith("<a href="):
        model_id_match = re.search(r">(.+?)<", get_model_name(record))
        if model_id_match:
            model_id = model_id_match.group(1)

    if "#thinking" in model_id:
        cache.generative_type[model_id] = "reasoning"
        return "reasoning"
    elif "#no-thinking" in model_id:
        cache.generative_type[model_id] = "instruction_tuned"
        return "instruction_tuned"

    # Remove extras from model ID
    model_id = model_id.split("@")[0].split("#")[0]

    while True:
        if model_id in cache.generative_type:
            return cache.generative_type[model_id]

        # Pre-fill based on keywords in model name
        null_keywords = ["bert", "xlm-r", "encoder"]
        base_keywords = ["-base", "-pt"]
        instruct_keywords = ["-instruct", "-it$", "-chat"]
        reasoning_keywords = ["^o[1-9]$", "^o[1-9]-", "deepseek-r1"]
        if any(
            re.search(pattern=keyword, string=model_id, flags=re.IGNORECASE)
            for keyword in null_keywords
        ):
            cache.generative_type[model_id] = None
            return None
        if any(
            re.search(pattern=keyword, string=model_id, flags=re.IGNORECASE)
            for keyword in base_keywords
        ):
            cache.generative_type[model_id] = "base"
            return "base"
        if any(
            re.search(pattern=keyword, string=model_id, flags=re.IGNORECASE)
            for keyword in instruct_keywords
        ):
            cache.generative_type[model_id] = "instruction_tuned"
            return "instruction_tuned"
        if any(
            re.search(pattern=keyword, string=model_id, flags=re.IGNORECASE)
            for keyword in reasoning_keywords
        ):
            cache.generative_type[model_id] = "reasoning"
            return "reasoning"

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
    # If model ID is anchor tag, extract the actual model ID
    model_id = get_model_name(record)
    if get_model_name(record).startswith("<a href="):
        model_id_match = re.search(r">(.+?)<", get_model_name(record))
        if model_id_match:
            model_id = model_id_match.group(1)

    # Remove extras from model ID
    model_id = model_id.split("@")[0].split("#")[0]

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
    # If model ID is anchor tag, extract the actual model ID
    model_id = get_model_name(record)
    if get_model_name(record).startswith("<a href="):
        model_id_match = re.search(r">(.+?)<", get_model_name(record))
        if model_id_match:
            model_id = model_id_match.group(1)

    # Remove extras from model ID
    model_id = model_id.split("@")[0].split("#")[0]

    # Check cache first
    base_model_cache = {
        (
            m.split("/")[0] + "/" + m.split("/")[1].split("-")[0] if "/" in m else m
        ): value
        for m, value in cache.trained_from_scratch.items()
    }
    base_model_id = (
        model_id.split("/")[0] + "/" + model_id.split("/")[1].split("-")[0]
        if "/" in model_id
        else model_id
    )
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
    # If model ID is anchor tag, extract the actual model ID
    model_id = get_model_name(record)
    if get_model_name(record).startswith("<a href="):
        model_id_match = re.search(r">(.+?)<", get_model_name(record))
        if model_id_match:
            model_id = model_id_match.group(1)

    # Remove extras from model ID
    model_id = model_id.split("@")[0].split("#")[0]

    # Return cached value if available
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
    # If model ID is anchor tag, extract the actual model ID
    model_id = get_model_name(record)
    if get_model_name(record).startswith("<a href="):
        model_id_match = re.search(r">(.+?)<", get_model_name(record))
        if model_id_match:
            model_id = model_id_match.group(1)

    # Remove revisions and parameters from model ID
    model_id = model_id.split("@")[0].split("#")[0]

    # Check cache first
    base_model_cache = {
        (
            m.split("/")[0] + "/" + m.split("/")[1].split("-")[0] if "/" in m else m
        ): value
        for m, value in cache.open.items()
    }
    base_model_id = (
        model_id.split("/")[0] + "/" + model_id.split("/")[1].split("-")[0]
        if "/" in model_id
        else model_id
    )
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


def _get_version(record: dict) -> str | None:
    """Get version from record, supporting both EEE and old formats.

    Args:
        record:
            A result record in either EEE or old EuroEval format.

    Returns:
        The version string with .dev suffix stripped, or None if not found.
    """
    # EEE format: eval_library.version
    if "eval_library" in record:
        version = record.get("eval_library", {}).get("version")
        # Strip .dev suffix for comparison
        if version:
            return re.sub(r"\.dev\d+", "", version)
    # Old format: euroeval_version
    return record.get("euroeval_version")


def _get_few_shot(record: dict) -> bool:
    """Get few_shot value from record, supporting both EEE and old formats.

    Args:
        record:
            A result record in either EEE or old EuroEval format.

    Returns:
        The few_shot boolean value, defaulting to True.
    """
    # EEE format: eval_library.additional_details.few_shot
    if "eval_library" in record:
        few_shot = (
            record.get("eval_library", {}).get("additional_details", {}).get("few_shot")
        )
        if few_shot is not None:
            if isinstance(few_shot, bool):
                return few_shot
            if isinstance(few_shot, str):
                return few_shot.lower() == "true"
    # Old format: top-level few_shot
    return record.get("few_shot", True)


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
    version = _get_version(record)
    if version is None or version in banned_versions or version < min_version:
        return False

    # Remove banned models
    if any(
        re.search(pattern=pattern, string=inner_model_id)
        for pattern in banned_model_patterns
    ):
        return False

    # Do not allow few-shot evaluation for API models
    few_shot = _get_few_shot(record)
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


def group_results_by_model(
    results: list[dict[str, t.Any]],
) -> dict[str, dict[str, list[tuple[list[float], float, float]]]]:
    """Group results by model ID.

    Args:
        results:
            The processed results.

    Returns:
        The results grouped by model ID. The dict structure is
        model_id -> dataset -> list of (raw_scores, total_score, std_err).
    """
    model_scores: dict[str, dict[str, list[tuple[list[float], float, float]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for record in results:
        model_ids = extract_model_ids_from_record(record=record)
        dataset = get_dataset(record)
        if not dataset:
            continue

        task = get_task(record)
        if not task:
            continue
        primary, secondary = task_metric_names(task)
        metrics = [primary] + ([secondary] if secondary is not None else [])

        for metric_type, metric in zip(("primary", "secondary"), metrics):
            # Get raw results - supports both EEE and old format
            raw_results = get_raw_results(record)
            if raw_results is None:
                continue

            # Extract raw scores for this metric
            raw_scores: list[float] = []
            if isinstance(raw_results, dict) and "test" in raw_results:
                # Old format with test split
                for result_dict in raw_results["test"]:
                    score = result_dict.get(
                        f"test_{metric}", result_dict.get(metric, -1)
                    )
                    if score >= 0:
                        raw_scores.append(score)
            elif isinstance(raw_results, list):
                # EEE format or old format flat list
                for result_dict in raw_results:
                    if isinstance(result_dict, dict):
                        score = result_dict.get(
                            f"test_{metric}", result_dict.get(metric, -1)
                        )
                        if score >= 0:
                            raw_scores.append(score)

            if not raw_scores:
                continue

            # Get total scores - supports both EEE and old format
            total_scores = get_total_scores(record)
            if total_scores is None:
                continue

            # EEE format uses keys like "test_mcc", old format also uses "test_mcc"
            total_score_key = f"test_{metric}"
            std_err_key = f"test_{metric}_se"

            # Try to get total score and std err
            total_score_val = total_scores.get(total_score_key)
            if total_score_val is None:
                # EEE format might not have "test_" prefix
                total_score_val = total_scores.get(metric)

            if total_score_val is None:
                log_once(
                    f"Could not find {metric_type} metric for {dataset!r} "
                    f"in {get_model_name(record)!r} ({total_score_key}). Only found "
                    f"{list(total_scores.keys())}.",
                    logging_level=logging.WARNING,
                )
                continue

            total_score: float = float(total_score_val)

            # Get std_err from old format, or compute from raw_scores for EEE
            std_err: float = total_scores.get(std_err_key, 0.0)
            if std_err == 0.0 and len(raw_scores) > 1:
                # Compute std_err from raw scores (EEE format doesn't have it directly)
                try:
                    std_err = statistics.stdev(raw_scores) / (len(raw_scores) ** 0.5)
                except statistics.StatisticsError:
                    std_err = 0.0

            # Sometimes the raw scores are normalised to [0, 1], so we need to scale
            # them back to [0, 100]
            if max(raw_scores) <= 1:
                raw_scores = [score * 100 for score in raw_scores]

            for model_id in model_ids:
                model_scores[model_id][dataset].append(
                    (raw_scores, total_score, std_err)
                )

    return model_scores


def extract_model_metadata(
    results: list[dict[str, t.Any]],
) -> dict[str, dict[str, t.Any]]:
    """Extract metadata from the results.

    Args:
        results:
            The processed results.

    Returns:
        The metadata.
    """
    logger.info("Extracting model metadata...")
    metadata_dict: dict[str, dict[str, t.Any]] = defaultdict(dict)
    for record in results:
        model_ids = extract_model_ids_from_record(record=record)

        # Support both EEE format (nested in model_info.additional_details) and old
        # format (top-level)
        is_eee = "schema_version" in record
        if is_eee:
            # EEE format
            additional = record.get("model_info", {}).get("additional_details", {})
            num_params_raw = additional.get("num_model_parameters", "-1")
            vocab_size_raw = additional.get("vocabulary_size", "-1")
            context_raw = additional.get("max_sequence_length", "-1")
            merge_raw = additional.get("merge", "false")
            generative_type = additional.get("generative_type", None)
            # Top-level fields in EEE format
            commercially_licensed = record.get("commercially_licensed", False)
            open_weights = record.get("open", None)
            trained_from_scratch = record.get("trained_from_scratch", None)
            model_url = additional.get("model_url", None)
        else:
            # Old format
            num_params_raw = record.get("num_model_parameters", -1)
            vocab_size_raw = record.get("vocabulary_size", -1)
            context_raw = record.get("max_sequence_length", -1)
            merge_raw = record.get("merge", False)
            generative_type = record.get("generative_type", None)
            commercially_licensed = record.get("commercially_licensed", False)
            open_weights = record.get("open", None)
            trained_from_scratch = record.get("trained_from_scratch", None)
            model_url = record.get("model_url", None)

        # Convert to appropriate types
        def _to_float_or_nan(val: str | float | int | None) -> float:
            if isinstance(val, (int, float)):
                return val if val >= 0 else float("nan")
            if isinstance(val, str):
                try:
                    num = float(val)
                    return num if num >= 0 else float("nan")
                except ValueError:
                    return float("nan")
            return float("nan")

        def _to_bool(val: str | bool | None) -> bool:
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() == "true"
            return False

        num_params = _to_float_or_nan(num_params_raw)
        vocab_size = _to_float_or_nan(vocab_size_raw)
        context = _to_float_or_nan(context_raw)
        merge = _to_bool(merge_raw)

        for model_id in model_ids:
            metadata_dict[model_id].update(
                dict(
                    parameters=num_params,
                    vocabulary_size=vocab_size,
                    context=context,
                    generative_type=generative_type,
                    commercial=commercially_licensed,
                    merge=merge,
                    open=open_weights,
                    trained_from_scratch=trained_from_scratch,
                    model_url=model_url,
                )
            )

        # Version column. The frontend hides these and doesn't sort by them,
        # so the plain version string is sufficient.
        version = record.get("euroeval_version", "<9.2.0")
        dataset = get_dataset(record)
        if dataset:
            metadata_dict[model_id][f"{dataset}_version"] = version

    logger.info("Extracted model metadata.")
    return metadata_dict
