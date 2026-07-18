"""Sync Hugging Face bucket for EuroEval results.

Uses the ``huggingface_hub`` package to sync the results bucket to the local
results directory. Also provides backup functionality.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from euroeval.data_models import BenchmarkResult

from .constants import HF_RESULTS_BUCKET, RESULTS_DIR
from .evaluation_common import resolve_hf_token

load_dotenv()

logger = logging.getLogger(__name__)


def sync_bucket() -> None:
    """Sync HF results bucket into the local results directory.

    Syncs from bucket to local directory using ``HfApi.sync_bucket``.
    Creates local directory if needed.

    Raises:
        RuntimeError:
            If no Hugging Face token is available.
    """
    hf_token = resolve_hf_token()
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN not set. Cannot sync results from Hugging Face bucket. "
            "Run 'hf auth login' or set the HF_TOKEN environment variable."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Syncing bucket {HF_RESULTS_BUCKET} -> {RESULTS_DIR}...")
    HfApi().sync_bucket(
        source=f"hf://buckets/{HF_RESULTS_BUCKET}/",
        dest=str(RESULTS_DIR),
        token=hf_token,
    )
    logger.info(f"Synced bucket {HF_RESULTS_BUCKET}.")


def upload_results_to_bucket(results_file: Path) -> None:
    """Upload local results to the Hugging Face results bucket.

    Reads the merged results file, splits into per-model JSONL files,
    and syncs to the bucket using hf sync.

    Args:
        results_file:
            Path to the merged results file (euroeval_benchmark_results.jsonl).

    Raises:
        RuntimeError:
            If no Hugging Face token is available.
    """
    hf_token = resolve_hf_token()
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN not set. Cannot upload results to Hugging Face bucket. "
            "Run 'hf auth login' or set the HF_TOKEN environment variable."
        )

    if not results_file.exists():
        logger.warning(
            f"Results file {results_file} does not exist. Nothing to upload."
        )
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_by_model: dict[str, list[str]] = defaultdict(list)
    logger.info(f"Reading results from {results_file}...")
    with results_file.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rec = json.loads(line)
                    result = BenchmarkResult.from_dict(config=rec)
                    if result.model:
                        model_key = _sanitise_model_id(model_id=result.model)
                        results_by_model[model_key].append(line.strip())
                except Exception as e:
                    logger.debug(f"Skipping invalid record during upload: {e}")

    if not results_by_model:
        logger.warning("No valid results found to upload.")
        return

    logger.info(f"Writing {len(results_by_model)} per-model files to {RESULTS_DIR}...")
    for model_key, lines in results_by_model.items():
        model_file = RESULTS_DIR / f"{model_key}.jsonl"
        with model_file.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    logger.info(f"Syncing local {RESULTS_DIR} -> bucket {HF_RESULTS_BUCKET}...")
    HfApi().sync_bucket(
        source=str(RESULTS_DIR),
        dest=f"hf://buckets/{HF_RESULTS_BUCKET}/",
        token=hf_token,
    )
    logger.info(f"Uploaded results to bucket {HF_RESULTS_BUCKET}.")


def _sanitise_model_id(model_id: str) -> str:
    """Convert a model ID to a safe filename.

    Replaces forward slashes with underscores to create valid filenames
    for the bucket structure.

    Args:
        model_id:
            The model identifier (e.g. "meta-llama/Llama-2-7b").

    Returns:
        Safe filename (e.g. "meta-llama_Llama-2-7b").
    """
    return model_id.replace("/", "_")
