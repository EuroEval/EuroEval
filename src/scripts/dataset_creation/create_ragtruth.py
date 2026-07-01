# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.31.0",
#     "httpx==0.28.0",
#     "python-dotenv==1.1.0",
#     "lettucedetect==1.0.0",
#     "tenacity==9.0.0",
#     "tqdm==4.67.0",
# ]
# ///

"""Download and prepare RAGTruth hallucination dataset for translation.

Downloads source_info.jsonl and response.jsonl from the RAGTruth repository,
joins them on source_id, and saves in HallucinationData format for subsequent
translation.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx
from datasets import Dataset
from dotenv import load_dotenv
from lettucedetect.datasets.hallucination_dataset import (
    HallucinationData,
    HallucinationSample,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("ragtruth")

# Load .env for API credentials
load_dotenv()

# RAGTruth raw data URLs
SOURCE_INFO_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/refs/heads/main/dataset/source_info.jsonl"
RESPONSE_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/refs/heads/main/dataset/response.jsonl"

# Output settings
OUTPUT_DIR = Path("data/ragtruth")
OUTPUT_FILE = OUTPUT_DIR / "ragtruth_data.json"

# HF Hub settings
HUB_REPO_ID = "EuroEval/ragtruth-raw"
PRIVATE_UPLOAD = True


def download_jsonl(url: str) -> list[dict[str, Any]]:
    """Download a JSONL file from a URL.

    Args:
        url: URL to the JSONL file.

    Returns:
        List of parsed JSON objects from each line.
    """
    logger.info(f"Downloading {url}...")
    with httpx.Client() as client:
        response = client.get(url)
        response.raise_for_status()

        lines = response.text.strip().split("\n")
        return [json.loads(line) for line in lines if line.strip()]


def load_ragtruth_data() -> HallucinationData:
    """Download and join RAGTruth dataset from GitHub.

    RAGTruth stores data in two JSONL files:
    - source_info.jsonl: source_id, task_type, source, source_info, prompt
    - response.jsonl: id, source_id, model, labels[], split, response

    These are joined on source_id to create HallucinationSample objects.

    Returns:
        HallucinationData with joined samples.
    """
    # Download both files
    source_info_list = download_jsonl(SOURCE_INFO_URL)
    response_list = download_jsonl(RESPONSE_URL)

    logger.info(f"Downloaded {len(source_info_list)} source records")
    logger.info(f"Downloaded {len(response_list)} response records")

    # Build lookup dict keyed by source_id
    source_lookup: dict[str, dict[str, Any]] = {
        src["source_id"]: src for src in source_info_list
    }

    # Join and transform to HallucinationSample
    samples: list[HallucinationSample] = []
    for resp in response_list:
        source_id = resp.get("source_id")
        if source_id not in source_lookup:
            logger.warning(f"No source info for response {resp.get('id')}")
            continue

        src = source_lookup[source_id]

        # Transform labels: {start, end, label_type} -> {start, end, label}
        labels = [
            {"start": lbl["start"], "end": lbl["end"], "label": lbl["label_type"]}
            for lbl in resp.get("labels", [])
        ]

        samples.append(
            HallucinationSample(
                prompt=src.get("prompt", ""),
                answer=resp.get("response", ""),
                labels=labels,
                split=resp.get("split", "train"),
                task_type=src.get("task_type", "unknown"),
                dataset="ragtruth",
                language="en",
            )
        )

    logger.info(f"Created {len(samples)} samples from RAGTruth dataset")
    return HallucinationData(samples=samples)


def save_to_json(data: HallucinationData, output_path: Path) -> None:
    """Save HallucinationData to JSON file.

    Args:
        data: Data to save.
        output_path: Output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data.to_json(), indent=2))
    logger.info(f"Saved {len(data.samples)} samples to {output_path}")


def push_to_hub(data: HallucinationData, repo_id: str, private: bool = True) -> None:
    """Push raw RAGTruth data to Hugging Face Hub.

    Args:
        data: Data to upload.
        repo_id: Target HF dataset repo ID.
        private: Whether to keep dataset private.
    """
    if not data.samples:
        logger.warning("No samples to upload; skipping Hub upload.")
        return

    rows = [
        {
            "prompt": sample.prompt,
            "answer": sample.answer,
            "labels": sample.labels,
            "split": sample.split,
            "task_type": sample.task_type,
            "dataset": sample.dataset,
            "language": sample.language,
        }
        for sample in data.samples
    ]

    dataset = Dataset.from_list(rows)
    dataset.push_to_hub(repo_id=repo_id, private=private)
    logger.info(f"Pushed {len(rows)} samples to hub: {repo_id}")


def main() -> None:
    """Download RAGTruth data, save locally, and optionally upload to HF Hub."""
    logger.info("Downloading RAGTruth dataset...")

    data = load_ragtruth_data()

    # Save locally
    save_to_json(data, OUTPUT_FILE)

    # Upload to HF Hub if API key is available
    api_key = os.getenv("HF_TOKEN")
    if api_key:
        logger.info("HF_TOKEN found, uploading to Hub...")
        push_to_hub(data, HUB_REPO_ID, PRIVATE_UPLOAD)
    else:
        logger.warning(
            "HF_TOKEN not set. Set it to upload to Hub or run manually:\n"
            f"  hf upload-dataset {OUTPUT_FILE} {HUB_REPO_ID}"
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
