# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the ltzGLUE RTE dataset and upload to HF Hub."""

import io
import logging

import pandas as pd
import requests

if __package__:
    from .dataset_utils import upload_private
else:
    from dataset_utils import upload_private

if __package__:
    from .dataset_processing import build_split_dataset
else:
    from dataset_processing import build_split_dataset

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
BASE_URL = "https://raw.githubusercontent.com/plumaj/ltzGLUE/main/data/rte"

# Capping limits for mini datasets
MAX_TRAIN = 1024
MAX_VAL = 256
MAX_TEST = 2048


def main() -> None:
    """Create the ltzGLUE-RTE dataset and upload to HF Hub."""
    dataset = build_split_dataset(
        load_split=_download_split,
        source_splits={"train": "train", "val": "dev", "test": "test"},
        limits={"train": MAX_TRAIN, "val": MAX_VAL, "test": MAX_TEST},
        columns=["text", "label"],
    )

    dataset_id = "EuroEval/ltzglue-rte-mini"
    logger.info(f"Uploading to {dataset_id}...")

    upload_private(dataset=dataset, repository_id=dataset_id)
    logger.info(f"✓ Uploaded {dataset_id}")


def _download_split(split: str) -> pd.DataFrame:
    """Download a single split from GitHub.

    Args:
        split:
            Split name (train, dev, test).

    Returns:
        DataFrame with premise, hypothesis, and label columns.
    """
    url = f"{BASE_URL}/{split}.tsv"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text), sep="\t")
    label_map = {0: "contradiction", 1: "entailment"}
    df["label"] = df["label"].map(label_map)
    df = df.rename(columns={"sentence1": "premise", "sentence2": "hypothesis"})
    return df[["premise", "hypothesis", "label"]]


if __name__ == "__main__":
    main()
