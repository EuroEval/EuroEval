# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the ltzGLUE TC dataset and upload to HF Hub."""

import logging

import pandas as pd

if __package__:
    from .dataset_utils import upload_private
else:
    from dataset_utils import upload_private

if __package__:
    from .dataset_processing import build_split_dataset
else:
    from dataset_processing import build_split_dataset

if __package__:
    from .dataset_downloads import download_json_records
else:
    from dataset_downloads import download_json_records

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
BASE_URL = "https://media.githubusercontent.com/media/plumaj/ltzGLUE/main/data/tc"

# Capping limits for mini datasets
MAX_TRAIN = 1024
MAX_VAL = 256
MAX_TEST = 2048


def main() -> None:
    """Create the ltzGLUE-TC dataset."""
    dataset = build_split_dataset(
        load_split=lambda split: _load_split(
            download_json_records(base_url=BASE_URL, split=split)
        ),
        source_splits={"train": "train", "val": "dev", "test": "test"},
        limits={"train": MAX_TRAIN, "val": MAX_VAL, "test": MAX_TEST},
        columns=["text", "label"],
    )

    dataset_id = "EuroEval/ltzglue-tc-mini"
    logger.info(f"Uploading to {dataset_id}...")

    upload_private(dataset=dataset, repository_id=dataset_id)
    logger.info(f"✓ Uploaded {dataset_id}")


def _load_split(data: list[dict]) -> pd.DataFrame:
    """Load topic classification data.

    Args:
        data:
            List of records from the JSON file.

    Returns:
        DataFrame with text and label columns.
    """
    return pd.DataFrame(
        [
            {
                "text": f"{item['title']}: {item['text']}",
                "label": str(item["category_name"]),
            }
            for item in data
        ]
    )


if __name__ == "__main__":
    main()
