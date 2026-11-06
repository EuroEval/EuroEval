# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the ltzGLUE Headline Classification dataset and upload to HF Hub."""

import logging

import pandas as pd

if __package__:
    from .dataset_utils import preflight_private_upload
else:
    from dataset_utils import preflight_private_upload

if __package__:
    from .dataset_processing import cap_dataframe, dataset_dict_from_frames
else:
    from dataset_processing import cap_dataframe, dataset_dict_from_frames

if __package__:
    from .dataset_downloads import download_json_records
else:
    from dataset_downloads import download_json_records

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
BASE_URL = "https://media.githubusercontent.com/media/plumaj/ltzGLUE/main/data/hc"

# Capping limits for mini datasets
MAX_TRAIN = 1024
MAX_VAL = 256
MAX_TEST = 2048


def main() -> None:
    """Create the ltzGLUE-HC dataset and upload to HF Hub."""
    logger.info("Downloading ltzGLUE-HC data from GitHub...")
    train_data = download_json_records(base_url=BASE_URL, split="train")
    val_data = download_json_records(base_url=BASE_URL, split="dev")
    test_data = download_json_records(base_url=BASE_URL, split="test")

    logger.info(
        f"Downloaded: {len(train_data)} train, {len(val_data)} val, "
        f"{len(test_data)} test"
    )

    train_df = _load_split(train_data)
    val_df = _load_split(val_data)
    test_df = _load_split(test_data)

    # Cap each split independently, preserving source split boundaries
    train_df = cap_dataframe(frame=train_df, maximum=MAX_TRAIN)
    val_df = cap_dataframe(frame=val_df, maximum=MAX_VAL)
    test_df = cap_dataframe(frame=test_df, maximum=MAX_TEST)

    logger.info(
        f"Capped splits: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test"
    )

    dataset = dataset_dict_from_frames(
        frames={"train": train_df, "val": val_df, "test": test_df},
        columns=["text", "label"],
    )

    dataset_id = "EuroEval/ltzglue-hc-mini"
    logger.info(f"Uploading to {dataset_id}...")

    preflight_private_upload(repository_id=dataset_id)
    dataset.push_to_hub(dataset_id, private=True)
    logger.info(f"✓ Uploaded {dataset_id}")


def _load_split(data: list[dict]) -> pd.DataFrame:
    """Load headline classification data.

    Args:
        data:
            List of records from the JSON file.

    Returns:
        DataFrame with text and label columns.
    """
    return pd.DataFrame(
        [
            {
                "text": f"{item['title']} | {item['text_long']}",
                "label": "yes" if item["is_correct"] == "True" else "no",
            }
            for item in data
        ]
    )


if __name__ == "__main__":
    main()
