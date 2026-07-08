# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "scikit-learn==1.6.1",
# ]
# ///

"""Create the LuxGen-Summ summarisation dataset and upload to HF Hub.

This dataset is based on the LuxGen benchmark from the paper
"Text Generation Models for Luxembourgish with Limited Data: A Balanced
Multilingual Strategy" (https://arxiv.org/abs/2412.09415).

The dataset contains Luxembourgish news articles from major outlets
(Luxemburger Wort, RTL Lëtzebuerg, paperjam.lu, today.lu, wort.lu) with
human-written summaries covering politics, economy, culture, and sports.

NOTE: The original LuxGen dataset is not publicly available. This script
expects the data to be provided in a specific format. Contact the paper
authors or wait for official release.
"""

import logging

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_luxgen_data() -> pd.DataFrame:
    """Load LuxGen summarisation data.

    Returns:
        DataFrame with text and target_text columns.

    Note:
        This is a placeholder. The actual data needs to be obtained from
        the LuxGen benchmark authors or extracted from Luxembourgish news
        outlets (Luxemburger Wort, RTL Lëtzebuerg, paperjam.lu, today.lu,
        wort.lu) with human-written summaries.
    """
    logger.warning(
        "LuxGen dataset is not publicly available. "
        "Please obtain data from the paper authors or create from news sources."
    )

    # Placeholder structure - replace with actual data loading
    data = []
    return pd.DataFrame(data)


def create_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits (1024/256/2048).

    Args:
        df:
            Full dataset.

    Returns:
        Tuple of (train, val, test) DataFrames.

    Note:
        Uses random sampling (stratification not applicable for summarisation).
    """
    n = len(df)
    n_train = min(1024, int(n * 0.5))
    n_val = min(256, int(n * 0.15))
    n_test = min(2048, int(n * 0.35))

    train, temp = train_test_split(df, train_size=n_train, random_state=42)
    val, test = train_test_split(temp, train_size=n_val, test_size=n_test, random_state=42)

    for d in [train, val, test]:
        d.reset_index(drop=True, inplace=True)

    return train, val, test


def main() -> None:
    """Create the LuxGen-Summ dataset and upload to HF Hub."""
    logger.info("Loading LuxGen summarisation data...")
    df = load_luxgen_data()

    if len(df) == 0:
        logger.error(
            "No data found. Please obtain LuxGen data from the paper authors "
            "or create from Luxembourgish news outlets with human summaries."
        )
        logger.info(
            "Expected format: DataFrame with 'text' (article) and "
            "'target_text' (summary) columns"
        )
        return

    logger.info(f"Loaded {len(df)} samples")

    final_train, final_val, final_test = create_splits(df)

    logger.info(
        f"Created splits: {len(final_train)} train, {len(final_val)} val, "
        f"{len(final_test)} test"
    )

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(final_train[["text", "target_text"]]),
            "val": Dataset.from_pandas(final_val[["text", "target_text"]]),
            "test": Dataset.from_pandas(final_test[["text", "target_text"]]),
        }
    )

    dataset_id = "EuroEval/luxgen-summ"
    logger.info(f"Uploading to {dataset_id}...")

    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)
    logger.info(f"✓ Uploaded {dataset_id}")


if __name__ == "__main__":
    main()
