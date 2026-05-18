# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Schibsted front-page title and SEO title datasets."""

import pandas as pd
from constants import MAX_NUM_CHARS_IN_ARTICLE, MIN_NUM_CHARS_IN_ARTICLE
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi

TRAIN_SIZE = 64
VAL_SIZE = 128
RANDOM_STATE = 4242


def main() -> None:
    """Create the Schibsted title datasets and upload to HF Hub."""
    # Norwegian front-page title dataset (VG newsroom)
    process_title_dataset(
        source_dataset_id="Schibsted/vg-front-title",
        title_column="front_title",
        euroeval_dataset_id="EuroEval/vg-front-title",
    )

    # Swedish SEO title dataset (SVD newsroom)
    process_title_dataset(
        source_dataset_id="Schibsted/svd-seo-title",
        title_column="google_title",
        euroeval_dataset_id="EuroEval/svd-seo-title",
    )


def process_title_dataset(
    source_dataset_id: str, title_column: str, euroeval_dataset_id: str
) -> None:
    """Process a single-newsroom Schibsted title dataset and upload to HF Hub.

    Samples 64 examples from the training split for the new training split.
    Pools the remaining training samples with the validation split (and test split if
    it exists), then samples 128 examples for the new validation split, and uses the
    rest (~1k) as the new test split.

    Args:
        source_dataset_id:
            The Hugging Face dataset ID of the source dataset.
        title_column:
            The column name containing the title in the source dataset.
        euroeval_dataset_id:
            The Hugging Face dataset ID for the EuroEval dataset.
    """
    dataset = load_dataset(source_dataset_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Rename article text and title columns to text and target_text
    dataset = dataset.rename_columns(
        column_mapping={"article_text": "text", title_column: "target_text"}
    )

    # Convert to pandas and filter by article length
    train_df = dataset["train"].to_pandas()
    val_key = "validation" if "validation" in dataset else "val"
    val_df = dataset[val_key].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)

    lower_bound = MIN_NUM_CHARS_IN_ARTICLE
    upper_bound = MAX_NUM_CHARS_IN_ARTICLE
    train_df = train_df[train_df.text.str.len().between(lower_bound, upper_bound)]
    val_df = val_df[val_df.text.str.len().between(lower_bound, upper_bound)]
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)

    # Sample 64 examples for the new training split
    new_train_df = train_df.sample(n=TRAIN_SIZE, random_state=RANDOM_STATE)
    remaining_train_df = train_df.drop(new_train_df.index)

    # Pool: remaining train + validation + test (if it exists)
    pool_parts = [remaining_train_df, val_df]
    if "test" in dataset:
        test_df = dataset["test"].to_pandas()
        assert isinstance(test_df, pd.DataFrame)
        test_df = test_df[test_df.text.str.len().between(lower_bound, upper_bound)]
        assert isinstance(test_df, pd.DataFrame)
        pool_parts.append(test_df)
    pool_df = pd.concat(pool_parts, ignore_index=True)

    # Sample 128 examples for the new validation split; rest becomes the test split
    new_val_df = pool_df.sample(n=VAL_SIZE, random_state=RANDOM_STATE)
    new_test_df = pool_df.drop(new_val_df.index).reset_index(drop=True)

    new_train_df = new_train_df.reset_index(drop=True)
    new_val_df = new_val_df.reset_index(drop=True)

    result = DatasetDict(
        {
            "train": Dataset.from_pandas(new_train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(new_val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(new_test_df, split=Split.TEST),
        }
    )

    HfApi().delete_repo(euroeval_dataset_id, repo_type="dataset", missing_ok=True)
    result.push_to_hub(euroeval_dataset_id, private=True)


if __name__ == "__main__":
    main()
