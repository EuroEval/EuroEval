# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create the Faroese Grammatical Correctness dataset and upload it to the HF Hub."""

import logging
import sys

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

logging.basicConfig(format="%(asctime)s ⋅ %(message)s", level=logging.INFO)
logger = logging.getLogger("create_faroese_grammatical_correctness")


def main(source: str) -> None:
    """Create the Faroese Grammatical Correctness dataset and upload it to the HF Hub.

    The source dataset consists of minimal pairs of an ungrammatical sentence and
    its corrected version, compiled from high school essays. This is a grammatical
    error correction dataset, where the model must correct the grammar of the
    ungrammatical sentence, and the result is evaluated against the corrected
    version.

    Args:
        source:
            Path or URL to the source JSONL file.
    """
    dataset = load_dataset("json", data_files=source, split="train")
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Verify the schema of the source dataset
    expected_columns = {"index_number", "original_sentence", "corrected_sentence"}
    assert expected_columns <= set(df.columns), (
        f"Expected columns {expected_columns}, but got {set(df.columns)}."
    )
    logger.info(f"Loaded {len(df)} minimal pairs from {source}.")

    # Build the samples, with the ungrammatical sentence as the input text and the
    # corrected sentence as the target text
    records: list[dict[str, str]] = []
    num_skipped = 0
    for _, row in df.iterrows():
        original = str(row.original_sentence).replace("\n", " ").strip()
        corrected = str(row.corrected_sentence).replace("\n", " ").strip()
        if not original or not corrected or original == corrected:
            num_skipped += 1
            continue
        records.append({"text": original, "target_text": corrected})
    new_df = pd.DataFrame(records)

    if num_skipped > 0:
        logger.warning(f"Skipped {num_skipped} malformed minimal pairs.")

    # Remove duplicates
    new_df.drop_duplicates(inplace=True)
    new_df.reset_index(drop=True, inplace=True)
    logger.info(f"Built {len(new_df)} samples from {len(df)} minimal pairs.")

    # Create the splits: train, val and test follow the standard EuroEval sizes, and
    # the samples left over after the test split go into the full training pool
    train_df, remaining_df = train_test_split(
        new_df, train_size=1_024, random_state=4242
    )
    val_df, remaining_df = train_test_split(
        remaining_df, train_size=256, random_state=4242
    )
    test_df, leftover_df = train_test_split(
        remaining_df, train_size=2_048, random_state=4242
    )
    full_train_df = pd.concat([train_df, leftover_df])

    # Reset the index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    full_train_df = full_train_df.reset_index(drop=True)

    logger.info(
        f"Split sizes: train={len(train_df)}, val={len(val_df)}, "
        f"test={len(test_df)}, full_train={len(full_train_df)}."
    )

    # Collect datasets in a dataset dictionary
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df, split=Split.TEST),
            "full_train": Dataset.from_pandas(full_train_df, split="full_train"),  # ty: ignore[invalid-argument-type]
        }
    )

    dataset_id = "EuroEval/faroese-grammatical-correctness"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset_dict.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <path-or-url-to-jsonl>")
    main(source=sys.argv[1])
