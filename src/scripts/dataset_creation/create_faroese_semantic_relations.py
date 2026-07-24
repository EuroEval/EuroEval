# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create the Faroese Semantic Relations dataset and upload it to the HF Hub."""

import logging
import random
import sys

import pandas as pd
from constants import CHOICES_MAPPING
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

logging.basicConfig(format="%(asctime)s ⋅ %(message)s", level=logging.INFO)
logger = logging.getLogger("create_faroese_semantic_relations")

LETTERS = ["a", "b", "c", "d", "e", "f"]


def main(source: str) -> None:
    """Create the Faroese Semantic Relations dataset and upload it to the HF Hub.

    The source dataset contains Faroese words, each with its correct antonym and
    five distractor words. We build a multiple-choice dataset where the model must
    pick the correct antonym of the word out of six candidates, with the answer
    options shuffled deterministically per sample.

    Args:
        source:
            Path or URL to the source JSONL file.
    """
    dataset = load_dataset("json", data_files=source, split="train")
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Verify the schema of the source dataset
    expected_columns = {"orð", "andheiti", "merksam", "orðaflokkur", "random_outliers"}
    assert expected_columns <= set(df.columns), (
        f"Expected columns {expected_columns}, but got {set(df.columns)}."
    )
    logger.info(f"Loaded {len(df)} samples from {source}.")

    # Build the multiple-choice samples, shuffling the answer options
    # deterministically per sample
    rng = random.Random(4242)
    records: list[dict[str, str]] = []
    num_skipped = 0
    for _, row in df.iterrows():
        word = str(row["orð"]).replace("\n", " ").strip()
        antonym = str(row["andheiti"]).replace("\n", " ").strip()
        distractors = [
            str(outlier).replace("\n", " ").strip()
            for outlier in row["random_outliers"]
        ]
        if not word or not antonym or len(distractors) != 5:
            num_skipped += 1
            continue
        options = [antonym] + distractors
        rng.shuffle(options)
        text = (
            f"Hvat er andheitið hjá orðinum '{word}'?\n"
            f"{CHOICES_MAPPING['fo']}:\n"
            + "\n".join(
                f"{letter}. {option}" for letter, option in zip(LETTERS, options)
            )
        )
        records.append({"text": text, "label": LETTERS[options.index(antonym)]})

    if num_skipped > 0:
        logger.warning(f"Skipped {num_skipped} malformed samples.")

    new_df = pd.DataFrame(records)

    # Remove duplicates
    new_df.drop_duplicates(inplace=True)
    new_df.reset_index(drop=True, inplace=True)
    logger.info(f"Built {len(new_df)} multiple-choice samples.")

    # Compute split sizes following the ratio of the standard EuroEval splits
    # (1,024 / 256 / 2,048 for train / val / test)
    train_size = int(len(new_df) * 1_024 / 3_328)
    val_size = int(len(new_df) * 256 / 3_328)
    logger.info(
        f"Splitting into {train_size} train / {val_size} val / "
        f"{len(new_df) - train_size - val_size} test samples."
    )

    # Create the splits
    train_df, remaining_df = train_test_split(
        new_df, train_size=train_size, random_state=4242
    )
    val_df, test_df = train_test_split(
        remaining_df, train_size=val_size, random_state=4242
    )

    # Reset the index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df, split=Split.TEST),
        }
    )

    # Create dataset ID
    dataset_id = "EuroEval/faroese-semantic-relations"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset_dict.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <path-or-url-to-jsonl>")
    main(source=sys.argv[1])
