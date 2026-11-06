# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "numpy==1.26.0",
#     "pandas==2.2.0",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create the Faroese Semantic Relations dataset and upload it to the HF Hub."""

import logging
import random
import sys

import numpy as np
import pandas as pd
from constants import CHOICES_MAPPING
from datasets import load_dataset

if __package__:
    from .dataset_utils import preflight_private_upload
else:
    from dataset_utils import preflight_private_upload

if __package__:
    from .dataset_processing import (
        format_multiple_choice_record,
        split_generated_dataset,
    )
else:
    from dataset_processing import (
        format_multiple_choice_record,
        split_generated_dataset,
    )

logging.basicConfig(format="%(asctime)s ⋅ %(message)s", level=logging.INFO)
logger = logging.getLogger("create_faroese_semantic_relations")

LETTERS = ["a", "b", "c", "d", "e", "f"]


def main(source: str) -> None:
    """Create the Faroese Semantic Relations dataset and upload it to the HF Hub.

    The source dataset contains Faroese words, each with its correct antonym and
    five distractor words. We build a multiple-choice dataset where the model must
    pick the correct antonym of the word out of six candidates, with the answer
    options shuffled deterministically.

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
    # deterministically
    rng = random.Random(4242)
    records: list[dict[str, str]] = []
    num_skipped = 0
    for _, row in df.iterrows():
        word = str(row["orð"]).replace("\n", " ").strip()
        antonym = str(row["andheiti"]).replace("\n", " ").strip()
        outliers = row["random_outliers"]
        if not isinstance(outliers, (list, tuple, np.ndarray)):
            num_skipped += 1
            continue
        distractors = [str(outlier).replace("\n", " ").strip() for outlier in outliers]
        if not word or not antonym or len(distractors) != 5:
            num_skipped += 1
            continue
        records.append(
            format_multiple_choice_record(
                prompt=f"Hvat er andheitið hjá orðinum '{word}'?",
                correct_answer=antonym,
                distractors=distractors,
                letters=LETTERS,
                choices_label=CHOICES_MAPPING["fo"],
                rng=rng,
            )
        )

    if num_skipped > 0:
        logger.warning(f"Skipped {num_skipped} malformed samples.")

    new_df = pd.DataFrame(records)

    # Remove duplicates
    new_df.drop_duplicates(inplace=True)
    new_df.reset_index(drop=True, inplace=True)
    logger.info(f"Built {len(new_df)} multiple-choice samples.")

    # Split generated examples proportionally, then enforce the standard caps.
    dataset_dict, was_truncated = split_generated_dataset(frame=new_df)
    dataset_id = "EuroEval/faroese-semantic-relations"
    if was_truncated:
        dataset_id += "-mini"

    # Push the dataset to the Hugging Face Hub
    preflight_private_upload(repository_id=dataset_id)
    dataset_dict.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <path-or-url-to-jsonl>")
    main(source=sys.argv[1])
