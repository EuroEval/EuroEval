"""Create the Finnish HellaSwag-mini dataset and upload it to the HF Hub."""

from collections import Counter
from logging import getLogger

import pandas as pd
from constants import (
    MAX_NUM_CHARS_IN_INSTRUCTION,
    MAX_NUM_CHARS_IN_OPTION,
    MAX_REPETITIONS,
    MIN_NUM_CHARS_IN_INSTRUCTION,
    MIN_NUM_CHARS_IN_OPTION,
)
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError

logger = getLogger(__name__)


def main() -> None:
    """Create the Finnish HellaSwag-mini dataset and upload it to the HF Hub.

    HellaSwag-fi is designed to EuroEval so it already has a
    1,024 / 256 / 2,048 train/val/test split. This script
    therefore does not perform the same filtering as seen in
    `src/scripts/create_hellaswag.py`.
    """
    # Define the base download URL
    repo_id = "Finnish-NLP/hellaswag-fi-google-translate"

    dataset = load_dataset(path=repo_id, token=True)
    dfs = {}
    for split in ["train", "validation", "test"]:
        df = dataset[split].to_pandas()

        df["endings"] = df["endings"].apply(process_endings)

        df = process_split(df=df, split=split)
        dfs[split] = df

    # Create validation split
    val_df = dfs["validation"]
    assert len(val_df) == 256

    # Create test split
    test_df = dfs["test"]
    assert len(test_df) == 2_048

    # Create train split
    train_df = dfs["train"]
    assert len(train_df) == 1_024

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    dataset_id = "EuroEval/hellaswag-fi-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def process_endings(raw_endings: str) -> list[str]:
    """Process the endings of the HellaSwag-fi dataset.

    Args:
        raw_endings: The endings of the HellaSwag-fi dataset.

    Returns:
        The processed endings.
    """
    # Remove starting/ending square brackets and quotes
    raw_endings = raw_endings[2:-2]

    # Most endings are separated by newline characters
    endings = raw_endings.split("\n")
    if len(endings) != 4:
        # If not, try a more hacky approach
        endings = _extract_endings(raw_endings=raw_endings)
        assert len(endings) == 4
    else:
        # Fix quotes
        endings = [_fix_quotes(ending) for ending in endings]
    return endings


def _fix_quotes(ending: str) -> str:
    r"""Remove starting and ending quotes.

    Some entries have mixed quotes, e.g.
        '\'Aloha"'
    which should be turned into the following:
        'Aloha'

    Args:
        ending: The ending to process.

    Returns:
        The processed ending.
    """
    ending = ending.strip()
    # Remove start and end quotes
    if ending[0] in ['"', "'"]:
        ending = ending[1:]
    if ending[-1] in ['"', "'"]:
        ending = ending[:-1]

    return ending


def _extract_endings(raw_endings: str) -> list[str]:
    """Extract endings from the HellaSwag-fi dataset.

    This strategy is used for samples where the endings are
    not simply separated by newline characters.

    Args:
        raw_endings: The endings of the HellaSwag-fi dataset.

    Returns:
        The processed endings.
    """
    endings: list[str] = []
    i: int = 0
    start: int = 0
    while i < len(raw_endings) - 2:
        if raw_endings[i].isalpha() or raw_endings[i] == "[":
            start = i

            while i < len(raw_endings) - 2:
                # An ending seems to end with one of the following
                # char combinations.
                ends = ['."', ".'", "'.", '".', ".\n"]
                if raw_endings[i : i + 2] in ends:
                    ending = raw_endings[start : i + 1].strip()
                    endings.append(ending)
                    i += 1
                    break
                i += 1
        else:
            i += 1

    # Last ending
    ending = raw_endings[start:].strip()
    endings.append(ending)
    return endings


def process_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Process the split of the HellaSwag-fi dataset.

    Args:
        df: The dataframe to process.
        split: The split of the dataframe.

    Returns:
        The processed dataframe.
    """
    _print_filtering_stats(df=df, split=split)

    # Make a `text` column with all the options in it
    df["text"] = [
        row.ctx.replace("\n", " ").strip() + "\n"
        "Vastausvaihtoehdot:\n"
        "a. " + row.endings[0].replace("\n", " ").strip() + "\n"
        "b. " + row.endings[1].replace("\n", " ").strip() + "\n"
        "c. " + row.endings[2].replace("\n", " ").strip() + "\n"
        "d. " + row.endings[3].replace("\n", " ").strip()
        for _, row in df.iterrows()
    ]

    # Fix the label column
    label_mapping = {0: "a", 1: "b", 2: "c", 3: "d"}
    df.label = df.label.map(label_mapping)

    # Only keep the columns `text`, `label` and `activity_label`
    df = df[["text", "label", "activity_label"]]

    return df


def _print_filtering_stats(df: pd.DataFrame, split: str) -> None:
    """Print filtering statistics.

    For the Finnish HellaSwag-mini dataset, we don't want to filter
    any samples. These prints give an overview about how many samples
    in the dataset that would be filtered out by the different filters
    that are applied in `src/scripts/create_hellaswag.py`.

    Args:
        df: The dataframe to print statistics for.
        split: The split of the dataframe
    """
    short_ctx_count = sum(df.ctx.str.len() < MIN_NUM_CHARS_IN_INSTRUCTION)
    long_ctx_count = sum(df.ctx.str.len() > MAX_NUM_CHARS_IN_INSTRUCTION)

    short_endings_count = sum(
        df.endings.map(
            lambda endings: min(len(ending) for ending in endings)
            < MIN_NUM_CHARS_IN_OPTION
        )
    )
    long_endings_count = sum(
        df.endings.map(
            lambda endings: max(len(ending) for ending in endings)
            > MAX_NUM_CHARS_IN_OPTION
        )
    )
    logger.info(f"\nSplit: {split}")
    logger.info(
        "Samples with too short context: "
        f"{short_ctx_count} ({short_ctx_count / len(df):.2%})"
    )
    logger.info(
        "Samples with too long context: "
        f"{long_ctx_count} ({long_ctx_count / len(df):.2%})"
    )
    logger.info(
        "Samples with too short options: "
        f"{short_endings_count} ({short_endings_count / len(df):.2%})"
    )
    logger.info(
        "Samples with too long options: "
        f"{long_endings_count} ({long_endings_count / len(df):.2%})"
    )

    def is_repetitive(text: str) -> bool:
        """Return True if the text is repetitive."""
        if text == "":
            return False
        max_repetitions = max(Counter(text.split()).values())
        return max_repetitions > MAX_REPETITIONS

    repetitive_ctx_count = sum(df.ctx.apply(is_repetitive))
    repetitive_endings_count = sum(
        df.endings.map(lambda endings: any(is_repetitive(ending) for ending in endings))
    )

    logger.info(
        "Samples with repetitive context: "
        f"{repetitive_ctx_count} ({repetitive_ctx_count / len(df):.2%})"
    )
    logger.info(
        "Samples with repetitive options: "
        f"{repetitive_endings_count} ({repetitive_endings_count / len(df):.2%})"
    )

    activity_label_counts = df["activity_label"].value_counts()
    infrequent_labels = activity_label_counts[activity_label_counts < 3].index.tolist()
    infrequent_label_count = sum(df["activity_label"].isin(infrequent_labels))

    logger.info(
        "Activity labels with fewer than 3 samples: "
        f"{len(infrequent_labels)} out of {len(activity_label_counts)}"
    )
    logger.info(
        "Samples with infrequent activity labels: "
        f"{infrequent_label_count} ({infrequent_label_count / len(df):.2%})"
    )


if __name__ == "__main__":
    main()
