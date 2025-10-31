# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Winogrande datasets and upload them to the HF Hub."""

import logging
import re
from collections import Counter

import pandas as pd
from constants import (
    CHOICES_MAPPING,
    MAX_NUM_CHARS_IN_INSTRUCTION,
    MAX_NUM_CHARS_IN_OPTION,
    MAX_REPETITIONS,
    MIN_NUM_CHARS_IN_INSTRUCTION,
    MIN_NUM_CHARS_IN_OPTION,
)
from datasets import Dataset, DatasetDict, Split, disable_progress_bars, load_dataset
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("create_winogrande")


LANGUAGES = [
    "bg",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fi",
    "fr",
    "hr",
    "it",
    "lt",
    "lv",
    "nl",
    "no",
    "pl",
    "pt",
    "sk",
    "sl",
    "sr",
    "sv",
    "uk",
]


def main() -> None:
    """Create the Winogrande datasets and upload them to the HF Hub."""
    disable_progress_bars()
    repo_id = "aialt/MuBench"

    for language in LANGUAGES:
        # Download the dataset
        dataset = load_dataset(
            path=repo_id, name=f"WinoGrandeDataset_local_template_{language}"
        )
        assert isinstance(dataset, DatasetDict)

        # Convert the dataset to a dataframe
        train_df = dataset["train"].to_pandas()
        test_df = dataset["test"].to_pandas()
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

        # Split test set into validation and test sets
        val_size = 128
        val_df = test_df.sample(n=val_size, random_state=42).reset_index(drop=True)
        test_df = test_df.drop(val_df.index.tolist()).reset_index(drop=True)

        train_df = prepare_dataframe(df=train_df, language=language)
        val_df = prepare_dataframe(df=val_df, language=language)
        test_df = prepare_dataframe(df=test_df, language=language)

        # Collect datasets in a dataset dictionary
        dataset = DatasetDict(
            train=Dataset.from_pandas(train_df, split=Split.TRAIN),
            val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
            test=Dataset.from_pandas(test_df, split=Split.TEST),
        )

        logger.info(
            f"Final sizes for the Winogrande {language} dataset: "
            f"{len(dataset['train'])} train, {len(dataset['val'])} val, "
            f"{len(dataset['test'])} test"
        )

        # Push the dataset to the Hugging Face Hub
        target_dataset_id = f"EuroEval/winogrande-{language}"
        HfApi().delete_repo(target_dataset_id, repo_type="dataset", missing_ok=True)
        dataset.push_to_hub(target_dataset_id, private=True)


def prepare_dataframe(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """Prepare the dataframe by filtering and processing.

    Args:
        df:
            Input dataframe.
        language:
            The language code that we are processing the dataframe for.

    Returns:
        Processed dataframe.
    """
    df.rename(columns=dict(prompt="instruction"), inplace=True)

    # Double check that there are exactly 6 lines in the instructions. These are:
    # 1. The instruction itself
    # 2. The equivalent of "What does the blank _ refer to?"
    # 3. The equivalent of "Option A: (...)"
    # 4. The equivalent of "Option B: (...)"
    # 5. "Answer with A or B."
    # 6. "Answer:"
    # We want to keep the actual instruction (lines 1-2) as well as the choices (lines
    # 3-4, without the "Option X" prefix), so we will trim the rest away.
    assert df.instruction.map(lambda x: x.count("\n")).eq(5).all(), (
        "Not all instructions have exactly 6 lines!"
    )
    df["option_a"] = df.instruction.map(
        lambda x: re.sub(r"[^ ]+ A: ?", "", x.split("\n")[2]).strip()
    )
    df["option_b"] = df.instruction.map(
        lambda x: re.sub(r"[^ ]+ B: ?", "", x.split("\n")[3]).strip()
    )
    df.instruction = df.instruction.map(lambda x: " ".join(x.split("\n")[:2]).strip())

    # Remove the samples with overly short or long texts
    df = df[
        (df.instruction.str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION)
        & (df.instruction.str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION)
        & (df.option_a.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_a.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        & (df.option_b.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_b.str.len() <= MAX_NUM_CHARS_IN_OPTION)
    ]

    # Remove overly repetitive samples
    df = df[
        ~df.instruction.apply(is_repetitive)
        & ~df.option_a.apply(is_repetitive)
        & ~df.option_b.apply(is_repetitive)
    ]

    # Make a `text` column with all the options in it
    df["text"] = [
        row.instruction.replace("\n", " ").strip() + "\n"
        f"{CHOICES_MAPPING[language]}:\n"
        "a. " + row.option_a.replace("\n", " ").strip() + "\n"
        "b. " + row.option_b.replace("\n", " ").strip()
        for _, row in df.iterrows()
    ]

    # Make the `label` column case-consistent with the `text` column
    df.label = df.label.map(lambda x: "a" if x == 0 else "b")

    # Only keep the `text` and `label` columns
    df = df[["text", "label"]]

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def is_repetitive(text: str) -> bool:
    """Return True if the text is repetitive."""
    max_repetitions = max(Counter(text.split()).values())
    return max_repetitions > MAX_REPETITIONS


if __name__ == "__main__":
    main()
