# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "polars==1.31.0",
#     "requests==2.32.3",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create the MMLU-mini datasets and upload them to the HF Hub."""

from collections import Counter

import pandas as pd
import polars as pl
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
from sklearn.model_selection import train_test_split


def main() -> None:
    """Create the MMLU-mini datasets and upload them to the HF Hub."""
    # Define the base download URL
    repo_id = "alexandrainst/m_mmlu"

    # Create a mapping with the word "Choices" in different languages
    choices_mapping = {
        "da": "Svarmuligheder",
        "no": "Svaralternativer",
        "sv": "Svarsalternativ",
        "is": "Svarmöguleikar",
        "de": "Antwortmöglichkeiten",
        "nl": "Antwoordopties",
        "en": "Choices",
        "fr": "Choix",
        "it": "Scelte",
        "es": "Opciones",
        "pt": "Opções",
    }

    for language in choices_mapping.keys():
        # Download the dataset
        try:
            dataset = load_dataset(path=repo_id, name=language, token=True)
        except ValueError as e:
            if language == "no":
                dataset = load_dataset(path=repo_id, name="nb", token=True)
            elif language == "pt":
                dataset = load_pt_dataset()
            else:
                raise e
        assert isinstance(dataset, DatasetDict)

        # Convert the dataset to a dataframe
        train_df = dataset["train"].to_pandas()
        val_df = dataset["val"].to_pandas()
        test_df = dataset["test"].to_pandas()
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

        # Concatenate the splits
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        # Rename the columns
        df.rename(columns=dict(answer="label"), inplace=True)

        # Remove the samples with overly short or long texts
        df = df[
            (df.instruction.str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION)
            & (df.instruction.str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION)
            & (df.option_a.str.len() >= MIN_NUM_CHARS_IN_OPTION)
            & (df.option_a.str.len() <= MAX_NUM_CHARS_IN_OPTION)
            & (df.option_b.str.len() >= MIN_NUM_CHARS_IN_OPTION)
            & (df.option_b.str.len() <= MAX_NUM_CHARS_IN_OPTION)
            & (df.option_c.str.len() >= MIN_NUM_CHARS_IN_OPTION)
            & (df.option_c.str.len() <= MAX_NUM_CHARS_IN_OPTION)
            & (df.option_d.str.len() >= MIN_NUM_CHARS_IN_OPTION)
            & (df.option_d.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        ]

        def is_repetitive(text: str) -> bool:
            """Return True if the text is repetitive."""
            max_repetitions = max(Counter(text.split()).values())
            return max_repetitions > MAX_REPETITIONS

        # Remove overly repetitive samples
        df = df[
            ~df.instruction.apply(is_repetitive)
            & ~df.option_a.apply(is_repetitive)
            & ~df.option_b.apply(is_repetitive)
            & ~df.option_c.apply(is_repetitive)
            & ~df.option_d.apply(is_repetitive)
        ]

        # Extract the category as a column
        df["category"] = df["id"].str.split("/").str[0]

        # Make a `text` column with all the options in it
        df["text"] = [
            row.instruction.replace("\n", " ").strip() + "\n"
            f"{choices_mapping[language]}:\n"
            "a. " + row.option_a.replace("\n", " ").strip() + "\n"
            "b. " + row.option_b.replace("\n", " ").strip() + "\n"
            "c. " + row.option_c.replace("\n", " ").strip() + "\n"
            "d. " + row.option_d.replace("\n", " ").strip()
            for _, row in df.iterrows()
        ]

        # Make the `label` column case-consistent with the `text` column
        df.label = df.label.str.lower()

        # Only keep the `text`, `label` and `category` columns
        df = df[["text", "label", "category"]]

        # Remove duplicates
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Create validation split
        val_size = 256
        traintest_arr, val_arr = train_test_split(
            df, test_size=val_size, random_state=4242, stratify=df.category
        )
        traintest_df = pd.DataFrame(traintest_arr, columns=df.columns)
        val_df = pd.DataFrame(val_arr, columns=df.columns)

        # Create test split
        test_size = 2048
        train_arr, test_arr = train_test_split(
            traintest_df,
            test_size=test_size,
            random_state=4242,
            stratify=traintest_df.category,
        )
        train_df = pd.DataFrame(train_arr, columns=df.columns)
        test_df = pd.DataFrame(test_arr, columns=df.columns)

        # Create train split
        train_size = 1024
        train_df = train_df.sample(train_size, random_state=4242)

        # Reset the index
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # Collect datasets in a dataset dictionary
        dataset = DatasetDict(
            train=Dataset.from_pandas(train_df, split=Split.TRAIN),
            val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
            test=Dataset.from_pandas(test_df, split=Split.TEST),
        )

        # Create dataset ID
        if language == "en":
            dataset_id = "EuroEval/mmlu-mini"
        else:
            dataset_id = f"EuroEval/mmlu-{language}-mini"

        # Remove the dataset from Hugging Face Hub if it already exists
        try:
            api = HfApi()
            api.delete_repo(dataset_id, repo_type="dataset")
        except HTTPError:
            pass

        # Push the dataset to the Hugging Face Hub
        dataset.push_to_hub(dataset_id, private=True)


def load_pt_dataset() -> DatasetDict:
    """Load and process PT-PT split from LumiOpen/opengpt-x_mmlux.

    Returns:
        DatasetDict: Hugging Face DatasetDict with train, val, and test splits.
    """

    def _process_split(split: str) -> pl.DataFrame:
        """Process a single split of the Portuguese MMLU dataset.

        Args:
            split (str): The split name ("dev", "validation", or "test").

        Returns:
            polars.DataFrame: Processed DataFrame for the split.
        """
        return (
            pl.read_ndjson(
                f"hf://datasets/LumiOpen/opengpt-x_mmlux/*PT-PT*{split}.jsonl"
            )
            .with_columns(
                pl.col("id").str.split("/").list.get(0).alias("category"),
                (pl.col("id") + f"/{split}").alias("id"),
            )
            .rename({"question": "instruction"})
            .with_columns(
                [
                    pl.col("choices").list.get(0).alias("option_a"),
                    pl.col("choices").list.get(1).alias("option_b"),
                    pl.col("choices").list.get(2).alias("option_c"),
                    pl.col("choices").list.get(3).alias("option_d"),
                ]
            )
            .with_columns(
                pl.col("answer").map_elements(
                    lambda x: {0: "a", 1: "b", 2: "c", 3: "d"}[x],
                    return_dtype=pl.String,
                )
            )
            .drop("choices")
            .drop("category")
        )

    train_df = _process_split("dev")
    val_df = _process_split("validation")
    test_df = _process_split("test")

    return DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )


if __name__ == "__main__":
    main()
