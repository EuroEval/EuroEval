# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create the GoldenSwag-mini datasets and upload them to the HF Hub."""

from collections import Counter

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
from sklearn.model_selection import train_test_split

OPTIONS_MAPPING = {
    "da": "Svarmuligheder",
    "de": "Antwortmöglichkeiten",
    "es": "Opciones",
    "fi": "Vastausvaihtoehdot",
    "fr": "Choix",
    "it": "Scelte",
    "nl": "Antwoordopties",
    "sv": "Svarsalternativ",
    "pt": "Opções",
}


def main() -> None:
    """Create the GoldenSwag datasets and upload them to the HF Hub."""
    # Define the base download URL
    repo_id = "LumiOpen/opengpt-x_goldenswagx"

    subset_language_code = {
        "DA": "da",
        "DE": "de",
        "ES": "es",
        "FI": "fi",
        "FR": "fr",
        "IT": "it",
        "NL": "nl",
        "SV": "sv",
        "PT-PT": "pt",
    }

    for subset_name, language_code in subset_language_code.items():
        val_dataset = load_dataset(
            path=repo_id, name=subset_name, token=True, split="validation"
        )
        val_df = process_(dataset=val_dataset, language_code=language_code)

        train_dataset = load_dataset(
            path=repo_id, name=subset_name, token=True, split="train"
        )
        train_df = process_(dataset=train_dataset, language_code=language_code)

        total_dataset = pd.concat([train_df, val_df])

        val_size, test_size, _ = 256, 2048, 1024

        val_df, rest = train_test_split(
            total_dataset,
            train_size=val_size,
            random_state=4242,
            stratify=total_dataset.activity_label,
        )

        train_df, test_df = train_test_split(
            rest, test_size=test_size, random_state=4242, stratify=rest.activity_label
        )

        # Collect datasets in a dataset dictionary
        dataset = DatasetDict(
            train=Dataset.from_pandas(train_df, split=Split.TRAIN),
            val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
            test=Dataset.from_pandas(test_df, split=Split.TEST),
        )

        dataset_id = f"EuroEval/goldenswag-{language_code}-mini"

        # Remove the dataset from Hugging Face Hub if it already exists
        try:
            api = HfApi()
            api.delete_repo(dataset_id, repo_type="dataset")
        except HTTPError:
            pass

        # Push the dataset to the Hugging Face Hub
        dataset.push_to_hub(dataset_id, private=True)


def process_(dataset: Dataset, language_code: str) -> pd.DataFrame:
    """Process the dataset.

    Args:
        dataset: HuggingFace Dataset to process.

    Returns:
        pandas.DataFrame with columns 'text', 'label', and 'activity_label'.
    """
    assert isinstance(dataset, Dataset)

    # Convert the dataset to a dataframe
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Remove the samples with overly short or long texts
    df = df[
        (df.ctx.str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION)
        & (df.ctx.str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION)
        & df.endings.map(
            lambda endings: min(len(ending) for ending in endings)
            >= MIN_NUM_CHARS_IN_OPTION
            and max(len(ending) for ending in endings) <= MAX_NUM_CHARS_IN_OPTION
        )
    ]

    def is_repetitive(text: str) -> bool:
        """Check if the text is repetitive.

        Args:
            text: input string to check for repetition

        Returns:
            True if the text is repetitive, False otherwise
        """
        max_repetitions = max(Counter(text.split()).values())
        return max_repetitions > MAX_REPETITIONS

    # Remove overly repetitive samples
    df = df[
        ~df.ctx.apply(is_repetitive)
        & ~df.endings.map(
            lambda endings: any(is_repetitive(ending) for ending in endings)
        )
    ]

    # Make a `text` column with all the options in it
    df["text"] = [
        row.ctx.replace("\n", " ").strip() + "\n"
        f"{OPTIONS_MAPPING[language_code]}:\n"
        "a. " + row.endings[0].replace("\n", " ").strip() + "\n"
        "b. " + row.endings[1].replace("\n", " ").strip() + "\n"
        "c. " + row.endings[2].replace("\n", " ").strip() + "\n"
        "d. " + row.endings[3].replace("\n", " ").strip()
        for _, row in df.iterrows()
    ]

    # Fix the label column
    label_mapping = {"0": "a", "1": "b", "2": "c", "3": "d"}
    df.label = df.label.map(label_mapping)

    # Only keep the samples whose `activity_label` has at least 3 samples
    acceptable_activity_labels = [
        activity_label
        for activity_label, count in df["activity_label"].value_counts().items()
        if count >= 3
    ]
    df = df[df["activity_label"].isin(acceptable_activity_labels)]

    # Remove duplicates
    df.drop_duplicates(subset="text", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Make the `label` column case-consistent with the `text` column
    df.label = df.label.str.lower()

    # Only keep the columns `text`, `label` and `activity_label`
    df = df[["text", "label", "activity_label"]]
    return df


if __name__ == "__main__":
    main()
