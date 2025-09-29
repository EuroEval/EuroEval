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

"""Create the MMLU-et dataset and upload to HF Hub."""

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
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


def main() -> None:
    """Create the MMLU-et dataset and upload to HF Hub."""
    disable_progress_bars()
    repo_id = "TalTechNLP/MMLU_et"

    # Get all the subsets of the dataset
    subsets = [
        config["config_name"]
        for config in HfApi()
        .repo_info(repo_id=repo_id, repo_type="dataset")
        .card_data.configs
    ]

    # Download the dataset
    dfs: list[pd.DataFrame] = list()
    for subset in tqdm(subsets, desc="Downloading subsets"):
        dataset = load_dataset(path=repo_id, name=subset, split="train")
        assert isinstance(dataset, Dataset)
        df = dataset.to_pandas()
        assert isinstance(df, pd.DataFrame)
        df["category"] = subset
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # Rename the columns
    df.rename(columns=dict(answer="label", question="instruction"), inplace=True)

    # Convert the labels to single lowercase letters
    df.label = df.label.map(lambda x: "abcd"[x])

    # Double check that there are exactly 4 choices per question
    assert df.choices.apply(len).eq(4).all(), "Not all questions have exactly 4 choices"

    # Split up the choices into separate columns
    df["option_a"] = df.choices.apply(lambda x: x[0])
    df["option_b"] = df.choices.apply(lambda x: x[1])
    df["option_c"] = df.choices.apply(lambda x: x[2])
    df["option_d"] = df.choices.apply(lambda x: x[3])

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
    assert isinstance(df, pd.DataFrame)

    # Make a `text` column with all the options in it
    df["text"] = [
        row.instruction.replace("\n", " ").strip() + "\n"
        f"{CHOICES_MAPPING['et']}:\n"
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

    # Push the dataset to the Hugging Face Hub
    dataset_id = "EuroEval/mmlu-et-mini"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
