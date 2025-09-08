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

"""Create the Exam-et dataset and upload it to the HF Hub."""

from collections import Counter

import pandas as pd
from constants import (
    MAX_NUM_CHARS_IN_INSTRUCTION,
    MAX_NUM_CHARS_IN_OPTION,
    MAX_REPETITIONS,
    MIN_NUM_CHARS_IN_INSTRUCTION,
    MIN_NUM_CHARS_IN_OPTION,
)
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


def main() -> None:
    """Create the Exam-et dataset and upload it to the HF Hub."""
    repo_id = "TalTechNLP/exam_et"
    target_repo_id = "EuroEval/exam-et"

    # Get all the subsets of the dataset
    api = HfApi(token=True)
    repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    subsets = [config["config_name"] for config in repo_info.card_data.configs]

    # Download all subsets and merge them
    rename_mapping = {
        "küsimus": "instruction",
        "vastusevariandid": "choices",
        "vastus": "label",
    }
    dfs: list[pd.DataFrame] = []
    for subset in subsets:
        ds = load_dataset(path=repo_id, name=subset, split="train", token=True)
        assert isinstance(ds, Dataset), f"Expected Dataset, got {type(ds)}"
        df = ds.to_pandas()
        assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"
        df.rename(columns=rename_mapping, inplace=True)
        df["category"] = subset
        df = df.loc[:, ["instruction", "choices", "label", "category"]]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # Remove the samples with overly short or long texts
    df = df.loc[
        (df.instruction.str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION)
        & (df.instruction.str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION)
        & df.choices.map(
            lambda endings: min(len(ending) for ending in endings)
            >= MIN_NUM_CHARS_IN_OPTION
            and max(len(ending) for ending in endings) <= MAX_NUM_CHARS_IN_OPTION
        )
    ]

    def is_repetitive(text: str) -> bool:
        """Return True if the text is repetitive."""
        max_repetitions = max(Counter(text.split()).values())
        return max_repetitions > MAX_REPETITIONS

    # Remove overly repetitive samples
    df = df.loc[
        ~df.instruction.apply(is_repetitive)
        & ~df.choices.map(
            lambda endings: any(is_repetitive(ending) for ending in endings)
        )
    ]

    # Make a `text` column with all the options in it
    label_str = "abcdefghijklmnopqrstuvwxyz"
    df["text"] = [
        row.instruction.replace("\n", " ").strip()
        + "\n"
        + "Vastusevariandid:\n"
        + "\n".join(
            f"{char}. {option.strip()}" for char, option in zip(label_str, row.choices)
        )
        for _, row in df.iterrows()
    ]
    df["label"] = df.label.map(lambda x: label_str[x])

    # Keep only the relevant columns
    df = df[["text", "label", "category"]]

    train_size = 512
    val_size = 64
    test_size = 896
    assert len(df) == train_size + val_size + test_size, (
        f"Expected {train_size + val_size + test_size:,} samples, got {len(df):,}"
    )

    # Create splits, stratifiying by category
    train_df, valtest_df = train_test_split(
        df, train_size=train_size, random_state=4242, stratify=df.category
    )
    val_df, test_df = train_test_split(
        valtest_df, test_size=test_size, random_state=4242, stratify=valtest_df.category
    )
    assert isinstance(train_df, pd.DataFrame), (
        f"Expected DataFrame, got {type(train_df)}"
    )
    assert isinstance(val_df, pd.DataFrame), f"Expected DataFrame, got {type(val_df)}"
    assert isinstance(test_df, pd.DataFrame), f"Expected DataFrame, got {type(test_df)}"

    # Shuffle the splits
    train_df = train_df.sample(frac=1, random_state=4242).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=4242).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=4242).reset_index(drop=True)

    # Convert to DatasetDict
    dataset = DatasetDict(
        dict(
            train=Dataset.from_pandas(train_df),
            val=Dataset.from_pandas(val_df),
            test=Dataset.from_pandas(test_df),
        )
    )

    api.delete_repo(repo_id=target_repo_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(target_repo_id, private=True)


if __name__ == "__main__":
    main()
