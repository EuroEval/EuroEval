# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
# ]
# ///

"""Create the CulturaQA dataset and upload it to the HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi

MAX_SPLIT_SIZES = dict(train=1_024, val=256, test=2_048)
RANDOM_SEED = 4242
REQUIRED_SPLITS = frozenset(MAX_SPLIT_SIZES)
SOURCE_COLUMNS = ["id", "question", "answer", "category"]
COLUMN_RENAMES = dict(question="text", answer="target_text")


def main() -> None:
    """Create the CulturaQA dataset and upload it to the HF Hub."""
    source_dataset_id = "IMISLab/CulturaQA"
    target_dataset_id = "EuroEval/culturaqa-mini"

    raw_dataset = load_dataset(path=source_dataset_id, name="all", token=True)
    assert isinstance(raw_dataset, DatasetDict)

    missing_splits = REQUIRED_SPLITS.difference(raw_dataset)
    assert not missing_splits, f"Missing source splits: {sorted(missing_splits)}"

    train_df = process_split(dataset=raw_dataset["train"], split_name="train")
    val_df = process_split(dataset=raw_dataset["val"], split_name="val")
    test_df = process_split(dataset=raw_dataset["test"], split_name="test")

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df, split=Split.TEST),
        }
    )

    HfApi().delete_repo(target_dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(target_dataset_id, private=True)


def process_split(dataset: Dataset, split_name: str) -> pd.DataFrame:
    """Convert one source split to EuroEval format and apply the split cap.

    Args:
        dataset:
            The source dataset split.
        split_name:
            The source split name.

    Returns:
        The processed and capped split dataframe.
    """
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    df = df[SOURCE_COLUMNS].rename(columns=COLUMN_RENAMES)
    return cap_split(df=df, split_name=split_name).reset_index(drop=True)


def cap_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Cap a split deterministically according to the EuroEval dataset rules.

    Args:
        df:
            The split dataframe.
        split_name:
            The source split name.

    Returns:
        The capped split dataframe.
    """
    max_size = MAX_SPLIT_SIZES[split_name]
    if len(df) <= max_size:
        return df
    return df.sample(n=max_size, random_state=RANDOM_SEED)


if __name__ == "__main__":
    main()
