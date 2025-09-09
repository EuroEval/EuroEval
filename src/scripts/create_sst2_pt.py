# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "numpy==1.23.2",
#     "pandas==2.2.0",
#     "scikit-learn==1.7.0",
# ]
# ///

"""Create the SST2-pt-mini dataset and upload to HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi

RANDOM_STATE = 4242
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 1024, 256, 2048
ORIGINAL_REPO_ID = "PORTULAN/extraglue"
FINAL_REPO_ID = "EuroEval/sst2-pt-mini"


def main() -> None:
    """Create the dataset and upload to HF Hub."""
    ds_raw = load_dataset(ORIGINAL_REPO_ID, name="sst2_pt-PT")
    train_base = cleanup(ds_raw["train"].to_pandas())
    val_df = cleanup(ds_raw["validation"].to_pandas(), n=VAL_SIZE)

    train_df, rest = stratified_sample(train_base, TRAIN_SIZE)
    test_df, _ = stratified_sample(rest, TEST_SIZE)

    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    assert not set(dataset["train"]["text"]) & set(dataset["val"]["text"])
    assert not set(dataset["train"]["text"]) & set(dataset["test"]["text"])
    assert not set(dataset["val"]["text"]) & set(dataset["test"]["text"])

    # Push the dataset to the Hugging Face Hub
    HfApi().delete_repo(FINAL_REPO_ID, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(FINAL_REPO_ID, private=True)


def cleanup(df: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
    """Clean up the dataset.

    Args:
        df: DataFrame with 'sentence' and 'label' columns.
        n: Optional number of samples to return.

    Returns:
        Cleaned DataFrame with 'text' and 'label' columns.
    """
    df = df.rename(columns={"sentence": "text"})
    df["label"] = df["label"].map({0: "negative", 1: "positive"})
    df = df[["text", "label"]].drop_duplicates()
    if n:
        df = df.sample(n=min(n, len(df)), random_state=RANDOM_STATE)
    return df.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sampled_subset, remainder) with perfectly balanced labels.

    Args:
        df: DataFrame with 'label' column containing 'positive' and 'negative'.
        n: Total number of samples to select (must be even).

    Returns:
        Tuple of (sampled DataFrame, remainder DataFrame).
    """
    per_class = n // 2
    pos = df[df["label"] == "positive"].sample(n=per_class, random_state=RANDOM_STATE)  # noqa: E501
    neg = df[df["label"] == "negative"].sample(n=per_class, random_state=RANDOM_STATE)  # noqa: E501
    sample = pd.concat([pos, neg])
    rest = df.drop(sample.index)
    return sample.reset_index(drop=True), rest.reset_index(drop=True)


if __name__ == "__main__":
    main()
