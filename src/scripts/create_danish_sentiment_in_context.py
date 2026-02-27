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

"""Create the Danish Sentiment in Context dataset and upload it to the HF Hub."""

import io
from zipfile import ZipFile

import pandas as pd
import requests as rq
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


def main() -> None:
    """Create the Danish Sentiment in Context dataset and upload it to the HF Hub."""
    # Download the ZIP file from the Danish Semantic Reasoning Benchmark repository
    url = (
        "https://github.com/kuhumcst/danish-semantic-reasoning-benchmark/raw/main"
        "/sentiment/sentiment.zip"
    )
    response = rq.get(url=url)
    response.raise_for_status()

    # Extract the TSV file from the ZIP
    with ZipFile(file=io.BytesIO(initial_bytes=response.content)) as zip_file:
        tsv_files = [
            zip_file.read(name=file_name)
            for file_name in zip_file.namelist()
            if file_name.endswith(".tsv")
        ]
        assert len(tsv_files) == 1, (
            f"Expected one TSV file in the ZIP file, but found {len(tsv_files)}."
        )
        df = pd.read_csv(
            filepath_or_buffer=io.BytesIO(initial_bytes=tsv_files[0]), sep="\t"
        )

    # Build the 'text' column by combining the target word and context
    df["text"] = "Ord: " + df["target"] + "\nKontekst: " + df["context"]

    # Map the sentiment scale (-3 to +3) to positive/neutral/negative labels
    def map_label(score: int) -> str:
        if score < 0:
            return "negative"
        elif score > 0:
            return "positive"
        else:
            return "neutral"

    df["label"] = df["label"].apply(map_label)

    # Keep only the relevant columns
    df = df[["text", "label"]].copy()

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create validation split
    val_size = 64
    traintest_df, val_df = train_test_split(
        df, test_size=val_size, random_state=4242, stratify=df["label"]
    )

    # Create train and test splits
    train_size = 128
    train_df, test_df = train_test_split(
        traintest_df,
        train_size=train_size,
        random_state=4242,
        stratify=traintest_df["label"],
    )

    # Reset the indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df, split=Split.TEST),
        }
    )

    # Create dataset ID
    dataset_id = "EuroEval/danish-sentiment-in-context"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
