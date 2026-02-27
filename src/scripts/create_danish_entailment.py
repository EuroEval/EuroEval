# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Danish Entailment dataset and upload it to the HF Hub."""

import io
import zipfile

import pandas as pd
import requests
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi


def main() -> None:
    """Create the Danish Entailment dataset and upload it to the HF Hub."""
    url = "https://raw.githubusercontent.com/kuhumcst/danish-semantic-reasoning-benchmark/main/entailment/entailment.zip"
    password = b"benchmark"

    # Download the ZIP archive
    response = requests.get(url)
    response.raise_for_status()

    # Extract and parse all .txt files from the ZIP
    zf = zipfile.ZipFile(io.BytesIO(response.content))
    rows: list[dict[str, str]] = []
    for name in sorted(zf.namelist()):
        if not name.endswith(".txt"):
            continue
        content = zf.read(name, pwd=password).decode("utf-8")
        for line in content.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            premise = parts[0].strip()
            hypothesis = parts[1].strip()
            label = parts[2].strip().lower()
            # Skip header/dummy rows and rows with unexpected labels
            if label not in {"true", "false", "neutral"}:
                continue
            rows.append(dict(premise=premise, hypothesis=hypothesis, label=label))

    df = pd.DataFrame(rows)

    # Normalise labels to EuroEval NLI convention
    label_mapping = {
        "true": "entailment",
        "false": "contradiction",
        "neutral": "neutral",
    }
    df["label"] = df["label"].map(label_mapping)

    # Build the combined `text` field used by the EuroEval evaluation framework
    df["text"] = "Udsagn 1: " + df["premise"] + "\nUdsagn 2: " + df["hypothesis"]

    # Keep only the columns needed for evaluation
    df = df[["text", "label"]].drop_duplicates().reset_index(drop=True)

    # Create splits. The dataset is small (318 samples after deduplication), so we
    # use all available samples across train / val / test.
    val_size = 64
    train_size = 64
    test_size = len(df) - val_size - train_size

    val_df = df.sample(n=val_size, random_state=4242)
    remaining_df = df.drop(val_df.index).reset_index(drop=True)

    test_df = remaining_df.sample(n=test_size, random_state=4242)
    train_df = remaining_df.drop(test_df.index).reset_index(drop=True)

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

    # Push the dataset to the Hugging Face Hub
    dataset_id = "EuroEval/danish-entailment"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
