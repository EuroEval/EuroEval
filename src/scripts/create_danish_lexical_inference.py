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

"""Create the Danish Lexical Inference dataset and upload it to the HF Hub."""

import io
from zipfile import ZipFile

import pandas as pd
import requests as rq
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


def main() -> None:
    """Create the Danish Lexical Inference dataset and upload it to the HF Hub."""
    # Download the repository ZIP archive
    url = (
        "https://github.com/kuhumcst/danish-semantic-reasoning-benchmark"
        "/archive/refs/heads/main.zip"
    )
    response = rq.get(url=url)
    response.raise_for_status()

    # Parse all the inference text files from the password-protected sub-archives
    records: list[dict[str, str]] = []
    num_skipped = 0
    with ZipFile(file=io.BytesIO(initial_bytes=response.content)) as outer_zip:
        role_names = ["AGENTIVE", "CONSTITUTIVE", "FORMAL", "TELIC"]
        for role in role_names:
            zip_path = (
                f"danish-semantic-reasoning-benchmark-main/inference/{role}.zip"
            )
            inner_zip_data = outer_zip.read(zip_path)
            with ZipFile(file=io.BytesIO(initial_bytes=inner_zip_data)) as inner_zip:
                inner_zip.setpassword(b"benchmark")
                for name in inner_zip.namelist():
                    if not name.endswith(".txt"):
                        continue
                    content = inner_zip.read(name).decode("utf-8")
                    for line in content.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        record = parse_line(line=line)
                        if record is not None:
                            records.append(record)
                        else:
                            num_skipped += 1

    if num_skipped > 0:
        print(f"Warning: skipped {num_skipped} malformed lines during parsing.")

    df = pd.DataFrame(records)

    # Remove duplicates and reset the index
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    expected_num_samples = 1020
    assert len(df) == expected_num_samples, (
        f"Expected {expected_num_samples} samples after deduplication, "
        f"but got {len(df)}. The upstream dataset may have changed."
    )

    # Create validation split
    val_size = 64
    traintest_arr, val_arr = train_test_split(
        df, test_size=val_size, random_state=4242, stratify=df["label"]
    )
    traintest_df = pd.DataFrame(traintest_arr, columns=df.columns)
    val_df = pd.DataFrame(val_arr, columns=df.columns)

    # Create train and test splits
    train_size = 128
    train_arr, test_arr = train_test_split(
        traintest_df,
        train_size=train_size,
        random_state=4242,
        stratify=traintest_df["label"],
    )
    train_df = pd.DataFrame(train_arr, columns=df.columns)
    test_df = pd.DataFrame(test_arr, columns=df.columns)

    # Reset the index
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
    dataset_id = "EuroEval/danish-lexical-inference"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def parse_line(line: str) -> dict[str, str] | None:
    """Parse a single line from an inference text file.

    The files have three different formats:
    - Standard (4 cols): context TAB target TAB label TAB relation
    - Empty-col (4 cols): context TAB TAB target TAB label
    - Merged-col (3 cols): context TAB target TAB 'label relation'

    Args:
        line:
            A single line from a data file.

    Returns:
        A dict with keys 'text' and 'label', or None if the line is malformed.
    """
    cols = line.split("\t")

    if len(cols) >= 4 and cols[1] == "":
        # Empty-col format: context, empty, target, label
        context = cols[0].strip()
        target = cols[2].strip()
        label_str = cols[3].strip().lower()
    elif len(cols) == 3:
        # Merged-col format: context, target, 'label relation'
        context = cols[0].strip()
        target = cols[1].strip()
        label_str = cols[2].strip().lower().split()[0]
    elif len(cols) >= 4:
        # Standard format: context, target, label, relation
        context = cols[0].strip()
        target = cols[1].strip()
        label_str = cols[2].strip().lower()
    else:
        return None

    if label_str.startswith("true"):
        label = "correct"
    elif label_str.startswith("false"):
        label = "incorrect"
    else:
        return None

    if not context or not target:
        return None

    # Combine context and target into a single text field
    text = context.rstrip(".") + ". " + target[0].upper() + target[1:]

    return {"text": text, "label": label}


if __name__ == "__main__":
    main()
