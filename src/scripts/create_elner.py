# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the elNER dataset and upload it to the HF Hub."""

import io
import tarfile
from collections import defaultdict

import pandas as pd
import requests
from datasets import Split
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from huggingface_hub.hf_api import HfApi


def main() -> None:
    """Create the elNER dataset and uploads it to the HF Hub."""
    # Define the download URL for the tar.gz file
    url = "https://github.com/nmpartzio/elNER/raw/master/dataset/elNER18/IOB2.tar.gz"

    # Download the dataset
    response = requests.get(url)
    response.raise_for_status()

    # Extract the tar.gz file
    tar_file = tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz")

    data = read_files_from_tar(tar_file=tar_file)

    dfs = parse_iob2(data=data)

    # Create splits
    train_size = 1024
    val_size = 256
    test_size = 2048

    val_df = dfs["val"].sample(n=val_size, random_state=4242)
    test_df = dfs["test"].sample(n=test_size, random_state=4242)
    train_df = dfs["train"].sample(n=train_size, random_state=4242)

    # Rename ner_tags to labels for consistency
    train_df = train_df.rename(columns={"ner_tags": "labels"}).reset_index(drop=True)
    val_df = val_df.rename(columns={"ner_tags": "labels"}).reset_index(drop=True)
    test_df = test_df.rename(columns={"ner_tags": "labels"}).reset_index(drop=True)

    # Convert the NER tags to standard format
    ner_conversion_dict = {
        # Keep O as is
        "O": "O",
        # Person entities
        "B-PERSON": "B-PER",
        "I-PERSON": "I-PER",
        # Organization entities
        "B-ORG": "B-ORG",
        "I-ORG": "I-ORG",
        # Location entities (including GPE and FAC)
        "B-LOC": "B-LOC",
        "I-LOC": "I-LOC",
        "B-GPE": "B-LOC",  # Geopolitical entity
        "I-GPE": "I-LOC",
        "B-FAC": "B-LOC",  # Facility
        "I-FAC": "I-LOC",
        # Miscellaneous entities
        "B-NORP": "B-MISC",  # Nationality, religion, political group
        "I-NORP": "I-MISC",
        "B-DATE": "O",
        "I-DATE": "O",
        "B-TIME": "O",
        "I-TIME": "O",
        "B-QUANTITY": "O",
        "I-QUANTITY": "O",
        "B-CARDINAL": "O",
        "I-CARDINAL": "O",
        "B-ORDINAL": "O",
        "B-PERCENT": "O",
        "B-MONEY": "O",
        "I-MONEY": "O",
        "B-EVENT": "B-MISC",
        "I-EVENT": "I-MISC",
        "B-WORK_OF_ART": "B-MISC",
        "I-WORK_OF_ART": "I-MISC",
        "B-PRODUCT": "B-MISC",
        "I-PRODUCT": "I-MISC",
        "B-LAW": "O",
        "I-LAW": "O",
        "B-LANGUAGE": "B-MISC",
    }

    # Apply the conversion to the labels
    train_df["labels"] = train_df["labels"].apply(
        lambda tags: [ner_conversion_dict.get(tag, "O") for tag in tags]
    )
    val_df["labels"] = val_df["labels"].apply(
        lambda tags: [ner_conversion_dict.get(tag, "O") for tag in tags]
    )
    test_df["labels"] = test_df["labels"].apply(
        lambda tags: [ner_conversion_dict.get(tag, "O") for tag in tags]
    )

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/elner-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def read_files_from_tar(tar_file: tarfile.TarFile) -> dict[str, list[str]]:
    """Read the files from the tar archive.

    Args:
        tar_file: The tar archive to read the files from.

    Returns:
        A dictionary with the split names as keys and the lines as values.
    """
    # Read the files from the tar archive
    # The structure needs to be checked - typically there are train/test/dev files
    file_members = tar_file.getmembers()

    # Extract and parse the IOB2 files
    # This is a placeholder - we need to see the actual structure first
    data = {}
    for member in file_members:
        if member.isfile() and not member.name.startswith("."):
            content = tar_file.extractfile(member)
            if content:
                text = content.read().decode("utf-8")
                # Determine split name from filename
                if "train" in member.name.lower():
                    split_name = "train"
                elif "test" in member.name.lower():
                    split_name = "test"
                elif "dev" in member.name.lower() or "val" in member.name.lower():
                    split_name = "val"
                else:
                    split_name = member.name

                data[split_name] = text.split("\n")

    return data


def parse_iob2(data: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    """Parse the IOB2 format.

    Args:
        data: A dictionary with the split names as keys and the lines as values.

    Returns:
        A dictionary with the split names as keys and the dataframes as values.
    """
    # Parse IOB2 format
    dfs = {}
    for split_name, lines in data.items():
        records = []
        data_dict: dict[str, list[str]] = defaultdict(list)

        for line in lines:
            line = line.strip()

            # Empty line indicates end of sentence
            if not line:
                if len(data_dict["tokens"]) > 0:
                    # Join tokens to create text
                    text = " ".join(data_dict["tokens"])
                    merged_data_dict = {
                        "tokens": data_dict["tokens"],
                        "ner_tags": data_dict["ner_tags"],
                        "text": text,
                    }
                    records.append(merged_data_dict)
                data_dict = defaultdict(list)
            else:
                # Parse token and tag (assuming space or tab separated)
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[0]
                    ner_tag = parts[1]
                    data_dict["tokens"].append(token)
                    data_dict["ner_tags"].append(ner_tag)

        # Don't forget the last sentence if file doesn't end with empty line
        if len(data_dict["tokens"]) > 0:
            text = " ".join(data_dict["tokens"])
            merged_data_dict = {
                "tokens": data_dict["tokens"],
                "ner_tags": data_dict["ner_tags"],
                "text": text,
            }
            records.append(merged_data_dict)

        df = pd.DataFrame.from_records(records)

        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        df = df.drop(columns=["text"])

        dfs[split_name] = df

    return dfs


if __name__ == "__main__":
    main()
