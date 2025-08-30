# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create and upload the FullStack-LV-mini NER dataset from CoNLL-U format."""

import glob
import logging
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from requests import HTTPError

logger = logging.getLogger(__name__)


def main() -> None:
    """Create the FullStack-LV-mini NER dataset and upload it to the HF Hub."""
    # Clone the FullStack repository
    repo_path = clone_fullstack_repository()

    # Load and parse the data
    records = load_fullstack_data(repo_path=repo_path)

    # Convert to DataFrame
    df = pd.DataFrame.from_records(records)

    # Map the labels to the standard NER tags
    label_mapping = {
        "O": "O",
        "B-person": "B-PER",
        "I-person": "I-PER",
        "B-organization": "B-ORG",
        "I-organization": "I-ORG",
        "B-location": "B-LOC",
        "I-location": "I-LOC",
        "B-GPE": "B-LOC",  # Geopolitical entity -> Location
        "I-GPE": "I-LOC",
        "B-entity": "B-MISC",  # Generic entity -> Miscellaneous
        "I-entity": "I-MISC",
        "B-event": "B-MISC",  # Event -> Miscellaneous
        "I-event": "I-MISC",
        "B-product": "B-MISC",  # Product -> Miscellaneous
        "I-product": "I-MISC",
        "B-money": "B-MISC",  # Money -> Miscellaneous
        "I-money": "I-MISC",
        "B-time": "B-MISC",  # Time -> Miscellaneous
        "I-time": "I-MISC",
    }
    df["labels"] = df["labels"].map(
        lambda ner_tags: [label_mapping[ner_tag] for ner_tag in ner_tags]
    )

    # Create splits
    val_size = 256
    test_size = 2048
    train_size = 1024

    val_df = df.sample(n=val_size, random_state=4242)
    df_filtered = df[~df.index.isin(val_df.index)]

    test_df = df_filtered.sample(n=test_size, random_state=4242)
    df_filtered = df_filtered[~df_filtered.index.isin(test_df.index)]

    train_df = df_filtered.sample(n=train_size, random_state=4242)

    # Reset indices
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
    dataset_id = "EuroEval/fullstack-ner-lv-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)

    # Delete the repository
    delete_fullstack_repository(repo_path=repo_path)


def clone_fullstack_repository(repo_name: str = "FullStack") -> Path:
    """Clone the FullStack repository if it doesn't already exist.

    Args:
        repo_name (str): Name of the directory to clone into.

    Returns:
        Path: Path to the cloned repository.
    """
    if not Path(repo_name).exists():
        print("Cloning FullStack repository...")
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/LUMII-AILab/FullStack.git",
                    repo_name,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Successfully cloned repository")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone repository: {e.stderr}") from e

    else:
        logger.info(f"Repository '{repo_name}' already exists, using existing copy")

    return Path(repo_name)


def load_fullstack_data(repo_path: Path) -> List[Dict[str, Union[List[str], str]]]:
    """Load and parse all FullStack NER data from the specified repository path.

    Args:
        repo_path (Path): Path to the FullStack repository directory.

    Returns:
        List[Dict[str, Union[List[str], str]]]: A list of sentence records.
    """
    # Path to the data directory within the repository
    data_dir = repo_path / "NamedEntities" / "data"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory '{data_dir}' not found in the repository."
        )

    # Find all .conll2003 files in the directory
    conll_files = glob.glob(os.path.join(data_dir, "*.conll2003"))

    all_records = []

    for file_path in sorted(conll_files):
        # Read the file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse the CoNLL-U data
        records = parse_conllu_data(content)
        all_records.extend(records)

    return all_records


def parse_conllu_data(raw_data: str) -> List[Dict[str, Union[List[str], str]]]:
    """Parse CoNLL-U format data and return a list of sentence records.

    Args:
        raw_data (str): The raw data in CoNLL-U format.

    Returns:
        List[Dict[str, Union[List[str], str]]]: A list of sentence records.
    """
    records = []
    lines = raw_data.strip().split("\n")

    # Initialize data dictionary for current sentence
    data_dict: Dict[str, List[str]] = defaultdict(list)  # type: ignore[assignment]

    for line in lines:
        line = line.strip()

        # Skip comments (lines starting with #)
        if line.startswith("#"):
            continue

        # Empty line indicates end of sentence
        elif line == "":
            if len(data_dict["tokens"]) > 0:
                # Create record with tokens, labels, and text
                record = {
                    "tokens": data_dict["tokens"],
                    "labels": data_dict["labels"],
                    "text": " ".join(data_dict["tokens"]),
                }
                records.append(record)
            # Reset for next sentence
            data_dict = defaultdict(list)

        # Parse token line
        else:
            columns = line.split("\t")
            if len(columns) >= 7:  # Ensure we have enough columns
                token_id = columns[0]
                token = columns[1]
                # Use column 6 for NER tags (0-indexed, so columns[6] is column 7)
                ner_tag = columns[6] if len(columns) > 6 else "O"

                # Skip multi-word tokens (those with "-" in ID)
                if "-" not in token_id and "." not in token_id:
                    data_dict["tokens"].append(token)
                    data_dict["labels"].append(ner_tag)

    # Handle last sentence if data doesn't end with empty line
    if len(data_dict["tokens"]) > 0:
        record = {
            "tokens": data_dict["tokens"],
            "labels": data_dict["labels"],
            "text": " ".join(data_dict["tokens"]),
        }
        records.append(record)

    return records  # type: ignore[return-value]


def delete_fullstack_repository(repo_path: Path) -> None:
    """Delete the FullStack repository.

    Args:
        repo_path (Path): Path to the FullStack repository.
    """
    if repo_path.exists():
        shutil.rmtree(repo_path)


if __name__ == "__main__":
    main()
