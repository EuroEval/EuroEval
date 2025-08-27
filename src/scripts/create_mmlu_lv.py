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

"""Create the MMLU-LV (Latvian) dataset from VTI-Data and upload to HF Hub."""

from collections import Counter
from typing import Any, Dict, List

import pandas as pd
import requests
from constants import (
    MAX_NUM_CHARS_IN_INSTRUCTION,
    MAX_NUM_CHARS_IN_OPTION,
    MAX_REPETITIONS,
    MIN_NUM_CHARS_IN_INSTRUCTION,
    MIN_NUM_CHARS_IN_OPTION,
)
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from requests import HTTPError
from sklearn.model_selection import train_test_split


def main() -> None:
    """Create the MMLU-LV dataset and upload to HF Hub."""
    # Get all subject files
    subjects = get_mmlu_subjects_from_github()
    assert len(subjects) == 57, f"Expected 57 subjects, got {len(subjects)}"

    # Download and process all subjects
    all_data = []
    for subject_info in subjects:
        subject_data = download_subject_data(subject_info)
        if subject_data:
            all_data.extend(subject_data)

    assert len(all_data) == 13941, f"Expected 13941 questions, got {len(all_data)}"

    # Process the data
    df = process_mmlu_data(all_data)

    if df.empty:
        print("No valid data after processing. Exiting.")
        return

    # Apply filtering from the original script
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

    # Remove overly repetitive samples
    df = df[
        ~df.instruction.apply(is_repetitive)
        & ~df.option_a.apply(is_repetitive)
        & ~df.option_b.apply(is_repetitive)
        & ~df.option_c.apply(is_repetitive)
        & ~df.option_d.apply(is_repetitive)
    ]

    # Create the text column with choices in Latvian
    choices_word = "Izvēles"  # "Choices" in Latvian
    df["text"] = [
        row.instruction.replace("\n", " ").strip() + "\n"
        f"{choices_word}:\n"
        "a. " + row.option_a.replace("\n", " ").strip() + "\n"
        "b. " + row.option_b.replace("\n", " ").strip() + "\n"
        "c. " + row.option_c.replace("\n", " ").strip() + "\n"
        "d. " + row.option_d.replace("\n", " ").strip()
        for _, row in df.iterrows()
    ]

    # Ensure label consistency
    df.label = df.label.str.lower()

    # Only keep the required columns
    df = df[["text", "label", "category"]]

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create validation split
    val_size = 256
    traintest_arr, val_arr = train_test_split(
        df,
        test_size=val_size,
        random_state=4242,
        stratify=df.category if df.category.nunique() > 1 else None,
    )
    traintest_df = pd.DataFrame(traintest_arr, columns=df.columns)
    val_df = pd.DataFrame(val_arr, columns=df.columns)

    # Create test split
    test_size = 2048
    train_arr, test_arr = train_test_split(
        traintest_df,
        test_size=test_size,
        random_state=4242,
        stratify=traintest_df.category if traintest_df.category.nunique() > 1 else None,
    )
    train_df = pd.DataFrame(train_arr, columns=df.columns)
    test_df = pd.DataFrame(test_arr, columns=df.columns)

    # Create train split
    train_size = 1024
    train_df = train_df.sample(train_size, random_state=4242)

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Create dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/mmlu-lv-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def get_mmlu_subjects_from_github() -> List[Dict[str, str]]:
    """Get all MMLU subject files from the VTI-Data repository.

    Returns:
        List of dictionaries with subject names and download URLs.
    """
    api_url = "https://api.github.com/repos/LUMII-AILab/VTI-Data/contents/mmlu"

    response = requests.get(api_url)
    response.raise_for_status()
    contents = response.json()

    # Extract JSON files and their download URLs
    json_files = [
        {
            "name": item["name"][:-5],  # Remove .json extension
            "download_url": item["download_url"],
        }
        for item in contents
        if item["type"] == "file"
        and item["name"].endswith(".json")
        and item["name"] != "LICENSE"  # Skip license file
    ]

    # Filter out the regular sociology subject, keeping only sociology_postedited
    json_files = [f for f in json_files if f["name"] != "sociology"]

    return sorted(json_files, key=lambda x: x["name"])


def download_subject_data(subject_info: Dict[str, str]) -> List[Dict[str, Any]]:
    """Download and parse data for a specific MMLU subject.

    Args:
        subject_info: Dictionary with subject name and download URL

    Returns:
        List of question dictionaries
    """
    response = requests.get(subject_info["download_url"])
    response.raise_for_status()
    data = response.json()

    # Add subject category to each question
    for item in data:
        item["category"] = subject_info["name"]

    return data


def process_mmlu_data(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Process raw MMLU data into the expected format.

    Args:
        data: List of raw question dictionaries

    Returns:
        Processed DataFrame with columns: instruction, option_a,
            option_b, option_c, option_d, label, category
    """
    processed_data = []

    for item in data:
        # Handle different possible data structures
        question = item.get("question", item.get("instruction", ""))
        choices = item.get("choices", item.get("options", []))
        answer = item.get("answer", item.get("correct_answer", ""))
        category = item.get("category", "unknown")

        # Ensure we have exactly 4 choices
        if len(choices) != 4:
            continue

        # Convert answer to lowercase letter format if it's numeric
        if isinstance(answer, int):
            answer = ["a", "b", "c", "d"][answer]
        elif isinstance(answer, str) and answer.isdigit():
            answer = ["a", "b", "c", "d"][int(answer)]
        elif isinstance(answer, str):
            answer = answer.lower().strip()

        processed_item = {
            "instruction": question,
            "option_a": choices[0],
            "option_b": choices[1],
            "option_c": choices[2],
            "option_d": choices[3],
            "label": answer,
            "category": category,
        }

        processed_data.append(processed_item)

    return pd.DataFrame(processed_data)


def is_repetitive(text: str) -> bool:
    """Return True if the text is repetitive.

    Args:
        text (str): The text to check.

    Returns:
        bool: True if the text is repetitive, False otherwise.
    """
    if not isinstance(text, str):
        return False
    max_repetitions = max(Counter(text.split()).values()) if text.split() else 0
    return max_repetitions > MAX_REPETITIONS


if __name__ == "__main__":
    main()
