# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "requests==2.32.3",
# ]
# ///


"""Create a Dutch proverbs dataset from the GPT-NL proverb dataset."""

import random
from typing import Any

import datasets
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the Dutch proverbs dataset and upload it to the HF Hub."""
    # Define the base download URL
    source_dataset_id = "GPT-NL/dutch-proverbs"
    dataset_id_euroeval = "EuroEval/dutch-proverbs"

    # Set a seed
    random.seed(42)

    # Download the dataset
    dataset = datasets.load_dataset(source_dataset_id)

    # format the questions for the benchmark
    dataset = dataset.map(format, remove_columns=dataset["train"].column_names)

    # remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id_euroeval, repo_type="dataset", missing_ok=True)
    except HTTPError:
        pass

    dataset.push_to_hub(dataset_id_euroeval, private=True)


def format(row: dict[str, Any]) -> dict[str, str]:
    """Format the dataset rows into promptable questions.

    The question for the model is to evaluate which proverbs fits the
    given sentence best.

    Args:
        row:
            A row of the original dataset containing multiple columns

    Returns:
        A dict with the prepared question in `text` and the correct answer in `label`
    """
    text = f"Scenario: {row['scenario']}\n"
    text += "Welk spreekwoord past hier het beste bij?\n"

    # Avoid correct or incorrect proverb to always have the same label
    if random.random() < 0.5:
        option_a = row["correct_proverb"]
        option_b = row["incorrect_proverb"]
        label = "a"
    else:
        option_a = row["incorrect_proverb"]
        option_b = row["correct_proverb"]
        label = "b"

    text += f"a. {option_a}\n"
    text += f"b. {option_b}"

    return {"text": text, "label": label}


if __name__ == "__main__":
    main()
