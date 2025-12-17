# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "requests==2.32.3",
# ]
# ///


"""Create a Dutch common sense reasoning dataset based on the English COPA."""

import io
import os
import tarfile
import tempfile
import urllib.request
from typing import Any

import datasets
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the Dutch Copa dataset and upload it to the HF Hub."""
    # Define the base download URL
    source_url = (
        "https://github.com/wietsedv/NLP-NL/archive/refs/tags/copa-nl-v1.0.tar.gz"
    )
    dataset_id_euroeval = "EuroEval/copa-nl"

    # Download the dataset
    response = urllib.request.urlopen(source_url)
    tar_bytes = io.BytesIO(response.read())

    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
            tar.extractall(path=temp_dir)

        target_dir = os.path.join(temp_dir, "NLP-NL-copa-nl-v1.0", "COPA-NL")

        # HuggingFace Datasets can directly load the jsonl
        # train, test and dev files from disk
        dataset = datasets.load_dataset(target_dir)
        dataset["val"] = dataset.pop("validation")
        dataset = dataset.shuffle(4242)

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

    There are two different types of questions in the dataset: cause and effect.
    A slightly different prompt is created for both.

    Args:
        row:
            A row of the original dataset containing multiple columns

    Returns:
        A dict with the prepared question in `text` and the correct answer in `label`
    """
    text = f"Premisse: {row['premise']}\n"

    if row["question"] == "effect":
        text += "Wat is hier het logische gevolg van?\n"
    elif row["question"] == "cause":
        text += "Wat is hier de logische oorzaak van?\n"
    else:
        raise ValueError(f"Unknown question: {row['question']}")

    text += f"a. {row['choice1']}\n"
    text += f"b. {row['choice2']}"
    return {
        "text": text,
        "label": ["a", "b"][row["label"]],  # 0 -> "a", 1 -> "b"
    }


if __name__ == "__main__":
    main()
