# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
#     "requests==2.32.5",
# ]
# ///
"""Create the Estonian Trivia dataset and upload to HF Hub."""

from typing import MutableMapping

from constants import CHOICES_MAPPING
from datasets import DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi


def main() -> None:
    """Create the Estonian Trivia dataset and upload to HF Hub."""
    target_repo_id = "EuroEval/trivia-et"

    # start from the official source
    ds = load_dataset("TalTechNLP/trivia_et", split="train")

    ds = ds.map(add_options_and_label)

    # examples that are OK to show in the documentation with answers
    doc_examples_ids = [96, 97, 98]

    doc_examples = ds.filter(lambda row: row["id"] in doc_examples_ids)
    ds = ds.filter(lambda row: row["id"] not in doc_examples_ids)

    ds = ds.shuffle(seed=42)

    train_size = 240 - len(doc_examples)
    val_size = 60
    test_size = 500

    train_ds = concatenate_datasets([ds.select(range(train_size)), doc_examples])
    val_ds = ds.skip(train_size).select(range(val_size))
    test_ds = ds.skip(train_size + val_size).select(range(test_size))

    ds = DatasetDict({"train": train_ds, "val": val_ds, "test": test_ds})

    ds = ds.select_columns(["text", "label"])

    # Push the dataset to the Hugging Face Hub
    HfApi().delete_repo(target_repo_id, repo_type="dataset", missing_ok=True)
    ds.push_to_hub(target_repo_id, private=True)


def add_options_and_label(row: MutableMapping) -> MutableMapping:
    """Add options to the text and transform labels to letters."""
    letters = ["a", "b", "c", "d"]

    question = row["küsimus"]
    answer = row["vastus"]
    options = row["vastusevariandid"]

    options = [f"{letters[i]}. {options[i]}" for i in range(len(options))]
    options = "\n".join(options)

    text = f"{question}\n{CHOICES_MAPPING['et']}:\n{options}"
    label = letters[answer]

    return {"text": text, "label": label}


if __name__ == "__main__":
    main()
