# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
#     "requests==2.32.5",
# ]
# ///

"""Create the Estonian NER dataset and upload to HF Hub."""

from typing import MutableMapping

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the Estonian NER dataset and upload to HF Hub."""
    target_repo_id = "EuroEval/estner-mini"

    # start from the official source
    ds = load_dataset("tartuNLP/EstNER")

    # target split sizes
    train_size = 1024
    val_size = 256
    test_size = 2048

    ds = ds.shuffle(seed=42)

    train_ds = ds["train"].select(range(train_size))
    val_ds = ds["dev"].select(range(val_size))
    test_ds = ds["test"].select(range(test_size))

    ds = DatasetDict(train=train_ds, val=val_ds, test=test_ds)

    ds = ds.rename_column("ner_tags", "labels")
    # a separate text column is not available
    ds = ds.map(lambda row: {"text": " ".join(row["tokens"])})
    ds = ds.select_columns(["text", "tokens", "labels"])
    # reduce the number of diverse labels by mapping to MISC
    ds = ds.map(convert_labels)

    # remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(target_repo_id, repo_type="dataset")
    except HTTPError:
        pass

    # push the dataset to the Hugging Face Hub
    ds.push_to_hub(target_repo_id, private=True)


def get_label_map() -> dict[str, str]:
    """Get the map from original labels to the EuroEval ones.

    Returns:
        The mapping.
    """
    original_labels = {
        "I-DATE",
        "I-PERCENT",
        "I-TITLE",
        "B-PERCENT",
        "B-LOC",
        "B-MONEY",
        "I-GPE",
        "B-ORG",
        "B-PROD",
        "I-LOC",
        "B-GPE",
        "B-TITLE",
        "I-EVENT",
        "I-MONEY",
        "I-TIME",
        "B-EVENT",
        "B-TIME",
        "I-PROD",
        "I-PER",
        "B-DATE",
        "B-PER",
        "I-ORG",
    }
    label_map = {}
    for label in original_labels:
        position, entity_type = label.split("-")
        if entity_type not in ("PER", "LOC", "ORG", "MISC"):
            if entity_type in ("DATE", "TIME", "PERCENT", "MONEY"):
                label_map[label] = "O"
            else:
                label_map[label] = f"{position}-{entity_type}"

    return label_map


def convert_labels(row: MutableMapping) -> MutableMapping:
    """Convert original labels in a row to new ones.

    Args:
        row:
            A row of the original dataset.

    Returns:
        The updated row, with the new labels.
    """
    label_map = get_label_map()
    row["labels"] = [label_map.get(label, label) for label in row["labels"]]
    return row


if __name__ == "__main__":
    main()
