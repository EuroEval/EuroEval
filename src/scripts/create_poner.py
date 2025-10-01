# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the PONER NER dataset and upload it to the HF Hub."""

from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi


def main() -> None:
    """Create the PONER NER dataset and upload it to the HF Hub."""
    # Define dataset ID
    repo_id = "romanjanik/PONER"

    # Load the dataset
    dataset = load_dataset(path=repo_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to dataframes
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    val_df = dataset["dev"].to_pandas()

    # Process the dataframes
    columns_to_drop = [
        col for col in train_df.columns if col not in ["tokens", "ner_tags"]
    ]
    train_df.drop(columns=columns_to_drop, inplace=True)
    test_df.drop(columns=columns_to_drop, inplace=True)
    val_df.drop(columns=columns_to_drop, inplace=True)

    train_df["text"] = train_df["tokens"].map(lambda tokens: " ".join(tokens))
    test_df["text"] = test_df["tokens"].map(lambda tokens: " ".join(tokens))
    val_df["text"] = val_df["tokens"].map(lambda tokens: " ".join(tokens))

    train_df.rename(columns={"ner_tags": "labels"}, inplace=True)
    test_df.rename(columns={"ner_tags": "labels"}, inplace=True)
    val_df.rename(columns={"ner_tags": "labels"}, inplace=True)

    # Create label mapping
    label_names = dataset["train"].features["ner_tags"].feature.names
    label_mapping = create_label_mapping(label_names)

    # Convert the NER tags from IDs to strings using our mapping
    train_df["labels"] = train_df["labels"].map(
        lambda ner_tags: [label_mapping.get(ner_tag, "O") for ner_tag in ner_tags]
    )
    test_df["labels"] = test_df["labels"].map(
        lambda ner_tags: [label_mapping.get(ner_tag, "O") for ner_tag in ner_tags]
    )
    val_df["labels"] = val_df["labels"].map(
        lambda ner_tags: [label_mapping.get(ner_tag, "O") for ner_tag in ner_tags]
    )

    # Create splits
    train_size = 1024
    val_size = 256
    test_size = 2048

    train_df = train_df.sample(n=min(train_size, len(train_df)), random_state=4242)
    val_df = val_df.sample(n=min(val_size, len(val_df)), random_state=4242)
    test_df = test_df.sample(n=min(test_size, len(test_df)), random_state=4242)

    # Reset the index
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
    dataset_id = "EuroEval/poner-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def create_label_mapping(label_names: list[str]) -> dict[int, str]:
    """Create mapping from PONER labels to standard BIO labels.

    Args:
        label_names: The list of label names.

    Returns:
        The mapping from PONER labels to standard BIO labels.
    """
    # Define a base mapping for the label prefixes
    base_mapping = {
        "B-p": "B-PER",
        "I-p": "I-PER",
        "B-i": "B-ORG",
        "I-i": "I-ORG",
        "B-g": "B-LOC",
        "I-g": "I-LOC",
        "B-t": "B-MISC",
        "I-t": "I-MISC",
        "B-o": "B-MISC",
        "I-o": "I-MISC",
    }

    # Create the mapping using the base mapping
    mapping = {
        i: base_mapping.get(label_name, "O") for i, label_name in enumerate(label_names)
    }

    return mapping


if __name__ == "__main__":
    main()
