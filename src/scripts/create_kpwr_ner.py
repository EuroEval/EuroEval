# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the KPWr NER dataset and upload it to the HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi


def main() -> None:
    """Create the KPWr NER dataset and uploads it to the HF Hub."""
    # Define dataset ID
    repo_id = "clarin-pl/kpwr-ner"

    # Download the dataset
    dataset = load_dataset(path=repo_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to dataframes
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Drop unnecessary columns (keep only tokens and ner)
    columns_to_drop = [col for col in train_df.columns if col not in ["tokens", "ner"]]
    train_df.drop(columns=columns_to_drop, inplace=True)
    test_df.drop(columns=columns_to_drop, inplace=True)

    # Add a `text` column by joining tokens
    train_df["text"] = train_df["tokens"].map(lambda tokens: " ".join(tokens))
    test_df["text"] = test_df["tokens"].map(lambda tokens: " ".join(tokens))

    # Rename `ner` to `labels`
    train_df.rename(columns={"ner": "labels"}, inplace=True)
    test_df.rename(columns={"ner": "labels"}, inplace=True)

    # Get label names and create mapping
    label_names = dataset["train"].features["ner"].feature.names
    label_mapping = create_label_mapping(label_names)

    # Convert the NER tags from IDs to strings using our mapping
    train_df["labels"] = train_df["labels"].map(
        lambda ner_tags: [label_mapping.get(ner_tag, "O") for ner_tag in ner_tags]
    )
    test_df["labels"] = test_df["labels"].map(
        lambda ner_tags: [label_mapping.get(ner_tag, "O") for ner_tag in ner_tags]
    )

    # Create validation split from train set
    val_size = 256
    val_df = train_df.sample(n=val_size, random_state=4242)

    # Update train split (remove validation samples)
    train_size = 1024
    remaining_train = train_df.drop(val_df.index)
    train_df = remaining_train.sample(
        n=min(train_size, len(remaining_train)), random_state=4242
    )

    # Create test split
    test_size = 2048
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
    dataset_id = "EuroEval/kpwr-ner"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def create_label_mapping(label_names: list[str]) -> dict[int, str]:
    """Create mapping from KPWr labels to standard BIO labels.

    Args:
        label_names: The list of label names.

    Returns:
        The mapping from KPWr labels to standard BIO labels.
    """
    mapping: dict[int, str] = {}

    for i, label_name in enumerate(label_names):
        if label_name == "O":
            mapping[i] = "O"
        elif "nam_liv" in label_name:
            # Living beings (persons, characters, etc.)
            if label_name.startswith("B-"):
                mapping[i] = "B-PER"
            elif label_name.startswith("I-"):
                mapping[i] = "I-PER"
        elif "nam_loc" in label_name:
            # Locations (cities, countries, etc.)
            if label_name.startswith("B-"):
                mapping[i] = "B-LOC"
            elif label_name.startswith("I-"):
                mapping[i] = "I-LOC"
        elif "nam_org" in label_name:
            # Organizations (companies, institutions, etc.)
            if label_name.startswith("B-"):
                mapping[i] = "B-ORG"
            elif label_name.startswith("I-"):
                mapping[i] = "I-ORG"
        else:
            # Everything else (events, products, etc.)
            if label_name.startswith("B-"):
                mapping[i] = "B-MISC"
            elif label_name.startswith("I-"):
                mapping[i] = "I-MISC"
            else:
                mapping[i] = "O"  # Fallback

    return mapping


if __name__ == "__main__":
    main()
