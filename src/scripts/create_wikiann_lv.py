# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the WikiANN-LV-mini NER dataset and upload it to the HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the WikiANN-LV-mini NER dataset and uploads it to the HF Hub."""
    # Load the WikiANN dataset for Latvian
    dataset_id = "unimelb-nlp/wikiann"

    # Load only the Latvian subset
    dataset = load_dataset(dataset_id, "lv", token=True)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to a dataframe
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Drop all columns except for `tokens` and `ner_tags`
    columns_to_drop = [
        col for col in train_df.columns if col not in ["tokens", "ner_tags"]
    ]
    train_df.drop(columns=columns_to_drop, inplace=True)
    val_df.drop(columns=columns_to_drop, inplace=True)
    test_df.drop(columns=columns_to_drop, inplace=True)

    # Add a `text` column
    train_df["text"] = train_df["tokens"].map(lambda tokens: " ".join(tokens))
    val_df["text"] = val_df["tokens"].map(lambda tokens: " ".join(tokens))
    test_df["text"] = test_df["tokens"].map(lambda tokens: " ".join(tokens))

    # Rename `ner_tags` to `labels`
    train_df.rename(columns={"ner_tags": "labels"}, inplace=True)
    val_df.rename(columns={"ner_tags": "labels"}, inplace=True)
    test_df.rename(columns={"ner_tags": "labels"}, inplace=True)

    # Convert the NER tags from IDs to strings
    ner_conversion_dict = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
        7: "B-MISC",
        8: "I-MISC",
    }

    train_df["labels"] = train_df["labels"].map(
        lambda ner_tags: [ner_conversion_dict.get(ner_tag, "O") for ner_tag in ner_tags]
    )
    val_df["labels"] = val_df["labels"].map(
        lambda ner_tags: [ner_conversion_dict.get(ner_tag, "O") for ner_tag in ner_tags]
    )
    test_df["labels"] = test_df["labels"].map(
        lambda ner_tags: [ner_conversion_dict.get(ner_tag, "O") for ner_tag in ner_tags]
    )

    # Create validation split
    val_size = 256
    val_df = val_df.sample(n=min(val_size, len(val_df)), random_state=4242)

    # Create test split
    test_size = 2048
    test_df = test_df.sample(n=min(test_size, len(test_df)), random_state=4242)

    # Create train split
    train_size = 1024
    train_df = train_df.sample(n=min(train_size, len(train_df)), random_state=4242)

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
    dataset_id = "EuroEval/wikiann-lv-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
