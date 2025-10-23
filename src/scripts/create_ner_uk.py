# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Ukrainian NER dataset and upload it to the HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi


def main() -> None:
    """Create the Ukrainian NER dataset and upload it to the HF Hub."""
    # Define dataset ID
    repo_id = "benjamin/ner-uk"

    # Load the dataset
    dataset = load_dataset(path=repo_id)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to dataframes
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    # Add a `text` column by joining tokens
    train_df["text"] = train_df["tokens"].map(lambda tokens: " ".join(tokens))
    val_df["text"] = val_df["tokens"].map(lambda tokens: " ".join(tokens))
    test_df["text"] = test_df["tokens"].map(lambda tokens: " ".join(tokens))

    # Rename `ner_tags` to `labels`
    train_df.rename(columns={"ner_tags": "labels"}, inplace=True)
    val_df.rename(columns={"ner_tags": "labels"}, inplace=True)
    test_df.rename(columns={"ner_tags": "labels"}, inplace=True)

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
        lambda ner_tags: [ner_conversion_dict[ner_tag] for ner_tag in ner_tags]
    )
    val_df["labels"] = val_df["labels"].map(
        lambda ner_tags: [ner_conversion_dict[ner_tag] for ner_tag in ner_tags]
    )
    test_df["labels"] = test_df["labels"].map(
        lambda ner_tags: [ner_conversion_dict[ner_tag] for ner_tag in ner_tags]
    )

    column_to_keep = ["tokens", "labels"]
    train_df = train_df[column_to_keep]
    val_df = val_df[column_to_keep]
    test_df = test_df[column_to_keep]

    # Make splits
    train_size = 1024
    val_size = 256
    test_size = 2048

    # Calculate the number of samples to transfer from train to test
    current_test_size = len(test_df)
    extra_samples_needed = test_size - current_test_size

    # Sample additional from the train set
    extra_test_samples = train_df.sample(n=extra_samples_needed, random_state=4242)

    # Add these samples to the test set and remove them from the train set
    test_df = pd.concat([test_df, extra_test_samples]).reset_index(drop=True)
    train_df = train_df.drop(extra_test_samples.index).reset_index(drop=True)

    # Sample the desired sizes
    train_df = train_df.sample(n=train_size, random_state=4242)
    val_df = val_df.sample(n=val_size, random_state=4242)

    # Confirm the test size
    assert len(test_df) == test_size

    # Reset the index
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # Create the dataset dictionary after resetting the index
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/ner-uk-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
