# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==2.15.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Dutch CoLA dataset and upload it to the HF Hub."""

import pandas as pd
from constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT  # noqa
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the Dutch CoLA dataset and upload it to the HF Hub."""
    # Define the base download URL
    repo_id = "GroNLP/dutch-cola"

    # Download the dataset
    dataset = load_dataset(path=repo_id, token=True)
    assert isinstance(dataset, DatasetDict)

    dataset = (
        dataset.select_columns(["Sentence", "Acceptability"])
        .rename_columns({"Sentence": "text", "Acceptability": "label"})
        .shuffle(4242)
    )

    label_mapping = {0: "incorrect", 1: "correct"}
    dataset = dataset.map(lambda sample: {"label": label_mapping[sample["label"]]})
    dataset["val"] = dataset.pop("validation")

    full_dataset_id = "EuroEval/dutch-cola-full"
    dataset_id = "EuroEval/dutch-cola"

    # Remove the dataset from Hugging Face Hub if it already exists
    for repo_id in [dataset_id, full_dataset_id]:
        try:
            api = HfApi()
            api.delete_repo(repo_id, repo_type="dataset", missing_ok=True)
        except HTTPError:
            pass

    dataset.push_to_hub(full_dataset_id, private=True)

    # Convert the dataset to a dataframe
    train_df = dataset["train"].to_pandas()
    val_df = dataset["val"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Create validation split
    val_size = 256
    incorrect_val_df = val_df.query("label == 'incorrect'").sample(
        val_size // 2, random_state=4242
    )
    correct_val_df = val_df.query("label == 'correct'").sample(
        val_size // 2, random_state=4242
    )
    val_df = (
        pd.concat([incorrect_val_df, correct_val_df])
        .sample(frac=1.0, random_state=4242)
        .reset_index(drop=True)
    )

    # Create test split
    test_size = 2048
    incorrect_test_df = test_df.query("label == 'incorrect'").sample(
        test_size // 2, random_state=4242
    )
    correct_test_df = test_df.query("label == 'correct'").sample(
        test_size // 2, random_state=4242
    )
    test_df = (
        pd.concat([incorrect_test_df, correct_test_df])
        .sample(frac=1.0, random_state=4242)
        .reset_index(drop=True)
    )

    # Create train split
    train_size = 1024
    incorrect_train_df = train_df.query("label == 'incorrect'").sample(
        train_size // 2, random_state=4242
    )
    correct_train_df = train_df.query("label == 'correct'").sample(
        train_size // 2, random_state=4242
    )
    train_df = (
        pd.concat([incorrect_train_df, correct_train_df])
        .sample(frac=1.0, random_state=4242)
        .reset_index(drop=True)
    )

    # Reset the index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Only work with samples where the document is not very large or small
    # We do it after we have made the splits to ensure that the dataset is minimally
    # affected.
    new_train_df = train_df.copy()
    new_train_df["text_len"] = new_train_df.text.str.len()
    new_train_df = new_train_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )
    new_val_df = val_df.copy()
    new_val_df["text_len"] = new_val_df.text.str.len()
    new_val_df = new_val_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )
    new_test_df = test_df.copy()
    new_test_df["text_len"] = new_test_df.text.str.len()
    new_test_df = new_test_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(new_train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(new_val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(new_test_df, split=Split.TEST),
    )

    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
