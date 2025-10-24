# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the BG-NER-BSNLP-mini NER dataset and upload it to the HF Hub."""

from ast import literal_eval

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi

NER_CONVERSION_DICT = {
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


def main() -> None:
    """Create the BG-NER-BSNLP-mini NER dataset and uploads it to the HF Hub."""
    # Define dataset ID
    repo_id = "usmiva/bg_ner_bsnlp"

    # Download the dataset
    dataset = load_dataset(path=repo_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to a dataframe
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    train_df = process(df=train_df)
    test_df = process(df=test_df)

    # Create validation split from train
    val_size = 256
    val_df = train_df.sample(n=val_size, random_state=4242)
    train_df = train_df.drop(val_df.index)

    # Create test split
    test_size = 2048
    test_df = test_df.sample(n=test_size, random_state=4242)

    # Create train split
    train_size = 1024
    train_df = train_df.sample(n=train_size, random_state=4242)

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
    dataset_id = "EuroEval/bg-ner-bsnlp-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe.

    Args:
        df:
            The dataframe to process.

    Returns:
        The processed dataframe.
    """
    df["ner_tags"] = df["ner_tags"].apply(literal_eval)
    df["tokens"] = df["tokens"].apply(literal_eval)

    assert all(
        len(ner_tags) == len(tokens)
        for ner_tags, tokens in zip(df["ner_tags"], df["tokens"])
    ), "The length of `ner_tags` and `tokens` are not equal in each row."

    df.rename(columns={"ner_tags": "labels"}, inplace=True)
    df["labels"] = df["labels"].map(
        lambda ner_tags: [NER_CONVERSION_DICT.get(ner_tag, "O") for ner_tag in ner_tags]
    )
    return df


if __name__ == "__main__":
    main()
