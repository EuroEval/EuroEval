# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the NorGLM NO-multi summarisation dataset."""

import pandas as pd
from constants import MAX_NUM_CHARS_IN_ARTICLE, MIN_NUM_CHARS_IN_ARTICLE
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the NorGLM NO-multi summarisation dataset and upload to HF Hub."""
    dataset_id = "NorGLM/NO-Multi-QA-Sum"

    dataset = load_dataset(dataset_id, split="train", token=True)
    assert isinstance(dataset, Dataset)

    dataset = dataset.rename_columns(
        column_mapping=dict(article="text", summary="target_text")
    )

    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Drop unneeded index column
    df.drop("Unnamed: 0", inplace=True, axis=1)

    # Drop non-article by index
    df.drop(index=359, inplace=True)

    # Shuffle and drop first duplicate
    df = df.sample(frac=1, random_state=4242).drop_duplicates(subset="text")

    # Reset the index
    df = df.reset_index(drop=True)

    # Only work with samples where the text is not very large or small
    lengths = df.text.str.len()
    lower_bound = MIN_NUM_CHARS_IN_ARTICLE
    upper_bound = MAX_NUM_CHARS_IN_ARTICLE
    df = df[lengths.between(lower_bound, upper_bound)]

    # Create validation split
    val_size = 64
    val_df = df.sample(n=val_size, random_state=4242)

    # Create test split
    test_size = 256
    filtered_df = df[~df.index.isin(val_df.index)]
    test_df = filtered_df.sample(n=test_size, random_state=4242)

    # Create train split
    train_df = filtered_df[~filtered_df.index.isin(test_df.index)]

    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/norglm-multi-sum"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api: HfApi = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
