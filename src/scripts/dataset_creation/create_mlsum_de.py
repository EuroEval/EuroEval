# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the German MLSum-de-mini summarisation dataset."""

import pandas as pd
from constants import MAX_NUM_CHARS_IN_ARTICLE, MIN_NUM_CHARS_IN_ARTICLE
from datasets import Dataset, DatasetDict, Split, load_dataset

if __package__:
    from .dataset_utils import preflight_private_upload
else:
    from dataset_utils import preflight_private_upload

if __package__:
    from .dataset_processing import filter_by_character_lengths, process_source_splits
else:
    from dataset_processing import filter_by_character_lengths, process_source_splits


def main() -> None:
    """Create the German MLSum-mini summarisation dataset and upload to HF Hub."""
    dataset_id = "GEM/mlsum"

    dataset = load_dataset(dataset_id, "de", token=True)
    assert isinstance(dataset, DatasetDict)

    dataset = dataset.rename_columns(column_mapping=dict(target="target_text"))

    # Remove the references column since the dataset has mismatched types in this column
    dataset = dataset.remove_columns(column_names=["references"])

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Only work with samples where the text is not very large or small
    length_bounds = {"text": (MIN_NUM_CHARS_IN_ARTICLE, MAX_NUM_CHARS_IN_ARTICLE)}
    train_df = filter_by_character_lengths(frame=train_df, bounds=length_bounds)
    val_df = filter_by_character_lengths(frame=val_df, bounds=length_bounds)
    test_df = filter_by_character_lengths(frame=test_df, bounds=length_bounds)

    frames = process_source_splits(
        frames={"train": train_df, "val": val_df, "test": test_df},
        limits={"train": 1024, "val": 256, "test": 2048},
    )
    train_df, val_df, test_df = frames["train"], frames["val"], frames["test"]

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df, split=Split.TEST),
        }
    )

    # Create dataset ID
    mini_dataset_id = "EuroEval/mlsum-de-mini"

    # Push the dataset to the Hugging Face Hub
    preflight_private_upload(repository_id=mini_dataset_id)
    dataset.push_to_hub(mini_dataset_id, private=True)


if __name__ == "__main__":
    main()
