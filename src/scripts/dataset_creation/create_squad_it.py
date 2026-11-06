# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the SQuAD-it-mini dataset and upload it to the HF Hub."""

import pandas as pd
from constants import (
    MAX_NUM_CHARS_IN_CONTEXT,
    MAX_NUM_CHARS_IN_QUESTION,
    MIN_NUM_CHARS_IN_CONTEXT,
    MIN_NUM_CHARS_IN_QUESTION,
)
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset

if __package__:
    from .dataset_utils import preflight_private_upload
else:
    from dataset_utils import preflight_private_upload

if __package__:
    from .dataset_processing import (
        dataset_dict_from_frames,
        filter_by_character_lengths,
        sample_dataframe,
    )
else:
    from dataset_processing import (
        dataset_dict_from_frames,
        filter_by_character_lengths,
        sample_dataframe,
    )


def main() -> None:
    """Create the SQuAD-it-mini dataset and upload it to the HF Hub."""
    dataset_id = "crux82/squad_it"

    # Load the dataset
    dataset = load_dataset(dataset_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to a dataframe
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    length_bounds = {
        "context": (MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT),
        "question": (MIN_NUM_CHARS_IN_QUESTION, MAX_NUM_CHARS_IN_QUESTION),
    }
    train_df = filter_by_character_lengths(frame=train_df, bounds=length_bounds)
    test_df = filter_by_character_lengths(frame=test_df, bounds=length_bounds)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Test split has repeat answer text and start indexes, so we just take the first
    test_df["answers"] = test_df.apply(
        lambda row: {
            "text": [row["answers"]["text"][0]],
            "answer_start": [row["answers"]["answer_start"][0]],
        },
        axis=1,
    )
    # Remove rows where the answer text is just a single character
    test_df = test_df[test_df["answers"].apply(lambda x: len(x["text"][0]) > 1)]

    # Create validation split
    val_size = 256
    val_df = sample_dataframe(frame=test_df, maximum=val_size)

    # Create train split
    train_size = 1024
    train_df = sample_dataframe(frame=train_df, maximum=train_size)

    # Create test split
    test_size = 2048
    test_df = sample_dataframe(
        frame=test_df.loc[~test_df.index.isin(val_df.index)], maximum=test_size
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Collect datasets in a dataset dictionary
    dataset = dataset_dict_from_frames(
        frames={"train": train_df, "val": val_df, "test": test_df},
        columns=list(train_df.columns),
    )

    # Push the dataset to the Hugging Face Hub
    dataset_id = "EuroEval/squad-it-mini"
    preflight_private_upload(repository_id=dataset_id)
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
