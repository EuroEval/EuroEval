# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the SQuAD-mini datasets and upload them to the HF Hub."""

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
    from .dataset_utils import filter_by_character_lengths, preflight_private_upload
else:
    from dataset_utils import filter_by_character_lengths, preflight_private_upload

if __package__:
    from .dataset_processing import dataset_dict_from_frames
else:
    from dataset_processing import dataset_dict_from_frames


def main() -> None:
    """Create the SQuAD-mini datasets and upload them to the HF Hub."""
    dataset_id = "squad_v2"

    # Load the dataset
    dataset = load_dataset(dataset_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to a dataframe
    train_df = dataset["train"].to_pandas()
    valtest_df = dataset["validation"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(valtest_df, pd.DataFrame)

    length_bounds = {
        "context": (MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT),
        "question": (MIN_NUM_CHARS_IN_QUESTION, MAX_NUM_CHARS_IN_QUESTION),
    }
    train_df = filter_by_character_lengths(frame=train_df, bounds=length_bounds)
    valtest_df = filter_by_character_lengths(frame=valtest_df, bounds=length_bounds)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(valtest_df, pd.DataFrame)

    # Extract information on which examples contain an answer
    def has_answer(example: dict) -> bool:
        """Check whether an example contains an answer.

        Args:
            example (dict):
                Example containing candidate answer text.

        Returns:
            Whether the example has a non-empty answer.
        """
        return len(example["text"]) > 0 and example["text"][0] != ""

    train_has_answer: pd.Series = train_df.answers.map(has_answer)
    valtest_has_answer: pd.Series = valtest_df.answers.map(has_answer)

    # Only work with the questions having answers in the context
    train_df_with_answer: pd.DataFrame = train_df.loc[train_has_answer]
    valtest_df_with_answer: pd.DataFrame = valtest_df.loc[valtest_has_answer]

    # Create validation split
    val_size = 256
    val_df = valtest_df_with_answer.sample(n=val_size, random_state=4242)

    # Create test split
    test_size = 2048
    valtest_df_with_answer_filtered: pd.DataFrame = valtest_df_with_answer.loc[
        ~valtest_df_with_answer.index.isin(val_df.index)
    ]
    test_df = valtest_df_with_answer_filtered.sample(n=test_size, random_state=4242)

    # Create train split
    train_size = 1024
    train_df = train_df_with_answer.sample(n=train_size, random_state=4242)

    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Collect datasets in a dataset dictionary
    dataset = dataset_dict_from_frames(
        frames={"train": train_df, "val": val_df, "test": test_df},
        columns=list(train_df.columns),
    )

    # Push the dataset to the Hugging Face Hub
    dataset_id = "EuroEval/squad-mini"
    preflight_private_upload(repository_id=dataset_id)
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
