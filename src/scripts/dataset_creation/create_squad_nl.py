# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the SQuAD-nl-mini dataset and upload them to the HF Hub."""

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
    """Create the SQuAD-nl-mini dataset and upload them to the HF Hub."""
    dataset_id = "GroNLP/squad-nl-v2.0"

    # Load the dataset
    dataset = load_dataset(dataset_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to a dataframe
    train_df = dataset["train"].to_pandas()
    val_df = dataset["dev"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    length_bounds = {
        "context": (MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT),
        "question": (MIN_NUM_CHARS_IN_QUESTION, MAX_NUM_CHARS_IN_QUESTION),
    }
    train_df = filter_by_character_lengths(frame=train_df, bounds=length_bounds)
    val_df = filter_by_character_lengths(frame=val_df, bounds=length_bounds)
    test_df = filter_by_character_lengths(frame=test_df, bounds=length_bounds)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Extract information on which examples contain an answer
    def has_answer(example: dict) -> bool:
        """Check whether an example contains an answer.

        Args:
            example (dict):
                Example containing answer start offsets.

        Returns:
            Whether the example has a valid answer offset.
        """
        return len(example["answer_start"]) > 0 and example["answer_start"][0] >= 0

    train_has_answer: pd.Series = train_df.answers.map(has_answer)
    val_has_answer: pd.Series = val_df.answers.map(has_answer)
    test_has_answer: pd.Series = test_df.answers.map(has_answer)

    # Only work with the questions having answers in the context
    train_df_with_answer: pd.DataFrame = train_df.loc[train_has_answer]
    val_df_with_answer: pd.DataFrame = val_df.loc[val_has_answer]
    test_df_with_answer: pd.DataFrame = test_df.loc[test_has_answer]

    # Create validation split
    val_size = 256
    val_df = val_df_with_answer.sample(n=val_size, random_state=4242)

    # Create train split
    train_size = 1024
    train_df = train_df_with_answer.sample(n=train_size, random_state=4242)

    # Create test split
    test_size = 2048
    test_df = pd.concat(
        [
            val_df_with_answer.loc[~val_df_with_answer.context.isin(val_df.context)],
            test_df_with_answer,
        ],
        ignore_index=True,
    ).sample(n=test_size, random_state=4242)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = dataset_dict_from_frames(
        frames={"train": train_df, "val": val_df, "test": test_df},
        columns=list(train_df.columns),
    )

    # Push the dataset to the Hugging Face Hub
    dataset_id = "EuroEval/squad-nl-v2-mini"
    preflight_private_upload(repository_id=dataset_id)
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
