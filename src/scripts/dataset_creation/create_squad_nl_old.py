# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the old SQuAD-nl-mini dataset and upload them to the HF Hub."""

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
    """Create the old SQuAD-nl-mini dataset and upload them to the HF Hub."""
    dataset_id = "yhavinga/squad_v2_dutch"

    # Load the dataset
    dataset = load_dataset(dataset_id, token=True)
    assert isinstance(dataset, DatasetDict)

    def add_answer_start(example: dict) -> dict:
        """Add character offsets for answers found in the context.

        Args:
            example (dict):
                Example with context and candidate answers.

        Returns:
            The example with unmatched answers removed and answer offsets added.
        """
        possible_answers: list[str] = example["answers"]["text"]
        if possible_answers:
            answers: list[str] = list()
            answer_starts: list[int] = list()
            for answer in possible_answers:
                answer_start = example["context"].find(answer)
                if answer_start >= 0:
                    answers.append(answer)
                    answer_starts.append(answer_start)
            example["answers"]["text"] = answers
            example["answers"]["answer_start"] = answer_starts
        else:
            example["answers"]["answer_start"] = []
        return example

    # Add `answer_start` key to `answers` column
    dataset = dataset.map(function=add_answer_start)

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
                Example containing answer start offsets.

        Returns:
            Whether the example has a valid answer offset.
        """
        return len(example["answer_start"]) > 0 and example["answer_start"][0] >= 0

    train_has_answer: pd.Series = train_df.answers.map(has_answer)
    valtest_has_answer: pd.Series = valtest_df.answers.map(has_answer)

    # Only work with the questions having answers in the context
    train_df_with_answer: pd.DataFrame = train_df.loc[train_has_answer]
    valtest_df_with_answer: pd.DataFrame = valtest_df.loc[valtest_has_answer]

    # Create validation split
    val_size = 256
    val_df = sample_dataframe(frame=valtest_df_with_answer, maximum=val_size)

    # Create test split
    test_size = 2048
    valtest_df_with_answer_filtered: pd.DataFrame = valtest_df_with_answer.loc[
        ~valtest_df_with_answer.index.isin(val_df.index)
    ]
    test_df = sample_dataframe(frame=valtest_df_with_answer_filtered, maximum=test_size)

    # Create train split
    train_size = 1024
    train_df = sample_dataframe(frame=train_df_with_answer, maximum=train_size)

    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = dataset_dict_from_frames(
        frames={"train": train_df, "val": val_df, "test": test_df},
        columns=list(train_df.columns),
    )

    # Push the dataset to the Hugging Face Hub
    dataset_id = "EuroEval/squad-nl-mini"
    preflight_private_upload(repository_id=dataset_id)
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
