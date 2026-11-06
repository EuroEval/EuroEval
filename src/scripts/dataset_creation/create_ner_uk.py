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

if __package__:
    from .dataset_utils import preflight_private_upload
else:
    from dataset_utils import preflight_private_upload

if __package__:
    from .dataset_processing import process_ner_frame, sample_dataframe
else:
    from dataset_processing import process_ner_frame, sample_dataframe


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

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    ner_mapping = {
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
    train_df = process_ner_frame(frame=train_df, label_mapping=ner_mapping)[
        ["tokens", "labels"]
    ]
    val_df = process_ner_frame(frame=val_df, label_mapping=ner_mapping)[
        ["tokens", "labels"]
    ]
    test_df = process_ner_frame(frame=test_df, label_mapping=ner_mapping)[
        ["tokens", "labels"]
    ]

    column_to_keep = ["tokens", "labels"]
    train_df = train_df[column_to_keep]
    val_df = val_df[column_to_keep]
    test_df = test_df[column_to_keep]

    # Make splits
    train_size = 1024
    val_size = 256
    test_size = 2048

    # Cap source splits independently; a smaller test split stays smaller.
    train_df = sample_dataframe(frame=train_df, maximum=train_size)
    val_df = sample_dataframe(frame=val_df, maximum=val_size)
    test_df = sample_dataframe(frame=test_df, maximum=test_size)

    # Reset the index
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Create the dataset dictionary after resetting the index
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df, split=Split.TEST),
        }
    )

    # Create dataset ID
    dataset_id = "EuroEval/ner-uk-mini"

    # Push the dataset to the Hugging Face Hub
    preflight_private_upload(repository_id=dataset_id)
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
