# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "openai==1.66.5",
#     "pandas==2.2.0",
#     "pydantic==2.6.0",
#     "python-dotenv==1.0.1",
#     "requests==2.32.3",
#     "tqdm==4.67.1",
# ]
# ///

"""Create umimeto-qa knowledge dataset."""

import pandas as pd
from constants import CHOICES_MAPPING
from datasets import Dataset, DatasetDict, Split, concatenate_datasets, load_dataset
from huggingface_hub import HfApi


def main() -> None:
    """Create umimeto-qa knowledge dataset."""
    # Load umimeto dataset - test split only
    configs = [
        "biology",
        "chemistry",
        "czech",
        "history",
        "informatics",
        "math",
        "physics",
    ]
    datasets = []

    for config in configs:
        config_dataset_dict = load_dataset("CZLC/umimeto-qa", config, split="test")
        datasets.append(config_dataset_dict)

    # Merge all configs together
    dataset = concatenate_datasets(datasets)
    assert isinstance(dataset, Dataset)

    df = process_dataset(dataset=dataset)

    # Create splits
    train_size = 32
    val_size = 32

    train_df = df.sample(train_size, random_state=42)
    df = df.drop(train_df.index.tolist())

    val_df = df.sample(val_size, random_state=42)
    test_df = df.drop(val_df.index.tolist())

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
    dataset_id = "EuroEval/umimeto-qa"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def process_dataset(dataset: Dataset) -> pd.DataFrame:
    """Process the dataset.

    Args:
        dataset: The dataset.

    Returns:
        Processed DataFrame with text and label columns.
    """
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    texts: list[str] = []
    correct_labels: list[str] = []

    for _, row in df.iterrows():
        # Map correct_answer to label
        if row.correct_answer == "A":
            correct_label = "a"
        elif row.correct_answer == "B":
            correct_label = "b"
        else:
            raise ValueError(f"Unexpected correct_answer value: {row.correct_answer}.")

        text = f"{row.question}\n{CHOICES_MAPPING['cs']}:\na. {row.A}\nb. {row.B}"

        texts.append(text)
        correct_labels.append(correct_label)

    df = pd.DataFrame({"text": texts, "label": correct_labels})
    df = df.drop_duplicates(subset="text")
    return df


if __name__ == "__main__":
    main()
