# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==2.15.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the MLQA dataset and upload them to the HF Hub."""

import pandas as pd
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from huggingface_hub.hf_api import HfApi
from requests.exceptions import HTTPError


def main() -> None:
    """Create the MLQA datasets and upload them to the HF Hub."""
    dataset_id = "facebook/mlqa"

    # Load the dataset
    dataset = load_dataset(dataset_id, token=True, name="mlqa.es.es")
    assert isinstance(dataset, DatasetDict)

    # The dataset only has splits val (500 samples) and test (5253 samples)
    # So we use the val split + 524 test samples for training
    # Then split the remaining test set into validation
    # (256 samples) and test (2048 samples)
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    # Create training split from validation data and random sample of test data
    train_test_sample = test_df.sample(n=524, random_state=42)
    train_df = pd.concat([val_df, train_test_sample])
    test_df = test_df.drop(train_test_sample.index)

    # Create validation split from remaining test data
    val_df = test_df.sample(n=256, random_state=42)
    test_df = test_df.drop(val_df.index)

    # Create test split with 2048 samples
    test_df = test_df.sample(n=2024, random_state=42)

    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    mini_dataset_id = "EuroEval/mlqa-es"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api: HfApi = HfApi()
        api.delete_repo(mini_dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(mini_dataset_id, private=True)


if __name__ == "__main__":
    main()
