# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
#     "requests==2.32.5",
#     "pandas==2.3.1",
# ]
# ///

"""Create the ERRNews summarisation dataset."""

import os
import tempfile
import zipfile

import pandas as pd
import requests
from datasets import Dataset, DatasetDict, concatenate_datasets
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the ERRNews summarisation dataset."""
    source_url = "https://cs.taltech.ee/staff/heharm/ERRnews/data.zip"
    target_repo_id = "EuroEval/err-news-mini"

    file_map = {
        "train": "data/train.csv",
        "val": "data/val.csv",
        "test": "data/test.csv",
    }
    ds = DatasetDict()
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        response = requests.get(source_url)
        temp_file.write(response.content)
        temp_filename = temp_file.name

    with zipfile.ZipFile(temp_filename) as zip_file:
        for key, value in file_map.items():
            with zip_file.open(value) as csv_file:
                df = pd.read_csv(csv_file)
                ds[key] = Dataset.from_pandas(df)

    ds = ds.select_columns(["transcript", "summary"])
    ds = ds.rename_columns({"transcript": "text", "summary": "target_text"})

    train_size = 1024
    val_size = 256
    test_size = 2048

    new_ds = DatasetDict()

    # rearrange the examples
    new_ds["train"] = ds["train"].select(range(train_size))
    new_ds["val"] = ds["val"].select(range(val_size))
    new_ds["test"] = concatenate_datasets(
        [
            ds["test"],
            ds["train"].skip(train_size).select(range(test_size - len(ds["test"]))),
        ]
    )

    os.unlink(temp_filename)

    # remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(target_repo_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    new_ds.push_to_hub(target_repo_id, private=True)


if __name__ == "__main__":
    main()
