# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the icesum summarisation dataset."""

import io
import json
from zipfile import ZipFile

import pandas as pd
import requests
from constants import MAX_NUM_CHARS_IN_ARTICLE, MIN_NUM_CHARS_IN_ARTICLE
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the icesum summarisation dataset and upload to HF Hub."""
    # Fetch data from their repository
    url = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/285/icesum.zip"
    response = requests.get(url)

    # Unzip and load json
    input_zip = ZipFile(io.BytesIO(response.content))
    json_files = {
        name.split("/")[1]: input_zip.read(name)
        for name in input_zip.namelist()
        if name.endswith(".json")
    }

    # Ignore unused variable error since it is being used but in the queries below
    splits = json.loads(json_files["splits.json"].decode("utf-8"))  # noqa: F841

    dataset = json.loads(json_files["icesum.json"].decode("utf-8"))
    df = pd.DataFrame(dataset).T.reset_index()
    assert isinstance(df, pd.DataFrame)

    df.rename(columns={"summary": "target_text"}, inplace=True)

    df.dropna(subset=["text", "target_text"], inplace=True)

    # Only work with samples where the text is not very large or small
    lengths = df.text.str.len()
    lower_bound = MIN_NUM_CHARS_IN_ARTICLE
    upper_bound = MAX_NUM_CHARS_IN_ARTICLE
    df = df[lengths.between(lower_bound, upper_bound)]

    # Create validation split
    val_df = df.query("index in @splits['valid']")
    assert isinstance(val_df, pd.DataFrame)

    # Create test split
    test_df = df.query("index in @splits['test']")
    assert isinstance(test_df, pd.DataFrame)

    # Create train split
    train_df = df.query("index in @splits['train']")
    assert isinstance(train_df, pd.DataFrame)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/icesum"

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
