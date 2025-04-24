# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Schibsted summarisation dataset."""

import pandas as pd
from constants import MAX_NUM_CHARS_IN_ARTICLE, MIN_NUM_CHARS_IN_ARTICLE
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError

NEWSROOM_TO_LANGUAGE = {
    "sno-commercial": "no",
    "vektklubb": "no",
    "e24": "no",
    "e24partnerstudio": "no",
    "dinepenger": "no",
    "vgpartnerstudio": "no",
    "bt": "no",
    "tekno": "no",
    "vg": "no",
    "ap": "no",
    "randaberg24": "no",
    "ab": "sv",
    "sa": "no",
}


def main() -> None:
    """Create the Schibsted summarisation dataset and upload to HF Hub."""
    dataset_id = "Schibsted/schibsted-article-summaries"

    dataset = load_dataset(dataset_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Rename validation split to val
    dataset["val"] = dataset.pop("validation")

    # Rename article_text_all to text
    dataset = dataset.rename_columns(
        column_mapping=dict(article_text_all="text", summary="target_text")
    )

    # Add article title to text, separated by a colon
    dataset = dataset.map(lambda x: {"text": f"{x['article_title']}: {x['text']}"})

    # Summary is a list of strings, but should be a single string
    dataset = dataset.map(lambda x: {"target_text": " ".join(x["target_text"])})

    # Ignore samples outside the bounds
    train_df = dataset["train"].to_pandas()
    val_df = dataset["val"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Only work with samples where the text is not very large or small
    train_lengths = train_df.text.str.len()
    val_lengths = val_df.text.str.len()
    test_lengths = test_df.text.str.len()
    lower_bound = MIN_NUM_CHARS_IN_ARTICLE
    upper_bound = MAX_NUM_CHARS_IN_ARTICLE
    train_df = train_df[train_lengths.between(lower_bound, upper_bound)]
    val_df = val_df[val_lengths.between(lower_bound, upper_bound)]
    test_df = test_df[test_lengths.between(lower_bound, upper_bound)]

    for df in [train_df, val_df, test_df]:
        # make column language based on newsroom
        df["language"] = df["newsroom"].map(NEWSROOM_TO_LANGUAGE)

    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Dataset is a mix of Swedish and Norwegian articles.
    # Make two separate datasets, one for each language.
    dataset_sv = dataset.filter(lambda x: x["language"] == "sv")

    dataset_no = dataset.filter(lambda x: x["language"] == "no")

    # Dataset IDs.
    dataset_id_sv = "EuroEval/schibsted-article-summaries-sv"
    dataset_id_no = "EuroEval/schibsted-article-summaries-no"

    for dataset, dataset_id in [
        (dataset_sv, dataset_id_sv),
        (dataset_no, dataset_id_no),
    ]:
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
