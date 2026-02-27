# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Schibsted front-page title and SEO title datasets."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi

from .constants import MAX_NUM_CHARS_IN_ARTICLE, MIN_NUM_CHARS_IN_ARTICLE

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


def process_title_dataset(
    source_dataset_id: str,
    title_column: str,
    euroeval_dataset_id_sv: str,
    euroeval_dataset_id_no: str,
) -> None:
    """Process a Schibsted title dataset and upload splits to HF Hub.

    Args:
        source_dataset_id:
            The Hugging Face dataset ID of the source dataset.
        title_column:
            The column name containing the title in the source dataset.
        euroeval_dataset_id_sv:
            The Hugging Face dataset ID for the Swedish EuroEval dataset.
        euroeval_dataset_id_no:
            The Hugging Face dataset ID for the Norwegian EuroEval dataset.
    """
    dataset = load_dataset(source_dataset_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Rename validation split to val
    if "validation" in dataset:
        dataset["val"] = dataset.pop("validation")

    # Rename article text and title columns to text and target_text
    dataset = dataset.rename_columns(
        column_mapping=dict(article_text_all="text", **{title_column: "target_text"})
    )

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

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    for df in [train_df, val_df, test_df]:
        # make column language based on newsroom
        df["language"] = df["newsroom"].map(NEWSROOM_TO_LANGUAGE)

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df, split=Split.TEST),
        }
    )

    # Dataset is a mix of Swedish and Norwegian articles.
    # Make two separate datasets, one for each language.
    dataset_sv = dataset.filter(lambda x: x["language"] == "sv")
    dataset_no = dataset.filter(lambda x: x["language"] == "no")

    for split_dataset, dataset_id in [
        (dataset_sv, euroeval_dataset_id_sv),
        (dataset_no, euroeval_dataset_id_no),
    ]:
        # Push the dataset to the Hugging Face Hub
        HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
        split_dataset.push_to_hub(dataset_id, private=True)


def main() -> None:
    """Create the Schibsted title datasets and upload to HF Hub."""
    process_title_dataset(
        source_dataset_id="Schibsted/schibsted-front-page-titles",
        title_column="front_title",
        euroeval_dataset_id_sv="EuroEval/schibsted-front-page-titles-sv",
        euroeval_dataset_id_no="EuroEval/schibsted-front-page-titles-no",
    )

    process_title_dataset(
        source_dataset_id="Schibsted/schibsted-seo-titles",
        title_column="seo_title",
        euroeval_dataset_id_sv="EuroEval/schibsted-seo-titles-sv",
        euroeval_dataset_id_no="EuroEval/schibsted-seo-titles-no",
    )


if __name__ == "__main__":
    main()
