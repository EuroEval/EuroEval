# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the FoNE-mini NER dataset and upload it to the HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the FoNE-mini NER dataset and uploads it to the HF Hub."""
    # Define dataset ID
    repo_id = "vesteinn/sosialurin-faroese-ner"

    # Download the dataset
    dataset = load_dataset(path=repo_id, token=True, split="train")
    assert isinstance(dataset, Dataset)

    # Convert the dataset to a dataframe
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Drop all columns except for `tokens` and `ner_tags`
    columns_to_drop = [col for col in df.columns if col not in ["tokens", "ner_tags"]]
    df.drop(columns=columns_to_drop, inplace=True)

    # Add a `text` column
    df["text"] = df["tokens"].map(lambda tokens: " ".join(tokens))

    # Rename `ner_tags` to `labels`
    df.rename(columns={"ner_tags": "labels"}, inplace=True)

    # Convert the NER tags from IDs to strings
    ner_conversion_dict = {
        0: "O",  # Original: B-Date
        1: "B-LOC",  # Original: B-Location
        2: "B-MISC",  # Original: B-Miscellaneous
        3: "O",  # Original: B-Money
        4: "B-ORG",  # Original: B-Organization
        5: "O",  # Original: B-Percent
        6: "B-PER",  # Original: B-Person
        7: "O",  # Original: B-Time
        8: "O",  # Original: I-Date
        9: "I-LOC",  # Original: I-Location
        10: "I-MISC",  # Original: I-Miscellaneous
        11: "O",  # Original: I-Money
        12: "I-ORG",  # Original: I-Organization
        13: "O",  # Original: I-Percent
        14: "I-PER",  # Original: I-Person
        15: "O",  # Original: I-Time
        16: "O",  # Original: O
    }
    df["labels"] = df["labels"].map(
        lambda ner_tags: [ner_conversion_dict[ner_tag] for ner_tag in ner_tags]
    )

    for token_list, ner_tag_list in zip(df["tokens"], df["labels"]):
        # Sanity check that the number of tokens and named entity tags are equal
        assert len(token_list) == len(ner_tag_list), (
            "The number of tokens and named entity tags are not equal."
        )

        # Fix invalid I-tags
        invalid_i_ner_tags = [
            ner_tag
            for token_idx, ner_tag in enumerate(ner_tag_list)
            if ner_tag.startswith("I-")
            and ner_tag_list[token_idx - 1] not in {f"B-{ner_tag[2:]}", ner_tag}
        ]
        while invalid_i_ner_tags:
            for invalid_i_ner_tag in invalid_i_ner_tags:
                ner_tag_list[ner_tag_list.index(invalid_i_ner_tag)] = (
                    f"B-{invalid_i_ner_tag[2:]}"
                )
            invalid_i_ner_tags = [
                ner_tag
                for token_idx, ner_tag in enumerate(ner_tag_list)
                if ner_tag.startswith("I-")
                and ner_tag_list[token_idx - 1] not in {f"B-{ner_tag[2:]}", ner_tag}
            ]

        # Sanity check that all I-tags are valid
        assert not invalid_i_ner_tags, (
            f"The following I- tags are invalid: {invalid_i_ner_tags}."
        )

    # Create validation split
    val_size = 256
    val_df = df.sample(
        n=val_size,
        random_state=4242,
        weights=[5.0 if len(set(labels)) > 1 else 1.0 for labels in df["labels"]],
    )

    # Create test split
    test_size = 2048
    filtered_df = df[~df.index.isin(val_df.index)]
    test_df = filtered_df.sample(
        n=test_size,
        random_state=4242,
        weights=[
            5.0 if len(set(labels)) > 1 else 1.0 for labels in filtered_df["labels"]
        ],
    )

    # Create train split
    train_size = 1024
    filtered_df = filtered_df[~filtered_df.index.isin(test_df.index)]
    train_df = filtered_df.sample(
        n=train_size,
        random_state=4242,
        weights=[
            5.0 if len(set(labels)) > 1 else 1.0 for labels in filtered_df["labels"]
        ],
    )

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
    dataset_id = "EuroEval/fone-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
