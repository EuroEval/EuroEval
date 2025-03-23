# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==2.15.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the DANSK-mini NER dataset and upload it to the HF Hub."""

from dataclasses import dataclass

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError


@dataclass
class NamedEntity:
    """A named entity."""

    start: int
    end: int
    label: str


@dataclass
class Token:
    """A token."""

    id: int
    start: int
    end: int


def main() -> None:
    """Create the DANSK-mini NER dataset and uploads it to the HF Hub."""
    # Define dataset ID
    repo_id = "chcaa/DANSK"

    # Download the dataset
    dataset = load_dataset(path=repo_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # This mapping is taken from Tedeschi, Simone, et al. "WikiNEuRal: Combined neural
    # and knowledge-based silver data creation for multilingual NER." Findings of the
    # Association for Computational Linguistics: EMNLP 2021. 2021.
    ontonotes_tag_to_conll_tag_mapping: dict[str, str] = {
        "CARDINAL": "O",
        "DATE": "O",
        "EVENT": "MISC",
        "FACILITY": "LOC",
        "GPE": "LOC",
        "LANGUAGE": "MISC",
        "LAW": "O",
        "LOCATION": "LOC",
        "MONEY": "O",
        "NORP": "MISC",
        "ORDINAL": "O",
        "ORGANIZATION": "ORG",
        "PERCENT": "O",
        "PERSON": "PER",
        "PRODUCT": "MISC",
        "QUANTITY": "O",
        "TIME": "O",
        "WORK OF ART": "MISC",
    }

    def extract_tokens_and_ner_tags(sample: dict) -> dict:
        tokens = [Token(**token) for token in sample["tokens"]]
        named_entities = [
            NamedEntity(**named_entity) for named_entity in sample["ents"]
        ]

        token_list: list[str] = list()
        ner_tag_list: list[str] = list()
        in_named_entity = None
        for token in tokens:
            token_str = sample["text"][token.start : token.end]
            token_list.append(token_str)
            matched_named_entities = [
                named_entity
                for named_entity in named_entities
                if named_entity.start <= token.start and named_entity.end >= token.end
            ]
            if matched_named_entities:
                ontonotes_ner_tag = matched_named_entities[0].label
                ner_tag = ontonotes_tag_to_conll_tag_mapping[ontonotes_ner_tag]
                if ner_tag != "O":
                    ner_tag = (
                        f"I-{ner_tag}" if in_named_entity == ner_tag else f"B-{ner_tag}"
                    )
                    in_named_entity = ner_tag[2:]
                else:
                    in_named_entity = None
                ner_tag_list.append(ner_tag)
            else:
                ner_tag_list.append("O")
                in_named_entity = None

        # Sanity checks
        assert len(token_list) == len(ner_tag_list), (
            "The number of tokens and named entity tags are not equal."
        )
        invalid_i_ner_tags = [
            ner_tag
            for token_idx, ner_tag in enumerate(ner_tag_list)
            if ner_tag.startswith("I-")
            and ner_tag_list[token_idx - 1] not in {f"B-{ner_tag[2:]}", ner_tag}
        ]
        assert not invalid_i_ner_tags, (
            f"The following I- tags are invalid: {invalid_i_ner_tags}"
        )

        return dict(text=sample["text"], tokens=token_list, labels=ner_tag_list)

    dataset = dataset.map(extract_tokens_and_ner_tags, num_proc=1)

    # Convert the dataset to a dataframe
    train_df = dataset["train"].to_pandas()
    val_df = dataset["dev"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Drop all columns except for `tokens` and `ner_tags`
    columns_to_drop = [
        col for col in train_df.columns if col not in ["text", "tokens", "labels"]
    ]
    train_df.drop(columns=columns_to_drop, inplace=True)
    val_df.drop(columns=columns_to_drop, inplace=True)
    test_df.drop(columns=columns_to_drop, inplace=True)

    # Create validation split. Since most of the samples in the dataset has no tags, we
    # add a weight to the samples that has tags to ensure that dataset has a reasonable
    # amount of samples with tags.
    val_size = 256
    val_df = val_df.sample(
        n=val_size,
        random_state=4242,
        weights=[5.0 if len(set(labels)) > 1 else 1.0 for labels in val_df["labels"]],
    )

    # Create test split. We add weights to the sampling as with the validation split.
    test_size = 1024
    test_df = test_df.sample(
        n=test_size,
        random_state=4242,
        weights=[5.0 if len(set(labels)) > 1 else 1.0 for labels in test_df["labels"]],
    )

    # Create train split. We add weights to the sampling as with the validation split.
    train_size = 1024
    train_df = train_df.sample(
        n=train_size,
        random_state=4242,
        weights=[5.0 if len(set(labels)) > 1 else 1.0 for labels in train_df["labels"]],
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
    dataset_id = "EuroEval/dansk-mini"

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
