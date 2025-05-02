# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Finnish Turku NER dataset and upload it to the HF Hub."""

import pandas as pd
import requests
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from requests import HTTPError

LABEL_MAPS = {
    "B-PRO": "B-MISC",
    "I-PRO": "I-MISC",
    "B-DATE": "B-MISC",
    "I-DATE": "I-MISC",
    "I-EVENT": "I-MISC",
    "B-EVENT": "B-MISC",
}


def main() -> None:
    """Create the Finnish Turku NER dataset and uploads it to the HF Hub."""
    # Read the CoNLL TSV files
    dfs = read_dataset()
    train_df = dfs["train"]
    val_df = dfs["val"]
    test_df = dfs["test"]

    # Add a `text` column
    train_df["text"] = train_df["tokens"].map(lambda tokens: " ".join(tokens))
    val_df["text"] = val_df["tokens"].map(lambda tokens: " ".join(tokens))
    test_df["text"] = test_df["tokens"].map(lambda tokens: " ".join(tokens))

    # Create validation split
    val_size = 256
    val_df = val_df.sample(n=min(val_size, len(val_df)), random_state=4242)

    # Create test split
    test_size = 2048
    test_df = test_df.sample(n=min(test_size, len(test_df)), random_state=4242)

    # Create train split
    train_size = 1024
    train_df = train_df.sample(n=min(train_size, len(train_df)), random_state=4242)

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

    mini_dataset_id = "EuroEval/turku-ner-fi-mini"
    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(mini_dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(mini_dataset_id, private=True)


def read_dataset() -> dict:
    """Read the CoNLL-formatted TSV files and convert them to the required format."""
    base_url = "https://raw.githubusercontent.com/TurkuNLP/turku-ner-corpus/refs/heads/master/data/conll/{}.tsv"
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    data = dict(
        train=requests.get(train_url).text.split("\n\n"),
        val=requests.get(val_url).text.split("\n\n"),
        test=requests.get(test_url).text.split("\n\n"),
    )

    dfs = {}

    for split_name, split_raw_data in data.items():
        split_samples: list[dict[str, list[str]]] = []

        for raw_sample in split_raw_data:
            if "DOCSTART" in raw_sample or not raw_sample:
                continue

            tokens = []
            labels = []
            for line in raw_sample.split("\n"):
                token, label = line.split("\t")

                # Turku has additional labels that
                # we will not use.
                if label in LABEL_MAPS:
                    label = LABEL_MAPS[label]

                tokens.append(token)
                labels.append(label)

            split_samples.append({"tokens": tokens, "labels": labels})

        dfs[split_name] = pd.DataFrame.from_records(split_samples)

    # According to the paper, there should be 12,217 / 1,364 / 1,555
    # samples in the train/val/test splits, respectively.
    assert len(dfs["train"]) == 12_217
    assert len(dfs["val"]) == 1_364
    assert len(dfs["test"]) == 1_555

    return dfs


if __name__ == "__main__":
    main()
