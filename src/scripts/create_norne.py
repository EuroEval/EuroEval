# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "tqdm==4.67.1",
# ]
# ///

"""Create the NorNE-mini NER datasets and upload them to the HF Hub."""

import re
from collections import defaultdict
from typing import Dict, List, Union

import pandas as pd
import requests
from datasets import Split
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from huggingface_hub.hf_api import HfApi
from requests.exceptions import HTTPError
from tqdm.auto import tqdm


def main() -> None:
    """Create the NorNE-mini NER datasets and uploads them to the HF Hub."""
    # Define the base download URLs
    base_urls = dict(
        nb="https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal/raw/master/"
        "no_bokmaal-ud-{}.conllu",
        nn="https://github.com/UniversalDependencies/UD_Norwegian-Nynorsk/raw/master/"
        "no_nynorsk-ud-{}.conllu",
    )

    # Define conversion dict to be used with the NER tags
    ner_conversion_dict = {
        "O": "O",
        "B-LOC": "B-LOC",
        "I-LOC": "I-LOC",
        "B-PER": "B-PER",
        "I-PER": "I-PER",
        "B-ORG": "B-ORG",
        "I-ORG": "I-ORG",
        "B-MISC": "B-MISC",
        "I-MISC": "I-MISC",
        "B-GPE_LOC": "B-LOC",
        "I-GPE_LOC": "I-LOC",
        "B-GPE_ORG": "B-ORG",
        "I-GPE_ORG": "I-ORG",
        "B-PROD": "B-MISC",
        "I-PROD": "I-MISC",
        "B-DRV": "B-MISC",
        "I-DRV": "I-MISC",
        "B-EVT": "B-MISC",
        "I-EVT": "I-MISC",
    }

    # Set up the progress bar and iterate over the languages
    with tqdm(["nb", "nn"], desc="Creating NorNE-mini datasets") as pbar:
        for lang in pbar:
            # Define download URLs
            train_url = base_urls[lang].format("train")
            val_url = base_urls[lang].format("dev")
            test_url = base_urls[lang].format("test")

            # Download the data
            data = dict(
                train=requests.get(train_url).text.split("\n"),
                val=requests.get(val_url).text.split("\n"),
                test=requests.get(test_url).text.split("\n"),
            )

            # Update the progress bar description
            pbar.set_description(f"Creating NorNE datasets - {lang}")

            # Iterate over the data splits
            dfs = dict()
            for split, lines in data.items():
                # Initialise the records, data dictionary and document
                records = list()
                data_dict: Dict[str, List[Union[int, str]]] = defaultdict(list)
                doc = ""

                # Iterate over the data for the given split
                for line in lines:
                    # If we are at the first line of an entry then extract the document
                    if line.startswith("# text = "):
                        doc = re.sub("# text = ", "", line)

                    # Otherwise, if the line is a comment then ignore it
                    elif line.startswith("#"):
                        continue

                    # Otherwise, if we have reached the end of an entry then store it
                    # to the list of records and reset the data dictionary and document
                    elif line == "":
                        merged_data_dict: Dict[str, Union[str, List[Union[int, str]]]]
                        if len(data_dict["tokens"]) > 0:
                            merged_data_dict = {**data_dict, "doc": doc}
                            records.append(merged_data_dict)
                        data_dict = defaultdict(list)
                        doc = ""

                    # Otherwise we are in the middle of an entry which is not a
                    # comment, so we extract the data from the line and store it in the
                    # data dictionary
                    else:
                        data_tup = line.split("\t")
                        ner_tag = data_tup[9].replace("name=", "").split("|")[-1]
                        data_dict["ids"].append(data_tup[0])
                        data_dict["tokens"].append(data_tup[1])
                        data_dict["ner_tags"].append(ner_conversion_dict[ner_tag])

                # Convert the records to a dataframe and store it
                dfs[split] = pd.DataFrame.from_records(records)

            # Merge the dataframes
            df = pd.concat(dfs.values()).reset_index(drop=True)

            # Remove duplicates
            df = df.drop_duplicates().reset_index(drop=True)

            # Create new splits
            val_df = df.sample(n=256, random_state=4242)
            df_filtered = df[~df.index.isin(val_df.index)]
            test_df = df_filtered.sample(n=2048, random_state=4242)
            full_train_df = df_filtered[~df_filtered.index.isin(test_df.index)]
            train_df = full_train_df.sample(n=1024, random_state=4242)

            # Collect datasets in a dataset dictionary
            dataset = DatasetDict(
                train=Dataset.from_pandas(train_df, split=Split.TRAIN),
                val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
                test=Dataset.from_pandas(test_df, split=Split.TEST),
                full_train=Dataset.from_pandas(full_train_df, split="full_train"),
            )

            # Create dataset ID
            dataset_id = f"EuroEval/norne-{lang}-mini"

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
