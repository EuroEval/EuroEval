# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==2.15.0",
#     "huggingface-hub==0.24.0",
#     "lxml==5.3.1",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the SUC3-mini NER dataset and upload it to the HF Hub."""

import bz2
import io
import re
from typing import Dict, List, Union

import pandas as pd
import requests
from datasets import Split
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from huggingface_hub.hf_api import HfApi
from lxml import etree
from requests.exceptions import HTTPError


def main() -> None:
    """Create the DaNE-mini NER dataset and uploads it to the HF Hub."""
    # Define download URLs
    url = "https://spraakbanken.gu.se/lb/resurser/meningsmangder/suc3.xml.bz2"

    # Download the data
    compressed_xml_data = requests.get(url).content

    # Decompress the data
    xml_data = bz2.decompress(compressed_xml_data)

    # Parse the XML data
    context = etree.iterparse(io.BytesIO(xml_data), events=("start", "end"))

    # Define a conversion dict to process the NER tags
    conversion_dict = dict(
        O="O",
        animal="MISC",
        event="MISC",
        inst="ORG",
        myth="MISC",
        other="MISC",
        person="PER",
        place="LOC",
        product="MISC",
        work="MISC",
    )

    # Initialise a `ner_tag` variable as the "no tag", `O`
    ner_tag = "O"

    # Iterate over the context
    records: List[Dict[str, Union[str, List[str]]]] = list()
    tokens: List[str] = list()
    ner_tags: List[str] = list()
    for action, elt in context:
        # If the current element begins a name then set the `ner_tag` to the
        # corresponding `B` value
        if elt.tag == "name" and action == "start":
            ner_tag = f"B-{conversion_dict[elt.attrib['type']]}"

        # If the current element ends a name then reset the `ner_tag` to `O`
        elif elt.tag == "name" and action == "end":
            ner_tag = "O"

        # If the current element starts a word (i.e., `w`) then add the word to the
        # list of tokens, and add the current NER tag to the list of NER tags
        elif elt.tag == "w" and action == "start":
            if elt.text:
                tokens.append(elt.text)
                ner_tags.append(ner_tag)

        # If the current element ends a word then set the current NER tag to the `I`
        # version of the previous NER tag
        elif elt.tag == "w" and action == "end":
            if ner_tag.startswith("B-"):
                ner_tag = f"I-{ner_tag[2:]}"

        # If the current element ends a sentence then store all the data in the list of
        # records
        elif elt.tag == "sentence" and action == "end" and len(tokens) > 0:
            # Create document from the tokens
            doc = re.sub(" ([.,])", "\1", " ".join(tokens))

            # Double-check that there are the same number of tokens as NER tags
            assert len(tokens) == len(ner_tags)

            # Create the record and append it to the list of records
            record: dict[str, str | list[str]] = dict(
                doc=doc, tokens=tokens, ner_tags=ner_tags
            )
            records.append(record)

        # If the current element starts a sentence then reset the list of tokens and
        # NER tags, and set the current NER tag to `O`
        elif elt.tag == "sentence" and action == "start":
            tokens = list()
            ner_tags = list()
            ner_tag = "O"

    # Convert the records to a DataFrame
    df = pd.DataFrame.from_records(records)

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
    )

    # Create dataset ID
    dataset_id = "EuroEval/suc3-mini"

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
