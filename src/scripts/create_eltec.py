# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "nltk==3.9.1",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "tqdm==4.67.1",
#     "urllib3==2.3.0",
# ]
# ///

"""Create the ELTEC-mini NER dataset and upload it to the HF Hub."""

import io
import json
import logging
import re
import warnings
from collections import defaultdict
from zipfile import ZipFile

import nltk
import pandas as pd
import requests as rq
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from tqdm.auto import tqdm
from urllib3.exceptions import InsecureRequestWarning

logging.basicConfig(format="%(asctime)s ⋅ %(message)s", level=logging.INFO)
logger = logging.getLogger("create_eltec")


nltk.download("punkt_tab")


def main() -> None:
    """Create the ELTEC-mini NER dataset and upload it to the HF Hub."""
    # Download the zip file
    logger.info("Downloading the zip file...")
    url = (
        "https://dspace-clarin-it.ilc.cnr.it/repository/xmlui/bitstream/handle/"
        "20.500.11752/OPEN-986/French_ELTEC_NER_Open_Dataset.zip"
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InsecureRequestWarning)
        response = rq.get(url, verify=False)
    response.raise_for_status()

    # Unpack the zip file
    logger.info("Unpacking the zip file...")
    with ZipFile(file=io.BytesIO(initial_bytes=response.content)) as zip_file:
        all_ids = {
            match.group()
            for name in zip_file.namelist()
            if (match := re.search(pattern=r"FRA[0-9]{5}", string=name)) is not None
        }
        texts: dict[str, str] = {
            id_: zip_file.read(name=name).decode("utf-8")
            for id_ in all_ids
            for name in zip_file.namelist()
            if name.startswith("French_ELTEC_NER_Open_Dataset/texts/tr") and id_ in name
        }
        annotations: dict[str, list[dict]] = {
            id_: json.loads(zip_file.read(name=name))["entities"]
            for id_ in all_ids
            for name in zip_file.namelist()
            if name.startswith("French_ELTEC_NER_Open_Dataset/annotations/pool/")
            and id_ in name
        }

    # Sanity check that we got all the texts and annotations
    assert all(id_ in texts for id_ in all_ids), (
        f"Expected all IDs to be in the texts, but {all_ids - set(texts)} are not."
    )
    assert all(id_ in annotations for id_ in all_ids), (
        "Expected all IDs to be in the annotations, but "
        f"{all_ids - set(annotations)} are not."
    )

    # Mapping that converts the event IDs in the dataset to readable entities
    event_to_entity = dict(
        e_1="PER",  # Original: "PERS"
        e_2="LOC",
        e_3="ORG",
        e_4="MISC",  # Original: "OTHER"
        e_5="O",  # Original: "WORK"
        e_6="O",  # Original: "DEMO"
        e_7="O",  # Original: "ROLE"
        e_8="O",  # Original: "EVENT"
    )

    data_dict: dict[str, list] = defaultdict(list)
    for id_, text in tqdm(texts.items(), desc="Processing texts"):
        # Extract the entities for the current text
        entities = sorted(
            [
                dict(
                    text=offset_dict["text"],
                    offset=int(offset_dict["start"]),
                    label=event_to_entity[ent_dict["classId"]],
                )
                for ent_dict in annotations[id_]
                for offset_dict in ent_dict["offsets"]
            ],
            key=lambda entity: entity["offset"],
        )

        # Sanity check that the offsets and text match
        if not all(
            text[entity["offset"] : entity["offset"] + len(entity["text"])]
            == entity["text"]
            for entity in entities
        ):
            continue

        # Extract the tokens and their character intervals
        try:
            tokens = nltk.word_tokenize(text=text, language="french")
            token_idx_to_char_idxs = dict()
            char_idx = 0
            for token_idx, token in enumerate(tokens):
                start_idx = text.index(token, char_idx)
                token_idx_to_char_idxs[token_idx] = (start_idx, start_idx + len(token))
                char_idx = start_idx + len(token)
        except ValueError:
            continue

        # Create the NER tags
        ner_tags = ["O"] * len(tokens)
        for entity in entities:
            start_idx = entity["offset"]
            end_idx = start_idx + len(entity["text"])
            token_idxs = [
                token_idx
                for token_idx, (
                    token_start,
                    token_end,
                ) in token_idx_to_char_idxs.items()
                if token_start < end_idx and token_end > start_idx
            ]
            label = entity["label"]
            if label == "O":
                continue
            elif len(token_idxs) == 1:
                ner_tags[token_idxs[0]] = f"B-{label}"
            else:
                ner_tags[token_idxs[0]] = f"B-{label}"
                for token_idx in token_idxs[1:]:
                    ner_tags[token_idx] = f"I-{label}"

        # Split up the texts into sentences
        sentences = nltk.sent_tokenize(text=" ".join(tokens), language="french")
        token_start_idx = 0
        for sentence in sentences:
            sentence_tokens = sentence.split()
            token_end_idx = token_start_idx + len(sentence_tokens)
            sentence_tokens = tokens[token_start_idx:token_end_idx]
            sentence_ner_tags = ner_tags[token_start_idx:token_end_idx]
            data_dict["tokens"].append(sentence_tokens)
            data_dict["labels"].append(sentence_ner_tags)
            token_start_idx = token_end_idx

    df = pd.DataFrame(data_dict)

    # Create validation split. Since most of the samples in the dataset has no tags, we
    # add a weight to the samples that has tags to ensure that dataset has a reasonable
    # amount of samples with tags.
    val_size = 256
    val_df = df.sample(
        n=val_size,
        random_state=4242,
        weights=[5.0 if len(set(labels)) > 1 else 1.0 for labels in df["labels"]],
    )

    # Create test split. We add weights to the sampling as with the validation split.
    test_size = 1024
    df = df[~df.index.isin(val_df.index)]
    test_df = df.sample(
        n=test_size,
        random_state=4242,
        weights=[5.0 if len(set(labels)) > 1 else 1.0 for labels in df["labels"]],
    )

    # Create train split. We add weights to the sampling as with the validation split.
    train_size = 1024
    df = df[~df.index.isin(test_df.index)]
    train_df = df.sample(
        n=train_size,
        random_state=4242,
        weights=[5.0 if len(set(labels)) > 1 else 1.0 for labels in df["labels"]],
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
    dataset_id = "EuroEval/eltec-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except rq.HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
