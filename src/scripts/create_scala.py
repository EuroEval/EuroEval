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

"""Create the ScaLA datasets and upload them to the HF Hub."""

import random
import re
import warnings
from typing import List, Tuple, Union

import pandas as pd
from constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT  # noqa
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from huggingface_hub.hf_api import HfApi
from load_ud_pos import (
    load_dadt_pos,
    load_dedt_pos,
    load_endt_pos,
    load_esdt_pos,
    load_fidt_pos,
    load_fodt_pos,
    load_frdt_pos,
    load_isdt_pos,
    load_itdt_pos,
    load_nldt_pos,
    load_nodt_nb_pos,
    load_nodt_nn_pos,
    load_svdt_pos,
)
from pandas.errors import SettingWithCopyWarning
from requests.exceptions import HTTPError
from tqdm.auto import tqdm


def main() -> None:
    """Create the ScaLA datasets and upload them to the HF Hub."""
    # Set up the POS dataset loaders
    pos_datasets = {
        "da": load_dadt_pos,
        "nb": load_nodt_nb_pos,
        "nn": load_nodt_nn_pos,
        "sv": load_svdt_pos,
        "is": load_isdt_pos,
        "fo": load_fodt_pos,
        "de": load_dedt_pos,
        "nl": load_nldt_pos,
        "en": load_endt_pos,
        "fr": load_frdt_pos,
        "it": load_itdt_pos,
        "es": load_esdt_pos,
        "fi": load_fidt_pos,
    }

    # Set up the progress bar and iterate over the languages
    with tqdm(pos_datasets.items(), desc="Creating ScaLA datasets") as pbar:
        for lang, fn in pbar:
            # Update the progress bar description
            pbar.set_description(f"Creating ScaLA datasets - {lang}")

            # Load the POS dataset
            pos_dataset = fn()

            # Merge the DDT POS dataframes to a single dataframe, with columns `ids`,
            # `tokens`, `doc` and `pos_tags`
            df = pd.concat(pos_dataset.values(), ignore_index=True)

            # Drop the duplicates
            df = df.drop_duplicates(subset="doc").reset_index(drop=True)

            # Remove samples with five or fewer tokens
            df = df[df.tokens.map(lambda lst: len(lst) > 5)]

            # Remove samples with five or fewer distinct POS tags
            df = df[df.pos_tags.map(lambda lst: len(set(lst)) > 5)]

            # Remove samples with an odd number of quotes
            df = df[df.doc.map(lambda doc: doc.count('"') % 2 == 0)]

            # Remove samples which starts with punctuation
            df = df[df.pos_tags.map(lambda lst: lst[0] not in ["PUNCT", "SYM"])]

            # Remove samples containing more than one '=' character, as this is used to
            # indicate a tag
            df = df[df.doc.map(lambda doc: doc.count("=") <= 1)]

            # Remove samples containing 'SLUTORD', as this is used to indicate a tag
            df = df[~df.doc.str.contains("SLUTORD")]

            # Create a training, validation, test and small training set. Note that we
            # will corrupt the data, so this is only half the size of the final
            # datasets. In the case where the dataframe does not contain enough samples
            # for all the splits, we keep halving the test size until we have enough
            # samples.
            test_size = 1024
            while test_size >= 128:
                try:
                    val_df = df.sample(n=128, random_state=4242)
                    df_filtered = df[~df.index.isin(val_df.index)]
                    test_df = df_filtered.sample(n=test_size, random_state=4242)
                    full_train_df = df_filtered[~df_filtered.index.isin(test_df.index)]
                    train_df = full_train_df.sample(n=512, random_state=4242)
                    break
                except ValueError:
                    test_size //= 2
            else:
                raise ValueError(
                    f"Not enough samples to create the splits. Found {len(df):,} "
                    f"samples, but need at least 768."
                )

            # Only work with samples where the document is not very large or small We do
            # it after we have made the splits to ensure that the dataset is minimally
            # affected.
            new_train_df = train_df.copy()
            new_train_df["text_len"] = new_train_df.doc.str.len()
            new_train_df = new_train_df.query(
                "text_len >= @MIN_NUM_CHARS_IN_DOCUMENT"
            ).query("text_len <= @MAX_NUM_CHARS_IN_DOCUMENT")
            new_val_df = val_df.copy()
            new_val_df["text_len"] = new_val_df.doc.str.len()
            new_val_df = new_val_df.query(
                "text_len >= @MIN_NUM_CHARS_IN_DOCUMENT"
            ).query("text_len <= @MAX_NUM_CHARS_IN_DOCUMENT")
            new_test_df = test_df.copy()
            new_test_df["text_len"] = new_test_df.doc.str.len()
            new_test_df = new_test_df.query(
                "text_len >= @MIN_NUM_CHARS_IN_DOCUMENT"
            ).query("text_len <= @MAX_NUM_CHARS_IN_DOCUMENT")
            new_full_train_df = full_train_df.copy()
            new_full_train_df["text_len"] = new_full_train_df.doc.str.len()
            new_full_train_df = new_full_train_df.query(
                "text_len >= @MIN_NUM_CHARS_IN_DOCUMENT"
            ).query("text_len <= @MAX_NUM_CHARS_IN_DOCUMENT")

            # Add the corrupted data and turn the dataframes into Hugging Face Dataset
            # objects
            train = prepare_df(new_train_df, split="train")
            val = prepare_df(new_val_df, split="val")
            test = prepare_df(new_test_df, split="test")
            full_train = prepare_df(new_full_train_df, split="train")

            # Collect datasets in a dataset dictionary
            dataset = DatasetDict(
                train=train, val=val, test=test, full_train=full_train
            )

            # Create dataset ID
            dataset_id = f"EuroEval/scala-{lang}"

            # Remove the dataset from Hugging Face Hub if it already exists
            try:
                api = HfApi()
                api.delete_repo(dataset_id, repo_type="dataset")
            except HTTPError:
                pass

            # Push the dataset to the Hugging Face Hub
            dataset.push_to_hub(dataset_id, private=True)


def join_tokens(tokens: List[str]) -> str:
    """Joins a list of tokens into a string.

    Args:
        tokens:
            The list of tokens to join.

    Returns:
        The joined string.
    """
    # Form document
    doc = " ".join(tokens)

    # Remove whitespace around punctuation
    doc = (
        doc.replace(" .", ".")
        .replace(" ,", ",")
        .replace(" ;", ";")
        .replace(" :", ":")
        .replace("( ", "(")
        .replace(" )", ")")
        .replace("[ ", "[")
        .replace(" ]", "]")
        .replace("{ ", "{")
        .replace(" }", "}")
        .replace(" ?", "?")
        .replace(" !", "!")
    )

    # Remove whitespace around quotes
    if doc.count('"') % 2 == 0:
        doc = re.sub('" ([^"]*) "', '"\\1"', doc)

    # Return the document
    return doc


def delete(tokens: List[str], pos_tags: List[str]) -> Union[str, None]:
    """Delete a random token from a list of tokens.

    The POS tags are used to prevent deletion of a token which does not make the
    resulting sentence grammatically incorrect, such as removing an adjective or an
    adverb.

    Args:
        tokens:
            The list of tokens to delete from.
        pos_tags:
            The list of POS tags for the tokens.

    Returns:
        The deleted token, or None if no token could be deleted.
    """
    # Copy the token list
    new_tokens = tokens.copy()

    # Get candidate indices to remove. We do not remove adjectives, adverbs,
    # punctuation, determiners or numbers, as the resulting sentence will probably
    # still be grammatically correct. Further, we do not remove nouns or proper nouns
    # if they have another noun or proper noun as neighbour, as that usually does not
    # make the sentence incorrect either.
    indices = [
        idx
        for idx, pos_tag in enumerate(pos_tags)
        if pos_tag not in ["ADJ", "ADV", "PUNCT", "SYM", "DET", "NUM"]
        and (
            pos_tag not in ["NOUN", "PROPN"]
            or (
                (idx == 0 or pos_tags[idx - 1] not in ["NOUN", "PROPN"])
                and (
                    idx == len(new_tokens) - 1
                    or pos_tags[idx + 1] not in ["NOUN", "PROPN"]
                )
            )
        )
    ]

    # If there are no candidates then return None
    if len(indices) == 0:
        return None

    # Get the random index
    rnd_idx = random.choice(indices)

    # Delete the token at the index
    new_tokens.pop(rnd_idx)

    # Join up the new tokens and return the string
    return join_tokens(new_tokens)


def flip_neighbours(tokens: List[str], pos_tags: List[str]) -> Union[str, None]:
    """Flip a pair of neighbouring tokens.

    The POS tags are used to prevent flipping of tokens which does not make the
    resulting sentence grammatically incorrect, such as flipping two adjectives.

    Args:
        tokens:
            The list of tokens to flip.
        pos_tags:
            The list of POS tags for the tokens.

    Returns:
        The flipped string, or None if no flip was possible.
    """
    # Copy the token list
    new_tokens = tokens.copy()

    # Collect all indices that are proper words, and which has a neighbour which is
    # also a proper word as well as having a different POS tag
    indices = [
        idx for idx, pos_tag in enumerate(pos_tags) if pos_tag not in ["PUNCT", "SYM"]
    ]
    indices = [
        idx
        for idx in indices
        if (idx + 1 in indices and pos_tags[idx] != pos_tags[idx + 1])
        or (idx - 1 in indices and pos_tags[idx] != pos_tags[idx - 1])
    ]

    # If there are fewer than two relevant tokens then return None
    if len(indices) < 2:
        return None

    # Get the first random index
    rnd_fst_idx = random.choice(indices)

    # Get the second (neighbouring) index
    if rnd_fst_idx == 0:
        rnd_snd_idx = rnd_fst_idx + 1
    elif rnd_fst_idx == len(tokens) - 1:
        rnd_snd_idx = rnd_fst_idx - 1
    elif (
        pos_tags[rnd_fst_idx + 1] in ["PUNCT", "SYM"]
        or pos_tags[rnd_fst_idx] == pos_tags[rnd_fst_idx + 1]
        or {pos_tags[rnd_fst_idx], pos_tags[rnd_fst_idx + 1]} == {"PRON", "AUX"}
    ):
        rnd_snd_idx = rnd_fst_idx - 1
    elif (
        pos_tags[rnd_fst_idx - 1] in ["PUNCT", "SYM"]
        or pos_tags[rnd_fst_idx] == pos_tags[rnd_fst_idx - 1]
        or {pos_tags[rnd_fst_idx], pos_tags[rnd_fst_idx + 1]} == {"PRON", "AUX"}
    ):
        rnd_snd_idx = rnd_fst_idx + 1
    elif random.random() > 0.5:
        rnd_snd_idx = rnd_fst_idx - 1
    else:
        rnd_snd_idx = rnd_fst_idx + 1

    # Flip the two indices
    new_tokens[rnd_fst_idx] = tokens[rnd_snd_idx]
    new_tokens[rnd_snd_idx] = tokens[rnd_fst_idx]

    # If we flipped the first character, then ensure that the new first character is
    # title-cased and the second character is of lower case. We only do this if they
    # are not upper cased, however.
    if rnd_fst_idx == 0 or rnd_snd_idx == 0:
        if new_tokens[0] != new_tokens[0].upper():
            new_tokens[0] = new_tokens[0].title()
        if new_tokens[1] != new_tokens[1].upper():
            new_tokens[1] = new_tokens[1].lower()

    # Join up the new tokens and return the string
    return join_tokens(new_tokens)


def corrupt(
    tokens: List[str], pos_tags: List[str], num_corruptions: int = 1
) -> List[Tuple[str, str]]:
    """Corrupt a list of tokens.

    This randomly either flips two neighbouring tokens or deletes a random token.

    Args:
        tokens:
            The list of tokens to corrupt.
        pos_tags:
            The list of POS tags for the tokens.
        num_corruptions:
            The number of corruptions to perform. Defaults to 1.

    Returns:
        The list of (corrupted_string, corruption_type)
    """
    # Define the list of corruptions
    corruptions: List[Tuple[str, str]] = list()

    # Continue until we have achieved the desired number of corruptions
    while len(corruptions) < num_corruptions:
        # Choose which corruption to perform, at random
        corruption_fn = random.choice([flip_neighbours, delete])

        # Corrupt the tokens
        corruption = corruption_fn(tokens, pos_tags)

        # If the corruption succeeded, and that we haven't already performed the same
        # corruption, then add the corruption to the list of corruptions
        if corruption not in corruptions and corruption is not None:
            corruptions.append((corruption, corruption_fn.__name__))

    # Return the list of corruptions
    return corruptions


def prepare_df(df: pd.DataFrame, split: str) -> Dataset:
    """Prepare a dataframe by adding an equal number of corruptions to it.

    Args:
        df:
            The dataframe to prepare.
        split:
            The split to prepare the dataframe for.

    Returns:
        The prepared dataset.
    """
    # Reset the index of the dataframe
    df.reset_index(drop=True, inplace=True)

    # Get the corrupted strings
    corrupted_list = [
        corrupt(tokens=tokens, pos_tags=pos_tags, num_corruptions=1)[0]
        for tokens, pos_tags in zip(df.tokens, df.pos_tags)
    ]

    # Add the corrupted strings to the dataframe
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SettingWithCopyWarning)
        df["corrupted"] = [tup[0] for tup in corrupted_list]
        df["corruption_type"] = [tup[1] for tup in corrupted_list]

    # Restructure the dataframe to have columns `text`, `corruption_type` and `label`,
    # with one sample per row
    df = pd.concat(
        [
            pd.DataFrame(
                dict(
                    text=df.tokens.map(join_tokens).tolist(),
                    corruption_type=[None for _ in range(len(df))],
                    label=["correct" for _ in range(len(df))],
                )
            ),
            pd.DataFrame(
                dict(
                    text=df.corrupted.explode().tolist(),
                    corruption_type=df.corruption_type.explode().tolist(),
                    label=["incorrect" for _ in range(len(df))],
                )
            ),
        ]
    )

    # Shuffle the dataframe
    df = df.sample(frac=1.0, random_state=4242).reset_index(drop=True)

    # Convert the dataframe to a Hugging Face Dataset and return it
    return Dataset.from_pandas(df, split=split)


if __name__ == "__main__":
    main()
