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

"""Create the GerLangMod GED datasets and upload them to the HF Hub.

Source: https://github.com/noahmanu/gerlangmod (ver. 1.1)
Paper: Noah-Manuel Michael and Andrea Horbach (2025). GermDetect: Verb Placement Error
Detection Datasets for Learners of Germanic Languages. In Proceedings of the 20th
Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2025).
"""

import io

import pandas as pd
import requests
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from tqdm.auto import tqdm

_BASE_URL = (
    "https://raw.githubusercontent.com/noahmanu/gerlangmod/main/"
    "Datasets/verb_error_datasets_v1_1/{lang}/{prefix}_{treebank}_{split}.tsv"
)

# Mapping from language code to list of (treebank, splits) pairs
_LANG_FILES: dict[str, list[tuple[str, list[str]]]] = {
    "da": [("ddt", ["train", "dev", "test"])],
    "nl": [
        ("alpino", ["train", "dev", "test"]),
        ("lassysmall", ["train", "dev", "test"]),
    ],
    "fo": [("farpahc", ["train", "dev", "test"]), ("oft", ["test"])],
    "de": [
        ("gsd", ["train", "dev", "test"]),
        ("hdt", ["train", "dev", "test"]),
        ("lit", ["test"]),
        ("pud", ["test"]),
    ],
    "is": [
        ("icepahc", ["train", "dev", "test"]),
        ("gc", ["train", "dev", "test"]),
        ("modern", ["train", "dev", "test"]),
        ("pud", ["test"]),
    ],
    "nb": [("bokmaal", ["train", "dev", "test"])],
    "nn": [("nynorsk", ["train", "dev", "test"])],
    "sv": [
        ("talbanken", ["train", "dev", "test"]),
        ("lines", ["train", "dev", "test"]),
        ("pud", ["test"]),
    ],
}

# Both Norwegian variants use "no_*" file prefixes instead of "nb_*" / "nn_*"
_FILE_PREFIX_OVERRIDES = {("nb", "bokmaal"): "no", ("nn", "nynorsk"): "no"}

# Dataset IDs on HuggingFace Hub
_DATASET_IDS = {
    "da": "EuroEval/gerlangmod-da",
    "nl": "EuroEval/gerlangmod-nl",
    "fo": "EuroEval/gerlangmod-fo",
    "de": "EuroEval/gerlangmod-de",
    "is": "EuroEval/gerlangmod-is",
    "nb": "EuroEval/gerlangmod-nb",
    "nn": "EuroEval/gerlangmod-nn",
    "sv": "EuroEval/gerlangmod-sv",
}


def _convert_labels(raw_labels: list[str]) -> list[str]:
    """Convert GerLangMod O/C/F labels to IOB2 O/B-ERR/I-ERR format.

    Args:
        raw_labels: List of raw labels ('O', 'C', or 'F').

    Returns:
        List of IOB2 labels. 'O' and 'C' become 'O'. The first 'F' in a
        consecutive run becomes 'B-ERR' and subsequent 'F's become 'I-ERR'.
    """
    iob2 = []
    prev = "O"
    for label in raw_labels:
        if label == "F":
            iob2.append("I-ERR" if prev == "F" else "B-ERR")
        else:
            iob2.append("O")
        prev = label
    return iob2


def _download_tsv(lang: str, treebank: str, split: str) -> pd.DataFrame:
    """Download a single TSV file and return it as a DataFrame.

    Returns:
        DataFrame with raw GerLangMod columns.
    """
    prefix = _FILE_PREFIX_OVERRIDES.get((lang, treebank), lang)
    url = _BASE_URL.format(lang=lang, prefix=prefix, treebank=treebank, split=split)
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text), sep="\t", dtype=str, na_filter=False)
    return df


def _process_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extract tokens and IOB2 labels from a raw GerLangMod DataFrame.

    Returns:
        DataFrame with 'tokens' (list of str) and 'labels' (list of str) columns.
    """
    records = []
    for _, row in df.iterrows():
        tokens = row["no_punc_lower_permuted"].split()
        raw_labels = row["permuted_gold"].split()
        if len(tokens) != len(raw_labels):
            continue
        labels = _convert_labels(raw_labels)
        records.append({"tokens": tokens, "labels": labels})
    return pd.DataFrame.from_records(records)


def main() -> None:
    """Create the GerLangMod GED datasets and upload them to the HF Hub."""
    for lang in tqdm(_LANG_FILES, desc="Languages"):
        # Download and concatenate all TSV files for this language
        dfs: list[pd.DataFrame] = []
        for treebank, splits in _LANG_FILES[lang]:
            for split in splits:
                raw_df = _download_tsv(lang, treebank, split)
                processed = _process_df(raw_df)
                dfs.append(processed)

        df = pd.concat(dfs, ignore_index=True)

        # Remove any duplicate sentences
        df = df.drop_duplicates(subset=["tokens"]).reset_index(drop=True)

        # Create validation split (256 samples, or all available if fewer)
        n_val = min(256, len(df))
        val_df = df.sample(n=n_val, random_state=4242)
        remaining = df.loc[~df.index.isin(val_df.index)]

        # Create test split (2048 samples), up-weighting sentences with errors
        n_test = min(2048, len(remaining))
        weights = [5.0 if "B-ERR" in labels else 1.0 for labels in remaining["labels"]]
        test_df = remaining.sample(n=n_test, random_state=4242, weights=weights)
        full_train_df = remaining.loc[~remaining.index.isin(test_df.index)]
        assert isinstance(full_train_df, pd.DataFrame)

        # Create train split (1024 samples, or all available if fewer)
        n_train = min(1024, len(full_train_df))
        train_df = full_train_df.sample(n=n_train, random_state=4242)

        # Reset indices
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        full_train_df = full_train_df.reset_index(drop=True)

        # Build HuggingFace DatasetDict
        dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
                "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
                "test": Dataset.from_pandas(test_df, split=Split.TEST),
                "full_train": Dataset.from_pandas(
                    full_train_df,
                    split="full_train",  # pyrefly: ignore[bad-argument-type]
                ),
            }
        )

        # Push to the Hugging Face Hub
        dataset_id = _DATASET_IDS[lang]
        HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
        dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
