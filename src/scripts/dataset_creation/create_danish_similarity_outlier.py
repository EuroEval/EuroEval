# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create and upload the Danish Similarity Outlier Detection dataset to the HF Hub."""

import ast
import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests as rq
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

try:
    from constants import CHOICES_MAPPING
except ModuleNotFoundError:
    from .constants import CHOICES_MAPPING

logging.basicConfig(format="%(asctime)s ⋅ %(message)s", level=logging.INFO)
logger = logging.getLogger("create_danish_similarity_outlier")

SOURCE_COMMIT = "3da3edf143fc386b02bb98dedca3cbfb8a905be0"
URL = (
    "https://raw.githubusercontent.com/kuhumcst/danish-semantic-reasoning-benchmark/"
    f"{SOURCE_COMMIT}/similarity/similarity.zip"
)
ZIP_PASSWORD = b"benchmark"
LETTERS = ["a", "b", "c", "d", "e", "f"]
GRANULARITY_FILES = {
    "fine": "outlier_similarity_fine.tsv",
    "medium": "outlier_similarity_medium.tsv",
    "coarse": "outlier_similarity_corse.tsv",
}
FILE_GRANULARITIES = {
    file_name: granularity for granularity, file_name in GRANULARITY_FILES.items()
}
MEDIUM_FINE_COLUMNS = [
    "candidates",
    "label",
    "outlier_position",
    "id",
    "chapter",
    "section",
]
SPLIT_CAPS = {"train": 1_024, "val": 256, "test": 2_048}
SPLIT_SEED = 4242
DATASET_ID = "EuroEval/danish-similarity-outlier-mini"
DATASET_CARD = f"""---
language:
- da
license: cc-by-nd-4.0
task_categories:
- text-classification
---

# Danish Similarity Outlier Detection

This dataset asks a model to identify the word that is least semantically similar to
the other five words in a list of six Danish words. It contains fine, medium, and coarse
similarity variants from the Danish Semantic Reasoning Benchmark.

The source archive was downloaded from commit `{SOURCE_COMMIT}` of the
[Danish Semantic Reasoning Benchmark](https://github.com/kuhumcst/danish-semantic-reasoning-benchmark).
Rows with malformed options were removed. The remaining rows were split by source ID so
all three granularity variants for an ID stay in the same split. The resulting splits
are 1,023 train, 255 validation, and 2,046 test rows.

The source data is licensed under [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/).
It is based on the Danish Thesaurus and credits the
[Society for Danish Language and Literature](https://dsl.dk/).

Citation:

> Bolette Pedersen, Nathalie Sørensen, Sussi Olsen, Sanni Nimb, and Simon Gray. 2024.
> [Towards a Danish Semantic Reasoning Benchmark - Compiled from Lexical-Semantic
> Resources for Assessing Selected Language Understanding Capabilities of Large Language
> Models](https://aclanthology.org/2024.lrec-main.1421/).
"""


def main() -> None:
    """Create the Danish dataset and upload it to the HF Hub."""
    response = rq.get(url=URL)
    response.raise_for_status()

    records: list[dict[str, str]] = []
    with zipfile.ZipFile(file=io.BytesIO(initial_bytes=response.content)) as zf:
        zf.setpassword(ZIP_PASSWORD)
        for file_name in GRANULARITY_FILES.values():
            records.extend(
                parse_tsv_file(file_name=file_name, content=zf.read(file_name))
            )

    df = pd.DataFrame(records)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Built {len(df)} multiple-choice samples.")

    split_dfs = split_dataframe(df=df)
    logger.info(
        "Splitting into "
        + " / ".join(f"{len(split_df)} {name}" for name, split_df in split_dfs.items())
        + " samples."
    )

    dataset = DatasetDict(
        {
            name: Dataset.from_pandas(split_df, split=split_name, preserve_index=False)
            for name, split_df, split_name in (
                ("train", split_dfs["train"], Split.TRAIN),
                ("val", split_dfs["val"], Split.VALIDATION),
                ("test", split_dfs["test"], Split.TEST),
            )
        }
    )

    api = HfApi()
    api.delete_repo(DATASET_ID, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(DATASET_ID, private=True)
    api.upload_file(
        path_or_fileobj=io.BytesIO(DATASET_CARD.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=DATASET_ID,
        repo_type="dataset",
        commit_message="Add dataset card",
    )


def parse_tsv_file(file_name: str, content: bytes) -> list[dict[str, str]]:
    """Parse a medium- or coarse-grained similarity outlier TSV file.

    Args:
        file_name:
            The TSV file name inside the upstream archive.
        content:
            The raw TSV bytes.

    Returns:
        Parsed multiple-choice records with text, label, source metadata, and
        granularity.

    Raises:
        ValueError:
            If the file name is not one of the source archive's similarity files.
    """
    granularity = FILE_GRANULARITIES.get(Path(file_name).name)
    if granularity is None:
        raise ValueError(f"Unknown Danish similarity file: {file_name!r}")

    if granularity == "coarse":
        df = pd.read_csv(
            filepath_or_buffer=io.BytesIO(initial_bytes=content), sep="\t", dtype=str
        )
        df = df.rename(columns={"core_group": "candidates", "outlier": "label"})
    else:
        df = pd.read_csv(
            filepath_or_buffer=io.BytesIO(initial_bytes=content),
            sep="\t",
            names=MEDIUM_FINE_COLUMNS,
            skiprows=1,
            dtype=str,
        )

    records: list[dict[str, str]] = []
    num_skipped = 0
    for _, row in df.iterrows():
        try:
            candidates = ast.literal_eval(str(row["candidates"]))
        except (SyntaxError, ValueError, TypeError):
            num_skipped += 1
            continue
        if (
            not isinstance(candidates, list)
            or len(candidates) != len(LETTERS)
            or not all(isinstance(candidate, str) for candidate in candidates)
        ):
            num_skipped += 1
            continue

        options = [candidate.replace("\n", " ").strip() for candidate in candidates]
        if any(not option for option in options) or len(set(options)) != len(options):
            num_skipped += 1
            continue

        try:
            outlier_position = int(str(row["outlier_position"]).strip())
        except (TypeError, ValueError):
            num_skipped += 1
            continue
        if not 0 <= outlier_position < len(LETTERS):
            num_skipped += 1
            continue

        source_id = str(row["id"]).strip()
        chapter = str(row["chapter"]).strip()
        section = str(row["section"]).strip()
        if (
            not source_id
            or source_id == "nan"
            or not chapter
            or chapter == "nan"
            or not section
            or section == "nan"
        ):
            num_skipped += 1
            continue

        text = (
            "Hvilket ord passer mindst sammen med de andre?\n"
            f"{CHOICES_MAPPING['da']}:\n"
            + "\n".join(
                f"{letter}. {option}" for letter, option in zip(LETTERS, options)
            )
        )
        records.append(
            {
                "text": text,
                "label": LETTERS[outlier_position],
                "source_id": source_id,
                "granularity": granularity,
                "chapter": chapter,
                "section": section,
            }
        )

    if num_skipped > 0:
        logger.warning(f"Skipped {num_skipped} malformed samples in {file_name}.")

    return records


def split_dataframe(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split complete source-ID groups into deterministic capped partitions.

    Args:
        df:
            A dataframe containing one row for each source ID and granularity.

    Returns:
        Dataframes for the train, validation, and test splits.

    Raises:
        ValueError:
            If there are not enough complete source-ID groups for the requested caps.
    """
    expected_granularities = set(GRANULARITY_FILES)
    group_sizes = df.groupby("source_id").size()
    group_granularities = df.groupby("source_id")["granularity"].agg(set)
    source_ids = sorted(
        source_id
        for source_id in group_sizes.index
        if group_sizes[source_id] == len(expected_granularities)
        and group_granularities[source_id] == expected_granularities
    )
    groups_per_split = {
        name: cap // len(expected_granularities) for name, cap in SPLIT_CAPS.items()
    }
    if len(source_ids) < sum(groups_per_split.values()):
        raise ValueError("Not enough complete source-ID groups for the requested caps.")

    train_ids, remaining_ids = train_test_split(
        source_ids, train_size=groups_per_split["train"], random_state=SPLIT_SEED
    )
    val_ids, remaining_ids = train_test_split(
        remaining_ids, train_size=groups_per_split["val"], random_state=SPLIT_SEED
    )
    test_ids, _ = train_test_split(
        remaining_ids, train_size=groups_per_split["test"], random_state=SPLIT_SEED
    )
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    complete_df = df[df["source_id"].isin(source_ids)]

    return {
        name: complete_df[complete_df["source_id"].isin(ids)]
        .sort_values(["source_id", "granularity"])
        .reset_index(drop=True)
        for name, ids in split_ids.items()
    }


if __name__ == "__main__":
    main()
