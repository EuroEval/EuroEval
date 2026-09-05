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

"""Create the Danish Similarity Outlier Detection dataset and upload it to the HF Hub."""

import ast
import io
import logging
import zipfile

import pandas as pd
import requests as rq
from constants import CHOICES_MAPPING
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

logging.basicConfig(format="%(asctime)s ⋅ %(message)s", level=logging.INFO)
logger = logging.getLogger("create_danish_similarity_outlier")

URL = (
    "https://raw.githubusercontent.com/kuhumcst/danish-semantic-reasoning-benchmark"
    "/main/similarity/similarity.zip"
)
ZIP_PASSWORD = b"benchmark"
LETTERS = ["a", "b", "c", "d", "e", "f"]
MEDIUM_FINE_COLUMNS = [
    "candidates",
    "label",
    "outlier_position",
    "id",
    "chapter",
    "section",
]


def main() -> None:
    """Create the Danish Similarity Outlier Detection dataset and upload it to the HF Hub."""
    response = rq.get(url=URL)
    response.raise_for_status()

    records: list[dict[str, str]] = []
    with zipfile.ZipFile(file=io.BytesIO(initial_bytes=response.content)) as zf:
        zf.setpassword(ZIP_PASSWORD)
        for file_name in ("outlier_similarity_medium.tsv", "outlier_similarity_corse.tsv"):
            records.extend(parse_tsv_file(file_name=file_name, content=zf.read(file_name)))

    df = pd.DataFrame(records)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Built {len(df)} multiple-choice samples.")

    train_size = int(len(df) * 1_024 / 3_328)
    val_size = int(len(df) * 256 / 3_328)
    logger.info(
        f"Splitting into {train_size} train / {val_size} val / "
        f"{len(df) - train_size - val_size} test samples."
    )

    train_df, remaining_df = train_test_split(
        df, train_size=train_size, random_state=4242
    )
    val_df, test_df = train_test_split(
        remaining_df, train_size=val_size, random_state=4242
    )

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(
                train_df.reset_index(drop=True), split=Split.TRAIN
            ),
            "val": Dataset.from_pandas(
                val_df.reset_index(drop=True), split=Split.VALIDATION
            ),
            "test": Dataset.from_pandas(
                test_df.reset_index(drop=True), split=Split.TEST
            ),
        }
    )

    dataset_id = "EuroEval/danish-similarity-outlier"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


def parse_tsv_file(file_name: str, content: bytes) -> list[dict[str, str]]:
    """Parse a medium- or coarse-grained similarity outlier TSV file.

    Args:
        file_name:
            The TSV file name inside the upstream archive.
        content:
            The raw TSV bytes.

    Returns:
        Parsed multiple-choice records with ``text`` and ``label`` columns.
    """
    if file_name.endswith("corse.tsv"):
        df = pd.read_csv(filepath_or_buffer=io.BytesIO(initial_bytes=content), sep="\t")
        df = df.rename(columns={"core_group": "candidates", "outlier": "label"})
    else:
        df = pd.read_csv(
            filepath_or_buffer=io.BytesIO(initial_bytes=content),
            sep="\t",
            names=MEDIUM_FINE_COLUMNS,
            skiprows=1,
        )

    records: list[dict[str, str]] = []
    num_skipped = 0
    for _, row in df.iterrows():
        candidates = row["candidates"]
        if not isinstance(candidates, list):
            try:
                candidates = ast.literal_eval(str(candidates))
            except (SyntaxError, ValueError):
                num_skipped += 1
                continue
        if not isinstance(candidates, list) or len(candidates) != 6:
            num_skipped += 1
            continue

        options = [str(candidate).replace("\n", " ").strip() for candidate in candidates]
        if any(not option for option in options):
            num_skipped += 1
            continue

        outlier_position = int(row["outlier_position"])
        text = (
            "Hvilket ord passer mindst sammen med de andre?\n"
            f"{CHOICES_MAPPING['da']}:\n"
            + "\n".join(
                f"{letter}. {option}" for letter, option in zip(LETTERS, options)
            )
        )
        records.append({"text": text, "label": LETTERS[outlier_position]})

    if num_skipped > 0:
        logger.warning(f"Skipped {num_skipped} malformed samples in {file_name}.")

    return records


if __name__ == "__main__":
    main()
