# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "scikit-learn==1.6.1",
# ]
# ///

"""Create the ScaLA-lb linguistic acceptability dataset from UD Luxembourgish.

Uses the ScaLA algorithm to create grammaticality judgments from the
UD Luxembourgish-LuxBank treebank by corrupting sentences.
"""

import random
from pathlib import Path

import pandas as pd
import requests
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

UD_LB_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_Luxembourgish-LuxBank/main/lb_luxbank-ud-train.conllu"


def parse_conllu(file_path: Path) -> list[dict]:
    """Parse CoNLL-U file and extract sentences.

    Args:
        file_path:
            Path to the CoNLL-U file.

    Returns:
        List of sentence dictionaries with text and tokens.
    """
    sentences = []
    current_sentence = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if current_sentence:
                    words = [
                        w["form"] for w in current_sentence if isinstance(w["id"], int)
                    ]
                    sentences.append(
                        {"text": " ".join(words), "tokens": current_sentence}
                    )
                    current_sentence = []
            elif "\t" in line:
                parts = line.split("\t")
                if not parts[0].isdigit():
                    continue
                token = {
                    "id": int(parts[0]),
                    "form": parts[1],
                    "lemma": parts[2],
                    "upos": parts[3],
                    "xpos": parts[4],
                    "feats": parts[5],
                    "head": int(parts[6]) if parts[6] != "_" else 0,
                    "deprel": parts[7],
                }
                current_sentence.append(token)

    return sentences


def corrupt_sentence(tokens: list) -> tuple[str, str, str] | None:
    """Create correct and corrupted versions of a sentence.

    Args:
        tokens:
            List of token dictionaries from CoNLL-U.

    Returns:
        Tuple of (correct_text, corrupted_text, corruption_type) or None if
        sentence is too short.
    """
    words = [t["form"] for t in tokens if isinstance(t["id"], int)]
    if len(words) < 3:
        return None

    correct_text = " ".join(words)

    corruption_type = random.choice(["swap", "delete"])

    if corruption_type == "swap" and len(words) >= 2:
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]
    elif len(words) >= 2:
        idx = random.randint(1, len(words) - 2)
        del words[idx]

    corrupted_text = " ".join(words)
    return correct_text, corrupted_text, corruption_type


def create_scala_dataset(sentences: list) -> pd.DataFrame:
    """Create ScaLA dataset with correct/corrupted pairs.

    Args:
        sentences:
            List of sentence dictionaries with text and tokens.

    Returns:
        DataFrame with text and label columns.
    """
    data = []
    random.seed(42)

    for sent in sentences:
        result = corrupt_sentence(sent["tokens"])
        if result:
            correct_text, corrupted_text, corruption_type = result
            data.append({"text": correct_text, "label": "correct"})
            data.append(
                {
                    "text": corrupted_text,
                    "label": "incorrect",
                    "corruption_type": corruption_type,
                }
            )

    return pd.DataFrame(data)


def make_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits.

    Args:
        df:
            Full dataset.

    Returns:
        Tuple of (train, val, test) DataFrames.
    """
    df = df[["text", "label"]]

    n = len(df)
    n_train = min(1024, int(n * 0.5))
    n_val = min(256, int(n * 0.15))

    train, temp = train_test_split(
        df, train_size=n_train, random_state=42, stratify=df["label"]
    )
    val, test = train_test_split(
        temp, train_size=n_val / len(temp), random_state=42, stratify=temp["label"]
    )

    for d in [train, val, test]:
        d.reset_index(drop=True, inplace=True)

    return train, val, test


def main() -> None:
    """Create ScaLA-lb dataset."""
    print("Downloading UD Luxembourgish treebank...")
    response = requests.get(UD_LB_URL, timeout=30)
    response.raise_for_status()

    cache_dir = Path.home() / ".cache" / "euroeval" / "scala-lb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    conllu_path = cache_dir / "lb_luxbank-ud-train.conllu"
    conllu_path.write_text(response.text)

    print("Parsing CoNLL-U file...")
    sentences = parse_conllu(conllu_path)
    print(f"Found {len(sentences)} sentences")

    print("Creating dataset...")
    df = create_scala_dataset(sentences)
    print(f"Created {len(df)} samples")

    print("Creating splits...")
    train, val, test = make_splits(df)
    print(f"Splits: {len(train)} train, {len(val)} val, {len(test)} test")

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train),
            "val": Dataset.from_pandas(val),
            "test": Dataset.from_pandas(test),
        }
    )

    dataset_id = "EuroEval/scala-lb"
    print(f"Uploading to {dataset_id}...")

    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)
    print(f"✓ Uploaded {dataset_id}")


if __name__ == "__main__":
    main()
