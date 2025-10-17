# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the UNER-sk NER dataset and upload it to the HF Hub."""

import json
import re

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi


def main() -> None:
    """Create the UNER-Slovak NER dataset and uploads it to the HF Hub."""
    repo_id = "universalner/uner_llm_inst_slovak"
    dataset = load_dataset(repo_id)

    # Assuming `inputs` is the text and `targets` contains the entities
    train_df = pd.DataFrame(dataset["train"])
    val_df = pd.DataFrame(dataset["validation"])
    test_df = pd.DataFrame(dataset["test"])

    train_df = process_df(train_df)
    val_df = process_df(val_df)
    test_df = process_df(test_df)

    # Make splits
    train_size = 1024
    val_size = 256
    test_size = 2048

    # Take additional samples from the training set for the test split
    additional_test_samples_needed = test_size - len(test_df)
    additional_test_samples = train_df.sample(
        n=additional_test_samples_needed, random_state=4242
    )
    final_test_df = pd.concat([test_df, additional_test_samples], ignore_index=True)
    remaining_train_df = train_df[~train_df.index.isin(additional_test_samples.index)]

    # Sample final splits
    final_test_df = final_test_df.sample(n=test_size, random_state=4242).reset_index(
        drop=True
    )
    final_val_df = val_df.sample(n=val_size, random_state=4242).reset_index(drop=True)
    final_train_df = remaining_train_df.sample(
        n=train_size, random_state=4242
    ).reset_index(drop=True)

    dataset_dict = DatasetDict(
        train=Dataset.from_pandas(final_train_df),
        val=Dataset.from_pandas(final_val_df),
        test=Dataset.from_pandas(final_test_df),
    )

    dataset_id = "EuroEval/uner-sk-mini"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset_dict.push_to_hub(dataset_id, private=True)


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """Process the DataFrame.

    Args:
        df: The DataFrame to process.

    Returns:
        The processed DataFrame.
    """
    df = df.drop_duplicates()

    df["text"] = df["inputs"].apply(lambda x: x.split("nižšie?")[-1].strip())
    df["tokens"] = df["text"].apply(lambda x: _split_with_punctuation(x))
    df["positions"] = df["inputs"].apply(
        lambda x: _get_positions(x.split("nižšie?")[-1].strip())
    )
    df["labels"] = df.apply(
        lambda row: _assign_labels(
            row["tokens"], _parse_targets(row["targets"]), row["positions"], row["text"]
        ),
        axis=1,
    )
    keep_columns = ["tokens", "labels"]
    df = df[keep_columns]
    return df


def _split_with_punctuation(text: str) -> list[str]:
    """Tokenize text, splitting on spaces and treating punctuation as separate tokens.

    Args:
        text: The text to tokenize.

    Returns:
        The tokenized text.
    """
    return re.findall(r"\w+|[^\w\s]", text)


def _parse_targets(target_str: str) -> list[dict]:
    """Parse the targets string into a list of dictionaries.

    Args:
        target_str: The targets string to parse.

    Returns:
        The parsed targets.
    """
    targets_json = json.loads(f"{{{target_str}}}")
    return targets_json.get("Results", [])


def _assign_labels(
    tokens: list[str], results: list[dict], positions: list[int], text: str
) -> np.ndarray:
    """Assign NER tags to tokens based on the results.

    Args:
        tokens: The tokens to assign labels to.
        results: The results to assign labels from.
        positions: The positions of the tokens in the text.
        text: The text.

    Returns:
        The assigned labels.
    """
    labels = np.array(["O"] * len(tokens), dtype=object)
    if not results:
        return labels

    for i, (token, token_start) in enumerate(zip(tokens, positions)):
        token_end = token_start + len(token)
        assert token == text[token_start:token_end], (
            f"Token {token} does not match text {text[token_start:token_end]}"
        )

        for entity in results:
            if token_start >= entity["Start"] and token_end <= entity["End"]:
                if token_start == entity["Start"]:
                    labels[i] = f"B-{entity['TypeName']}"
                else:
                    labels[i] = f"I-{entity['TypeName']}"

    return labels


def _get_positions(text: str) -> list[int]:
    """Get starting positions of tokens, including punctuation and words after spaces.

    Args:
        text: The text to get positions from.

    Returns:
        The starting positions of the tokens.
    """
    positions = []
    is_space_or_end = True

    for i, char in enumerate(text):
        if char.isspace():
            is_space_or_end = True
        elif re.match(r"[^\w\s]", char):
            # Punctuation: record the position
            positions.append(i)
            is_space_or_end = True
        elif is_space_or_end:
            # New word: record the position after a space
            positions.append(i)
            is_space_or_end = False

    return positions


if __name__ == "__main__":
    main()
