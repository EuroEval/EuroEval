"""Create the Spanish XGLUE NER dataset from local files."""

import os

import pandas as pd
from datasets import Dataset, DatasetDict, Split


def load_xglue_data(file_path: str) -> pd.DataFrame:
    """Load data from a file and convert to the required format.

    Args:
        file_path (str): The path to the file containing the data.

    Returns:
            pd.DataFrame: A pandas DataFrame containing the data.
    """
    tokens = []
    labels = []
    current_tokens = []
    current_labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # Split line into token and label
                parts = line.split()
                if len(parts) == 2:  # Only process lines with token and label
                    token, label = parts
                    current_tokens.append(token)
                    current_labels.append(label)
            else:
                # Empty line indicates end of sentence
                if current_tokens:
                    tokens.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens = []
                    current_labels = []

        # Add the last sentence if file doesn't end with empty line
        if current_tokens:
            tokens.append(current_tokens)
            labels.append(current_labels)

    def join_tokens(token_list: list[str]) -> str:
        """Join tokens into a string.

        Args:
            token_list (list[str]): The list of tokens to join.

        Returns:
            str: The joined string.
        """
        if not token_list:
            return ""

        result = ""
        for i, token in enumerate(token_list):
            # First token always gets added without preceding space
            if i == 0:
                result += token
                continue

            # Don't add space before closing punctuation
            if token in [",", ".", ";", ":", "!", "?", ")", "]", "}"]:
                result += token
            # Don't add space after opening punctuation
            elif token_list[i - 1] in ["(", "[", "{"]:
                result += token
            else:
                result += " " + token

        return result

    return pd.DataFrame(
        {
            "tokens": tokens,
            "labels": labels,  # Changed from ner_tags to labels directly
            "text": [join_tokens(t) for t in tokens],  # Add text column directly
        }
    )


def main() -> None:
    """Create the Spanish XGLUE NER dataset."""
    # Define base directory. Download https://microsoft.github.io/XGLUE/
    base_dir = "xglue_full_dataset/NER"

    # Load Spanish datasets
    dev_df = load_xglue_data(os.path.join(base_dir, "es.dev"))
    test_df = load_xglue_data(os.path.join(base_dir, "es.test"))

    for df in [dev_df, test_df]:
        for token_list, ner_tag_list in zip(df["tokens"], df["labels"]):
            # Sanity check that the number of tokens and named entity tags are equal
            assert len(token_list) == len(ner_tag_list), (
                "The number of tokens and named entity tags are not equal."
            )

            # Fix invalid I-tags
            invalid_i_ner_tags = [
                ner_tag
                for token_idx, ner_tag in enumerate(ner_tag_list)
                if ner_tag.startswith("I-")
                and (
                    token_idx == 0
                    or ner_tag_list[token_idx - 1] not in {f"B-{ner_tag[2:]}", ner_tag}
                )
            ]
            while invalid_i_ner_tags:
                for invalid_i_ner_tag in invalid_i_ner_tags:
                    ner_tag_list[ner_tag_list.index(invalid_i_ner_tag)] = (
                        f"B-{invalid_i_ner_tag[2:]}"
                    )
                invalid_i_ner_tags = [
                    ner_tag
                    for token_idx, ner_tag in enumerate(ner_tag_list)
                    if ner_tag.startswith("I-")
                    and (
                        token_idx == 0
                        or ner_tag_list[token_idx - 1]
                        not in {f"B-{ner_tag[2:]}", ner_tag}
                    )
                ]

    # Create validation split
    val_size = 256
    val_df = dev_df.sample(
        n=val_size,
        random_state=4242,
        weights=[5.0 if len(set(labels)) > 1 else 1.0 for labels in dev_df["labels"]],
    )

    # Create test split
    test_size = 1024
    test_df = test_df.sample(
        n=test_size,
        random_state=4242,
        weights=[5.0 if len(set(labels)) > 1 else 1.0 for labels in test_df["labels"]],
    )

    # Reset the index
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/xglue-ner-es-mini"

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
