# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the HAREM NER dataset and upload it to the HF Hub."""

import logging
import re
import urllib.request
from collections import Counter

import pandas as pd
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi

logging.basicConfig(format="%(asctime)s ⋅ %(message)s", level=logging.INFO)
logger = logging.getLogger("create_harem")

# Constants for dataset sizes
TRAIN_SIZE = 1024
VAL_SIZE = 256
TEST_SIZE = 1024
RANDOM_STATE = 4242

# URL for HAREM dataset
URL = (
    "https://raw.githubusercontent.com/davidsbatista/NER-datasets/master/Portuguese/"
    "HAREM/ColeccaoDouradaHAREM.txt"
)

# Regular expressions for parsing
TAG_RE = re.compile(r"<(/?)(\w+)(?:\s[^>]*)?>")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_RE = re.compile(r"[.!?]+")

# Tag mapping from HAREM to standard NER labels
TAG2LABEL: dict[str, str] = {
    "ORGANIZACAO": "ORG",
    "PESSOA": "PER",
    "LOCAL": "LOC",
    "TEMPO": "MISC",
    "ACONTECIMENTO": "MISC",
    "ABSTRACCAO": "MISC",
    "VALOR": "MISC",
    "VARIADO": "MISC",
    "OBRA": "MISC",
    "OMITIDO": "MISC",
    "COISA": "MISC",
    "ALT": "MISC",
}

# Label to ID mapping
LABEL2ID = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}

# ID to label mapping for conversion
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Counters for tracking
SEEN_TAGS: Counter = Counter()
SEEN_LABELS: Counter = Counter()


def main() -> None:
    """Create the HAREM NER dataset and upload it to the HF Hub."""
    # Download the HAREM dataset
    logger.info("Downloading HAREM dataset...")
    content = _download(URL)

    # Process the data
    logger.info("Processing HAREM data...")
    examples = _process_harem_data(content)

    logger.info(f"Total examples processed: {len(examples)}")
    logger.info("\nTag usage:")
    for tag, count in SEEN_TAGS.most_common():
        logger.info(f"  {tag:15s}: {count}")

    logger.info("\nLabel frequencies:")
    for label, count in SEEN_LABELS.most_common():
        logger.info(f"  {label:10s}: {count}")

    # Convert to DataFrame
    df = pd.DataFrame(examples)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Ensure we have enough examples for the desired splits
    total_needed = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    if len(df) < total_needed:
        logger.info(
            f"Warning: Only {len(df)} examples available, but {total_needed} needed"
        )
        # Adjust sizes proportionally
        ratio = len(df) / total_needed
        train_size = int(TRAIN_SIZE * ratio)
        val_size = int(VAL_SIZE * ratio)
        test_size = len(df) - train_size - val_size
    else:
        train_size = TRAIN_SIZE
        val_size = VAL_SIZE
        test_size = TEST_SIZE

    # Create splits
    train_df = df[:train_size].reset_index(drop=True)
    val_df = df[train_size : train_size + val_size].reset_index(drop=True)
    test_df = df[train_size + val_size : train_size + val_size + test_size].reset_index(
        drop=True
    )

    logger.info("\nDataset splits:")
    logger.info(f"  Train: {len(train_df)} examples")
    logger.info(f"  Validation: {len(val_df)} examples")
    logger.info(f"  Test: {len(test_df)} examples")

    # Convert labels from strings to IDs for consistency with Spanish script
    def convert_labels_to_ids(labels: list[str]) -> list[int]:
        return [LABEL2ID[label] for label in labels]

    train_df["labels"] = train_df["labels"].apply(convert_labels_to_ids)
    val_df["labels"] = val_df["labels"].apply(convert_labels_to_ids)
    test_df["labels"] = test_df["labels"].apply(convert_labels_to_ids)

    # Convert back to strings for final dataset (matching Spanish script format)
    train_df["labels"] = train_df["labels"].apply(
        lambda ids: [ID2LABEL[id] for id in ids]
    )
    val_df["labels"] = val_df["labels"].apply(lambda ids: [ID2LABEL[id] for id in ids])
    test_df["labels"] = test_df["labels"].apply(
        lambda ids: [ID2LABEL[id] for id in ids]
    )

    # Create dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/harem"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    logger.info(f"\nUploading dataset to {dataset_id}...")
    dataset.push_to_hub(dataset_id, private=True)
    logger.info("Dataset uploaded successfully!")


def _download(url: str) -> str:
    """Download content from URL with proper encoding.

    Args:
        url: The URL to download from.

    Returns:
        The decoded content as a string.
    """
    with urllib.request.urlopen(url) as response:
        return response.read().decode("iso-8859-1")


def _parse_doc(doc: str) -> tuple[list[str], list[int]] | None:
    """Parse a single HAREM document and return tokens and labels (BIO format).

    Args:
        doc: The document string to parse.

    Returns:
        A tuple of (tokens, labels) if the document is valid, otherwise None.
    """
    origem_match = re.search(r"<ORIGEM>\s*(\w+)\s*</ORIGEM>", doc)
    if not origem_match or origem_match.group(1).upper() != "PT":
        return None

    text_match = re.search(r"<TEXTO>(.*?)</TEXTO>", doc, flags=re.S)
    if not text_match:
        return None
    text = text_match.group(1)

    tokens: list[str] = []
    labels: list[int] = []
    stack: list[str] = []
    previous_entity_type: str | None = None
    in_entity = False

    pos = 0
    for tag in TAG_RE.finditer(text):
        pre = text[pos : tag.start()]
        for tok in TOKEN_RE.findall(pre):
            if not stack:
                label = "O"
                previous_entity_type = None
                in_entity = False
            else:
                current_type = TAG2LABEL.get(stack[-1], "MISC")
                if not in_entity or current_type != previous_entity_type:
                    label = f"B-{current_type}"
                    in_entity = True
                else:
                    label = f"I-{current_type}"
                previous_entity_type = current_type

            tokens.append(tok)
            labels.append(LABEL2ID[label])
            SEEN_LABELS[label] += 1

        pos = tag.end()
        closing, name = tag.group(1), tag.group(2)
        SEEN_TAGS[name] += 1

        if name not in TAG2LABEL:
            logger.info(f"Warning: Unknown tag <{name}>")

        if closing:
            if stack and stack[-1] == name:
                stack.pop()
            # Reset entity tracking if closed
            in_entity = False
            previous_entity_type = None
        else:
            stack.append(name)
            in_entity = False  # next token should be B-

    tail = text[pos:]
    for tok in TOKEN_RE.findall(tail):
        if not stack:
            label = "O"
            previous_entity_type = None
            in_entity = False
        else:
            current_type = TAG2LABEL.get(stack[-1], "MISC")
            if not in_entity or current_type != previous_entity_type:
                label = f"B-{current_type}"
                in_entity = True
            else:
                label = f"I-{current_type}"
            previous_entity_type = current_type

        tokens.append(tok)
        labels.append(LABEL2ID[label])
        SEEN_LABELS[label] += 1

    return tokens, labels


def _reconstruct_text(tokens: list[str]) -> str:
    """Reconstruct text from tokens, preserving original spacing.

    Args:
        tokens: List of tokens.

    Returns:
        The reconstructed text as a string.
    """
    if not tokens:
        return ""

    result = []
    for i, token in enumerate(tokens):
        if i == 0:
            # First token always gets added as-is
            result.append(token)
        elif re.match(r"[^\w\s]", token):
            # Punctuation - attach to previous token without space
            result.append(token)
        else:
            # Regular word - add space before
            result.append(" " + token)

    return "".join(result)


def _split_into_sentences(
    tokens: list[str], labels: list[int]
) -> list[tuple[list[str], list[int]]]:
    """Split tokens and labels into sentences.

    Args:
        tokens: List of tokens.
        labels: List of label IDs corresponding to tokens.

    Returns:
        List of (tokens, labels) tuples for each sentence.
    """
    sentences = []
    i = 0
    while i < len(tokens):
        current_tokens = []
        current_labels = []

        while i < len(tokens):
            current_tokens.append(tokens[i])
            current_labels.append(labels[i])

            if SENTENCE_RE.search(tokens[i]):
                i += 1
                # absorb following I-XXX tokens
                while i < len(tokens) and ID2LABEL[labels[i]].startswith("I-"):
                    current_tokens.append(tokens[i])
                    current_labels.append(labels[i])
                    i += 1
                break  # break inner loop

            i += 1

        sentences.append((current_tokens, current_labels))

    return sentences


def _process_harem_data(raw: str) -> list[dict]:
    """Process raw HAREM data into structured format.

    Args:
        raw: Raw string containing HAREM data.

    Returns:
        List of dicts, each with 'tokens', 'labels', and 'text' for a sentence.
    """
    docs = raw.split("<DOC>")
    examples = []

    for doc in docs:
        parsed = _parse_doc(doc)
        if parsed:
            tokens, labels = parsed

            # Split into sentences
            sentences = _split_into_sentences(tokens, labels)

            # Create examples for each sentence
            for sent_tokens, sent_labels in sentences:
                if len(sent_tokens) > 0:  # Only add non-empty sentences
                    # Convert labels back to strings
                    label_strings = [ID2LABEL[label] for label in sent_labels]

                    examples.append(
                        {
                            "tokens": sent_tokens,
                            "labels": label_strings,
                            "text": _reconstruct_text(sent_tokens),
                        }
                    )

    return examples


if __name__ == "__main__":
    main()
