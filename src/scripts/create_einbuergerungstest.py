# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "beautifulsoup4==4.12.3",
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "lxml==5.3.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Einbürgerungstest dataset and upload it to the HF Hub."""

import re

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi

from .constants import CHOICES_MAPPING

BASE_URL = "https://www.einbuergerungstest-online.eu"
FRAGEN_URL = f"{BASE_URL}/fragen/"


def main() -> None:
    """Create the Einbürgerungstest dataset and upload it to the HF Hub."""
    # Scrape the questions
    df = scrape_questions()

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    total = len(df)
    assert total >= 100, f"Expected at least 100 questions, got {total}"

    # Create splits: test gets the largest share, then val, then train
    test_size = min(128, total // 2)
    val_size = min(64, total // 4)
    train_size = total - test_size - val_size

    assert train_size > 0, "Not enough data for training split"

    df_shuffled = df.sample(frac=1, random_state=4242).reset_index(drop=True)

    train_df = df_shuffled.iloc[:train_size].reset_index(drop=True)
    val_df = df_shuffled.iloc[train_size : train_size + val_size].reset_index(drop=True)
    test_df = df_shuffled.iloc[train_size + val_size :].reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df, split=Split.TEST),
        }
    )

    # Create dataset ID
    dataset_id = "EuroEval/einbuergerungstest"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def scrape_questions() -> pd.DataFrame:
    """Scrape questions from the Einbürgerungstest website.

    Returns:
        A DataFrame with columns 'text' and 'label'.
    """
    response = requests.get(FRAGEN_URL, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")

    texts: list[str] = []
    labels: list[str] = []

    # The page lists all questions; find each question block
    # Try common patterns used on quiz/test websites
    question_blocks = soup.find_all("div", class_=re.compile(r"frage|question|item", re.I))
    if not question_blocks:
        question_blocks = soup.find_all("article")
    if not question_blocks:
        question_blocks = soup.find_all("li", class_=re.compile(r"frage|question", re.I))

    for block in question_blocks:
        result = _parse_question_block(block)
        if result is not None:
            text, label = result
            texts.append(text)
            labels.append(label)

    if not texts:
        # Fallback: try parsing the full page differently
        texts, labels = _parse_page_fallback(soup)

    df = pd.DataFrame({"text": texts, "label": labels})
    return df


def _parse_question_block(
    block: Tag,
) -> tuple[str, str] | None:
    """Parse a single question block from the HTML.

    Args:
        block:
            The HTML block containing the question.

    Returns:
        A tuple of (text, label) where text is the formatted question with choices
        and label is the correct answer letter ('a', 'b', 'c', or 'd'), or None if
        the block could not be parsed.
    """
    # Try to extract question text
    question_el = block.find(class_=re.compile(r"frage|question.?text|frage.?text", re.I))
    if question_el is None:
        question_el = block.find(["h2", "h3", "h4", "p"])
    if question_el is None:
        return None

    question_text = clean_text(question_el.get_text())
    if not question_text:
        return None

    # Try to find answer options
    options: dict[str, str] = {}
    correct_label: str | None = None

    # Look for list items or divs with options
    option_els = block.find_all(
        class_=re.compile(r"antwort|answer|option|choice", re.I)
    )
    if not option_els:
        option_els = block.find_all("li")

    for i, opt_el in enumerate(option_els[:4]):
        letter = "abcd"[i]
        option_text = clean_text(opt_el.get_text())

        # Strip leading letter prefix if present (e.g. "a. " or "a) ")
        option_text = re.sub(r"^[a-d][.)]\s*", "", option_text, flags=re.I)

        if not option_text:
            continue

        options[letter] = option_text

        # Check if this option is marked as correct
        classes = " ".join(opt_el.get("class", []))
        if re.search(r"richtig|correct|right", classes, re.I):
            correct_label = letter

    if len(options) != 4 or correct_label is None:
        return None

    text = (
        question_text
        + f"\n{CHOICES_MAPPING['de']}:\n"
        + "\n".join(f"{letter}. {options[letter]}" for letter in "abcd")
    )

    return text, correct_label


def _parse_page_fallback(soup: BeautifulSoup) -> tuple[list[str], list[str]]:
    """Fallback parser for when the main parser fails.

    Tries to find questions and answers from the page using a different approach,
    looking for numbered questions and their options.

    Args:
        soup:
            The BeautifulSoup object for the full page.

    Returns:
        A tuple of (texts, labels) lists.
    """
    texts: list[str] = []
    labels: list[str] = []

    # Look for question patterns in any text block
    all_text = soup.get_text(separator="\n")
    lines = [line.strip() for line in all_text.split("\n") if line.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]

        # Look for question numbers like "1.", "Frage 1", etc.
        if re.match(r"^(Frage\s+)?\d+[.)]\s+\S", line):
            question_text = re.sub(r"^(Frage\s+)?\d+[.)]\s+", "", line).strip()

            options: dict[str, str] = {}
            correct_label: str | None = None
            j = i + 1

            while j < len(lines) and len(options) < 4:
                opt_line = lines[j]
                m = re.match(r"^([a-d])[.)]\s+(.+)$", opt_line, re.I)
                if m:
                    letter = m.group(1).lower()
                    option_text = m.group(2).strip()
                    options[letter] = option_text
                j += 1

            # Check the next lines for a correct answer indicator
            for k in range(j, min(j + 3, len(lines))):
                m = re.match(r"^[Aa]ntwort[:\s]+([a-d])", lines[k])
                if m:
                    correct_label = m.group(1).lower()
                    break

            if len(options) == 4 and correct_label is not None:
                text = (
                    question_text
                    + f"\n{CHOICES_MAPPING['de']}:\n"
                    + "\n".join(f"{letter}. {options[letter]}" for letter in "abcd")
                )
                texts.append(text)
                labels.append(correct_label)
                i = j
                continue

        i += 1

    return texts, labels


def clean_text(text: str) -> str:
    """Clean some text.

    Args:
        text:
            The text to clean.

    Returns:
        The cleaned text.
    """
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    main()
