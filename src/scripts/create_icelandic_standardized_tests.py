# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "beautifulsoup4==4.12.3",
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "pymupdf==1.25.3",
#     "requests==2.32.3",
# ]
# ///

"""Create the Icelandic standardized tests datasets and upload to HF Hub."""

import io
import logging
import re
from urllib.parse import urljoin, urlparse

import fitz
import pandas as pd
import requests
from bs4 import BeautifulSoup
from constants import CHOICES_MAPPING
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi

logging.basicConfig(format="%(asctime)s ⋅ %(message)s", level=logging.INFO)
logger = logging.getLogger("create_icelandic_standardized_tests")


# Year-specific pages listing exams and answer keys
# "Prófhefti" = question booklet, "Matsreglur" = answer key / marking guidelines
YEAR_PAGES: list[tuple[str, str]] = [
    ("2013", "https://mms.is/prof-og-svor-2013"),
    ("2014", "https://mms.is/prof-og-svor-2014"),
    ("2015", "https://mms.is/prof-og-svor-2015"),
    ("2016", "https://www.mms.is/prof-og-svor-2016"),
    ("2017", "https://www.mms.is/prof-og-svor-2017"),
]

# Keywords used to identify subjects in page headings (lowercase)
SUBJECT_KEYWORDS: dict[str, list[str]] = {
    "is": ["íslenska", "islenska", "íslensku"],
    "math": ["stærðfræði", "staerdfraedi", "stærðfræðu", "stærðfr"],
}

LABEL_LETTERS = ["a", "b", "c", "d"]


def main() -> None:
    """Create the Icelandic standardized tests datasets and upload to HF Hub."""
    all_is_samples: list[dict] = []
    all_math_samples: list[dict] = []

    for year, page_url in YEAR_PAGES:
        logger.info(f"Fetching year page for {year}: {page_url}")
        try:
            pdf_links = fetch_pdf_links(year=year, page_url=page_url)
        except Exception as e:
            logger.warning(f"Failed to fetch PDF links for {year}: {e}")
            continue

        for subject, prof_urls, matsreglur_url in pdf_links:
            logger.info(
                f"Processing {subject} test from {year} "
                f"({len(prof_urls)} question sheet(s))..."
            )
            try:
                answers_pdf = download_pdf(url=matsreglur_url)
            except Exception as e:
                logger.warning(
                    f"Failed to download answer key for {subject} {year}: {e}"
                )
                continue

            combined_samples: list[dict] = []
            for prof_url in prof_urls:
                try:
                    test_pdf = download_pdf(url=prof_url)
                except Exception as e:
                    logger.warning(
                        f"Failed to download question sheet {prof_url} "
                        f"for {subject} {year}: {e}"
                    )
                    continue
                try:
                    samples = extract_multiple_choice_questions(
                        test_pdf=test_pdf,
                        answers_pdf=answers_pdf,
                        year=year,
                        subject=subject,
                    )
                    combined_samples.extend(samples)
                except Exception as e:
                    logger.warning(
                        f"Failed to extract questions from {subject} {year} "
                        f"({prof_url}): {e}"
                    )

            if subject == "is":
                all_is_samples.extend(combined_samples)
            else:
                all_math_samples.extend(combined_samples)

            logger.info(
                f"Extracted {len(combined_samples)} questions from {subject} {year}."
            )

    for subject, samples in [("is", all_is_samples), ("math", all_math_samples)]:
        if not samples:
            logger.warning(f"No samples found for {subject}. Skipping.")
            continue

        df = pd.DataFrame(samples)
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

        logger.info(f"Total {subject} samples after deduplication: {len(df)}")

        # Create splits (use all available data, keeping test years separate)
        # Use the oldest year(s) for train, middle year(s) for val, newest for test
        years = sorted(df["year"].unique())
        test_years = years[-2:] if len(years) >= 3 else years[-1:]
        val_years = years[-3:-2] if len(years) >= 3 else []
        train_years = [y for y in years if y not in test_years and y not in val_years]

        test_df = df[df["year"].isin(test_years)].copy()
        val_df = df[df["year"].isin(val_years)].copy() if val_years else pd.DataFrame()
        train_df = (
            df[df["year"].isin(train_years)].copy() if train_years else pd.DataFrame()
        )

        # If we don't have enough splits, fall back to random splitting
        if len(train_df) == 0 or len(val_df) == 0:
            logger.warning(
                f"Not enough years for clean splits for {subject}. "
                "Using random splitting."
            )
            df_shuffled = df.sample(frac=1, random_state=4242).reset_index(drop=True)
            n = len(df_shuffled)
            test_size = min(n // 2, 2048)
            val_size = min(n // 4, 256)
            train_size = min(max(0, n - test_size - val_size), 1024)
            test_df = df_shuffled.iloc[:test_size].copy()
            val_df = df_shuffled.iloc[test_size : test_size + val_size].copy()
            train_df = df_shuffled.iloc[
                test_size + val_size : test_size + val_size + train_size
            ].copy()

        # Keep only the necessary columns
        train_df = train_df[["text", "label"]].reset_index(drop=True)
        val_df = val_df[["text", "label"]].reset_index(drop=True)
        test_df = test_df[["text", "label"]].reset_index(drop=True)

        logger.info(
            f"Split sizes for {subject}: train={len(train_df)}, "
            f"val={len(val_df)}, test={len(test_df)}"
        )

        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
                "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
                "test": Dataset.from_pandas(test_df, split=Split.TEST),
            }
        )

        if subject == "is":
            dataset_id = "EuroEval/icelandic-lang-tests"
        else:
            dataset_id = "EuroEval/icelandic-math-tests"

        HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
        dataset_dict.push_to_hub(dataset_id, private=True)
        logger.info(f"Pushed {subject} dataset to {dataset_id}.")


def fetch_pdf_links(year: str, page_url: str) -> list[tuple[str, list[str], str]]:
    """Fetch "Prófhefti" and "Matsreglur" PDF links from a year page.

    Scrapes the given year page and returns PDF link groups, one per subject
    found. Each entry is (subject, [profhefti_url, ...], matsreglur_url).

    Args:
        year:
            The year string (used only for logging).
        page_url:
            The URL of the year page listing the exams.

    Returns:
        A list of (subject, prof_urls, matsreglur_url) triples for each
        subject found on the page. subject is "is" or "math".
    """
    response = requests.get(url=page_url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    # Derive the base URL (scheme + host) from the page URL
    parsed = urlparse(page_url)
    url_base = f"{parsed.scheme}://{parsed.netloc}"

    def resolve(href: str) -> str:
        return urljoin(url_base, href)

    def detect_subject(text: str, href: str = "") -> str | None:
        """Return the subject key matching text/href, or None."""
        for subj, keywords in SUBJECT_KEYWORDS.items():
            if any(kw in text or kw in href for kw in keywords):
                return subj
        return None

    # Walk through headings and links in document order, tracking subject context.
    results: dict[str, dict[str, list[str]]] = {}
    current_subject: str | None = None

    heading_tags = ["h1", "h2", "h3", "h4", "h5", "h6"]
    for element in soup.find_all(heading_tags + ["a"]):
        text_lower = element.get_text(separator=" ", strip=True).lower()

        # Detect subject headings
        if element.name in heading_tags:
            found = detect_subject(text_lower)
            if found is not None:
                current_subject = found
                if found not in results:
                    results[found] = {"profhefti": [], "matsreglur": []}

        # Detect links
        elif element.name == "a":
            href = element.get("href", "")
            if not href or not href.lower().endswith(".pdf"):
                continue

            link_text = element.get_text(strip=True).lower()
            full_url = resolve(href)

            # Fall back to detecting subject from link text / href
            link_subject = current_subject or detect_subject(link_text, href.lower())
            if link_subject is None:
                continue

            if link_subject not in results:
                results[link_subject] = {"profhefti": [], "matsreglur": []}

            if "prófhefti" in link_text or "profhefti" in link_text:
                results[link_subject]["profhefti"].append(full_url)
            elif "matsreglur" in link_text or "matsregl" in link_text:
                results[link_subject]["matsreglur"].append(full_url)

    output: list[tuple[str, list[str], str]] = []
    for subj, links in results.items():
        if not links["profhefti"] or not links["matsreglur"]:
            logger.warning(
                f"Incomplete links for subject={subj} year={year}: "
                f"profhefti={links['profhefti']}, matsreglur={links['matsreglur']}"
            )
            continue
        # Use the last matsreglur link if there are multiple (shouldn't happen)
        output.append((subj, links["profhefti"], links["matsreglur"][-1]))

    logger.info(
        f"Found {len(output)} subject(s) on year page {year}: "
        + ", ".join(s for s, _, _ in output)
    )
    return output


def download_pdf(url: str) -> bytes:
    """Download a PDF from the given URL.

    Args:
        url:
            The URL of the PDF to download.

    Returns:
        The bytes of the downloaded PDF.
    """
    response = requests.get(url=url, timeout=30)
    response.raise_for_status()
    return response.content


def extract_multiple_choice_questions(
    test_pdf: bytes, answers_pdf: bytes, year: str, subject: str
) -> list[dict]:
    """Extract multiple-choice questions from a test PDF and its answers PDF.

    Args:
        test_pdf:
            The bytes of the test PDF.
        answers_pdf:
            The bytes of the answers PDF.
        year:
            The year of the test.
        subject:
            The subject of the test ("is" for Icelandic, "math" for mathematics).

    Returns:
        A list of dicts with keys "text", "label", and "year".
    """
    test_text = extract_text_from_pdf(pdf_bytes=test_pdf)
    answers_text = extract_text_from_pdf(pdf_bytes=answers_pdf)

    answer_key = parse_answer_key(answers_text=answers_text)
    questions = parse_multiple_choice_questions(test_text=test_text)

    samples = []
    for q_num, question_data in questions.items():
        if q_num not in answer_key:
            continue
        correct_answer = answer_key[q_num].lower()
        if correct_answer not in LABEL_LETTERS:
            continue
        text = format_question_text(
            question=question_data["question"], options=question_data["options"]
        )
        if text is None:
            continue
        samples.append({"text": text, "label": correct_answer, "year": year})

    return samples


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract the text content from a PDF.

    Args:
        pdf_bytes:
            The PDF content as bytes.

    Returns:
        The extracted text content.
    """
    with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def parse_answer_key(answers_text: str) -> dict[int, str]:
    """Parse the answer key from the answers PDF text.

    Args:
        answers_text:
            The text extracted from the answers PDF.

    Returns:
        A dict mapping question number to the correct answer letter.
    """
    answer_key: dict[int, str] = {}
    # Match patterns like "1. a", "1) a", "1: a", "1 a", "1.a" etc.
    # Also handles answers in tables or listed formats
    patterns = [
        r"(?:^|\n)\s*(\d+)[.):\s]+([a-dA-D])\b",
        r"(?:^|\n)\s*(\d+)\s*[-–]\s*([a-dA-D])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern=pattern, string=answers_text, flags=re.MULTILINE)
        for q_num_str, answer in matches:
            q_num = int(q_num_str)
            if q_num not in answer_key:
                answer_key[q_num] = answer.lower()
    return answer_key


def parse_multiple_choice_questions(test_text: str) -> dict[int, dict]:
    """Parse multiple-choice questions from the test PDF text.

    Args:
        test_text:
            The text extracted from the test PDF.

    Returns:
        A dict mapping question number to a dict with "question" and "options" keys.
    """
    questions: dict[int, dict] = {}

    # Split the text into lines for easier processing
    lines = test_text.split("\n")
    lines = [line.strip() for line in lines if line.strip()]

    i = 0
    while i < len(lines):
        # Look for a question number at the start of a line
        q_match = re.match(r"^(\d+)[.)]\s*(.+)$", lines[i])
        if not q_match:
            i += 1
            continue

        q_num = int(q_match.group(1))
        question_text = q_match.group(2).strip()

        # Collect additional question lines until we hit options
        i += 1
        while i < len(lines):
            # Check if this line starts an option
            if re.match(r"^[a-dA-D][.)]\s*", lines[i]):
                break
            # Check if this line starts a new question
            if re.match(r"^\d+[.)]\s*", lines[i]):
                break
            question_text += " " + lines[i]
            i += 1

        # Collect options (a, b, c, d)
        options: dict[str, str] = {}
        while i < len(lines):
            opt_match = re.match(r"^([a-dA-D])[.)]\s*(.+)$", lines[i])
            if not opt_match:
                break
            opt_letter = opt_match.group(1).lower()
            opt_text = opt_match.group(2).strip()

            # Collect continuation lines for this option
            i += 1
            while i < len(lines):
                if re.match(r"^[a-dA-D][.)]\s*", lines[i]):
                    break
                if re.match(r"^\d+[.)]\s*", lines[i]):
                    break
                opt_text += " " + lines[i]
                i += 1

            options[opt_letter] = opt_text

        # Only keep questions with exactly 4 options (a, b, c, d)
        if set(options.keys()) == {"a", "b", "c", "d"}:
            questions[q_num] = {"question": question_text.strip(), "options": options}

    return questions


def format_question_text(question: str, options: dict[str, str]) -> str | None:
    """Format a multiple-choice question as a text string for EuroEval.

    Args:
        question:
            The question text.
        options:
            A dict mapping option letters to their text.

    Returns:
        The formatted text string, or None if the question is invalid.
    """
    if not question or len(question) < 10:
        return None
    if not all(options.get(letter) for letter in LABEL_LETTERS):
        return None

    choices_label = CHOICES_MAPPING["is"]
    text = (
        f"{question}\n"
        f"{choices_label}:\n"
        f"a. {options['a']}\n"
        f"b. {options['b']}\n"
        f"c. {options['c']}\n"
        f"d. {options['d']}"
    )
    return text


if __name__ == "__main__":
    main()
