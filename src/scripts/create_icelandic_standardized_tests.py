# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "openai==1.66.5",
#     "pandas==2.2.0",
#     "pymupdf==1.25.3",
#     "pydantic==2.6.0",
#     "python-dotenv==1.0.1",
#     "requests==2.32.3",
# ]
# ///

"""Create the Icelandic standardized tests datasets and upload to HF Hub."""

import base64
import io
import logging
import os
from typing import Annotated

import fitz
import pandas as pd
import requests
from constants import CHOICES_MAPPING
from datasets import Dataset, DatasetDict, Split
from dotenv import load_dotenv
from huggingface_hub import HfApi
from openai import OpenAI
from pydantic import BaseModel, Field, create_model

load_dotenv()

logging.basicConfig(format="%(asctime)s ⋅ %(message)s", level=logging.INFO)
logger = logging.getLogger("create_icelandic_standardized_tests")

GPT_MODEL = "gpt-5-mini"


# Each entry is (year, subject, [question_urls], answers_url).
# subject: "is" for Icelandic language, "math" for mathematics.
# Sources:
#   2013: https://mms.is/prof-og-svor-2013
#   2014: https://mms.is/prof-og-svor-2014
#   2015: https://mms.is/prof-og-svor-2015
#   2016: https://www.mms.is/prof-og-svor-2016
#   2017: https://www.mms.is/prof-og-svor-2017
TEST_PDFS: list[tuple[str, str, list[str], str]] = [
    # ── 2013 ─────────────────────────────────────────────────────────────────
    # Icelandic
    (
        "2013",
        "is",
        ["https://mms.is/sites/mms.is/files/1013_isl.pdf"],
        "https://mms.is/sites/mms.is/files/mat_2013_isl.pdf",
    ),
    (
        "2013",
        "is",
        [
            "https://mms.is/sites/mms.is/files/0713_isl1.pdf",
            "https://mms.is/sites/mms.is/files/0713_isl2.pdf",
        ],
        "https://mms.is/sites/mms.is/files/mat_2013_isl_7bekk.pdf",
    ),
    (
        "2013",
        "is",
        [
            "https://mms.is/sites/mms.is/files/0413_isl1.pdf",
            "https://mms.is/sites/mms.is/files/0413_isl2.pdf",
        ],
        "https://mms.is/sites/mms.is/files/mat_2013_isl_4bekk.pdf",
    ),
    # Math
    (
        "2013",
        "math",
        ["https://mms.is/sites/mms.is/files/1013_sta.pdf"],
        "https://mms.is/sites/mms.is/files/mat_2013_sta.pdf",
    ),
    (
        "2013",
        "math",
        [
            "https://mms.is/sites/mms.is/files/0713_sta1.pdf",
            "https://mms.is/sites/mms.is/files/0713_sta2.pdf",
        ],
        "https://mms.is/sites/mms.is/files/mat_2013_sta_7bekk.pdf",
    ),
    (
        "2013",
        "math",
        [
            "https://mms.is/sites/mms.is/files/0413_sta1.pdf",
            "https://mms.is/sites/mms.is/files/0413_sta2.pdf",
        ],
        "https://mms.is/sites/mms.is/files/mat_2013_sta_4bekk.pdf",
    ),
    # ── 2014 ─────────────────────────────────────────────────────────────────
    # Icelandic
    (
        "2014",
        "is",
        ["https://mms.is/sites/mms.is/files/isl_10_2014_hefti.pdf"],
        "https://mms.is/sites/mms.is/files/matsreglur_10_2014_isl_0.pdf",
    ),
    (
        "2014",
        "is",
        [
            "https://mms.is/sites/mms.is/files/isl_07_2014_1hefti.pdf",
            "https://mms.is/sites/mms.is/files/isl_07_2014_2hefti.pdf",
        ],
        "https://mms.is/sites/mms.is/files/matsreglur_07_2014_isl.pdf",
    ),
    (
        "2014",
        "is",
        [
            "https://mms.is/sites/mms.is/files/isl_04_2014_1hefti.pdf",
            "https://mms.is/sites/mms.is/files/isl_04_2014_2hefti.pdf",
        ],
        "https://mms.is/sites/mms.is/files/matsreglur_04_2014_isl.pdf",
    ),
    # Math
    (
        "2014",
        "math",
        ["https://mms.is/sites/mms.is/files/sta_10_2014_hefti.pdf"],
        "https://mms.is/sites/mms.is/files/matsreglur_10_2014_sta.pdf.pdf",
    ),
    (
        "2014",
        "math",
        [
            "https://mms.is/sites/mms.is/files/sta_07_2014_1hefti.pdf",
            "https://mms.is/sites/mms.is/files/sta_07_2014_2hefti.pdf",
        ],
        "https://mms.is/sites/mms.is/files/matsreglur_07_2014_sta.pdf.pdf",
    ),
    (
        "2014",
        "math",
        [
            "https://mms.is/sites/mms.is/files/sta_04_2014_1hefti.pdf",
            "https://mms.is/sites/mms.is/files/sta_04_2014_2hefti.pdf",
        ],
        "https://mms.is/sites/mms.is/files/matsreglur_04_2014_sta.pdf",
    ),
    # ── 2015 ─────────────────────────────────────────────────────────────────
    # Icelandic
    (
        "2015",
        "is",
        ["https://mms.is/sites/mms.is/files/isl_10_2015_heftid.pdf"],
        "https://mms.is/sites/mms.is/files/10_isl_mat_2015.pdf",
    ),
    (
        "2015",
        "is",
        [
            "https://mms.is/sites/mms.is/files/isl_1_7_bekk_2015.pdf",
            "https://mms.is/sites/mms.is/files/isl_2_7_bekk_2015.pdf",
        ],
        "https://mms.is/sites/mms.is/files/7_isl_mat_2015.pdf",
    ),
    (
        "2015",
        "is",
        [
            "https://mms.is/sites/mms.is/files/isl_04_2015_1hefti.pdf",
            "https://mms.is/sites/mms.is/files/isl_04_2015_2hefti.pdf",
        ],
        "https://mms.is/sites/mms.is/files/4_isl_mat_2015.pdf",
    ),
    # Math
    (
        "2015",
        "math",
        ["https://mms.is/sites/mms.is/files/sta_10_2015_hefti.pdf"],
        "https://mms.is/sites/mms.is/files/10_sta_mat_2015.pdf",
    ),
    (
        "2015",
        "math",
        [
            "https://mms.is/sites/mms.is/files/sta_07_2015_1hefti.pdf",
            "https://mms.is/sites/mms.is/files/sta_07_2015_2hefti.pdf",
        ],
        "https://mms.is/sites/mms.is/files/7_sta_mat_2015.pdf",
    ),
    (
        "2015",
        "math",
        [
            "https://mms.is/sites/mms.is/files/sta_04_2015_1hefti.pdf",
            "https://mms.is/sites/mms.is/files/sta_04_2015_2hefti.pdf",
        ],
        "https://mms.is/sites/mms.is/files/4_sta_mat_2015.pdf",
    ),
    # ── 2016 ─────────────────────────────────────────────────────────────────
    # (PDF format changed this year; no 10th-grade tests available)
    # Icelandic
    (
        "2016",
        "is",
        ["https://www.mms.is/sites/mms.is/files/isl7_2016.pdf"],
        "https://www.mms.is/sites/mms.is/files/isl_7b_2016-nyrra.pdf",
    ),
    (
        "2016",
        "is",
        ["https://www.mms.is/sites/mms.is/files/isl4_2016.pdf"],
        "https://www.mms.is/sites/mms.is/files/isl_4b_2016_002.pdf",
    ),
    # Math
    (
        "2016",
        "math",
        ["https://www.mms.is/sites/mms.is/files/stae7_2016.pdf"],
        "https://www.mms.is/sites/mms.is/files/stae_7b_2016.pdf",
    ),
    (
        "2016",
        "math",
        ["https://www.mms.is/sites/mms.is/files/stae4_2016.pdf"],
        "https://www.mms.is/sites/mms.is/files/stae_4b_2016-nyrra.pdf",
    ),
    # ── 2017 ─────────────────────────────────────────────────────────────────
    # (Two alternative tests per grade: A and B)
    # Icelandic
    (
        "2017",
        "is",
        ["https://www.mms.is/sites/mms.is/files/isl7a_5pr.pdf"],
        "https://mms.is/sites/mms.is/files/isl_7b_a.pdf",
    ),
    (
        "2017",
        "is",
        ["https://www.mms.is/sites/mms.is/files/isl7b_5pr.pdf"],
        "https://mms.is/sites/mms.is/files/isl_7b_b.pdf",
    ),
    (
        "2017",
        "is",
        ["https://www.mms.is/sites/mms.is/files/isl4a_5pr.pdf"],
        "https://mms.is/sites/mms.is/files/isl_4b_a.pdf",
    ),
    (
        "2017",
        "is",
        ["https://www.mms.is/sites/mms.is/files/isl4b_5pr.pdf"],
        "https://mms.is/sites/mms.is/files/isl_4b_b.pdf",
    ),
    # Math
    (
        "2017",
        "math",
        ["https://www.mms.is/sites/mms.is/files/stae7a.pdf"],
        "https://mms.is/sites/mms.is/files/stae_7b_a.pdf",
    ),
    (
        "2017",
        "math",
        ["https://www.mms.is/sites/mms.is/files/stae7b.pdf"],
        "https://mms.is/sites/mms.is/files/stae_7b_b.pdf",
    ),
    (
        "2017",
        "math",
        ["https://www.mms.is/sites/mms.is/files/stae4a.pdf"],
        "https://mms.is/sites/mms.is/files/stae_4b_a.pdf",
    ),
    (
        "2017",
        "math",
        ["https://www.mms.is/sites/mms.is/files/stae4b.pdf"],
        "https://mms.is/sites/mms.is/files/stae_4b_b.pdf",
    ),
]

LABEL_LETTERS = ["a", "b", "c", "d"]


# Pydantic models for structured GPT output ---------------------------------


class McOption(BaseModel):
    """A single multiple-choice option."""

    a: str
    b: str
    c: str
    d: str


class McQuestion(BaseModel):
    """A single multiple-choice question with four options."""

    question: str
    options: McOption


class GeneralReadingPassage(BaseModel):
    """A long general reading passage."""

    id: int
    text: Annotated[str, Field(min_length=100)]


class McQuestionWithPassage(BaseModel):
    """A single multiple-choice question with a passage and four options."""

    passage_id: int
    question: str
    options: McOption


class AnswerEntry(BaseModel):
    """A single answer-key entry."""

    number: int
    answer: str


class AnswerKey(BaseModel):
    """Complete answer key extracted from a marking guide."""

    answers: list[AnswerEntry]


# ---------------------------------------------------------------------------


def main() -> None:
    """Create the Icelandic standardized tests datasets and upload to HF Hub."""
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    all_is_samples: list[dict] = []
    all_math_samples: list[dict] = []

    for year, subject, question_urls, answers_url in TEST_PDFS:
        logger.info(
            f"Processing {subject} test from {year} "
            f"({len(question_urls)} question sheet(s))..."
        )
        try:
            answers_pdf = download_pdf(url=answers_url)
        except Exception as e:
            logger.warning(
                f"Failed to download answer key for {subject} {year} "
                f"({answers_url}): {e}"
            )
            continue

        try:
            answer_key = extract_answer_key(pdf_bytes=answers_pdf, client=client)
        except Exception as e:
            logger.warning(f"Failed to extract answer key for {subject} {year}: {e}")
            continue

        combined_samples: list[dict] = []
        for q_url in question_urls:
            try:
                test_pdf = download_pdf(url=q_url)
            except Exception as e:
                logger.warning(
                    f"Failed to download question sheet {q_url} "
                    f"for {subject} {year}: {e}"
                )
                continue
            try:
                output = extract_questions(
                    pdf_bytes=test_pdf,
                    client=client,
                    subject=subject,
                    ids=list(answer_key.keys()),
                )
            except Exception as e:
                logger.warning(
                    f"Failed to extract questions from {subject} {year} ({q_url}): {e}"
                )
                continue

            questions: list[McQuestion] | list[McQuestionWithPassage]
            if subject == "is":
                passages: list[GeneralReadingPassage] = output.passages
                questions = output.questions
                for question in questions:
                    passage = next((p for p in passages if p.id == question.passage_id))
                    question.question = f"{passage.text}\n\n{question.question}"
            else:
                questions = output.questions

            for q in questions:
                correct = answer_key.get(q.number, "")
                if correct not in LABEL_LETTERS:
                    logger.warning(
                        f"Invalid answer for question {q.number} ({correct}) in "
                        f"answer key for {subject} {year}"
                    )
                    continue
                text = format_question_text(
                    question=q.question,
                    options={
                        "a": q.options.a,
                        "b": q.options.b,
                        "c": q.options.c,
                        "d": q.options.d,
                    },
                )
                if text is None:
                    logger.warning(
                        f"Failed to format question {q.number} for {subject} {year}"
                    )
                    continue
                combined_samples.append({"text": text, "label": correct, "year": year})

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

        # Create splits based on year: oldest → train, middle → val, newest → test.
        years = sorted(df["year"].unique())
        test_years = years[-2:] if len(years) >= 3 else years[-1:]
        val_years = years[-3:-2] if len(years) >= 3 else []
        train_years = [y for y in years if y not in test_years and y not in val_years]

        test_df = df[df["year"].isin(test_years)].copy()
        val_df = df[df["year"].isin(val_years)].copy() if val_years else pd.DataFrame()
        train_df = (
            df[df["year"].isin(train_years)].copy() if train_years else pd.DataFrame()
        )

        # Fall back to random splitting if year-based splits are incomplete.
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


def pdf_to_images(pdf_bytes: bytes, dpi: int = 150) -> list[str]:
    """Render each page of a PDF as a base64-encoded PNG string.

    Args:
        pdf_bytes:
            The PDF content as bytes.
        dpi:
            Resolution for rendering (higher = better quality, larger payload).

    Returns:
        A list of base64-encoded PNG strings, one per page.
    """
    images: list[str] = []
    zoom = dpi / 72  # fitz default is 72 dpi
    mat = fitz.Matrix(zoom, zoom)
    with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            png_bytes = pix.tobytes("png")
            images.append(base64.b64encode(png_bytes).decode("utf-8"))
    return images


def _images_to_content(images: list[str]) -> list[dict]:
    """Build the image content blocks for an OpenAI vision message.

    Args:
        images:
            Base64-encoded PNG strings (one per PDF page).

    Returns:
        A list of content dicts with type ``"image_url"``.
    """
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
        }
        for b64 in images
    ]


def extract_questions(
    pdf_bytes: bytes, client: OpenAI, subject: str, ids: list[int]
) -> tuple[str, list[McQuestion]]:
    """Use GPT-4.1 vision to extract multiple-choice questions from a question booklet.

    For Icelandic language booklets (subject ``"is"``) a general reading passage
    accompanies all questions and is also extracted so it can be prepended to each
    question text.

    Args:
        pdf_bytes:
            The PDF bytes of the question booklet.
        client:
            An authenticated OpenAI client.
        subject:
            Subject code: ``"is"`` for Icelandic language, ``"math"`` for mathematics.
        ids:
            A list of question numbers to extract.

    Returns:
        A tuple of (passage, questions) where passage is the general reading text
        (empty string for mathematics booklets) and questions is the list of
        McQuestion objects extracted from the booklet.
    """
    images = pdf_to_images(pdf_bytes=pdf_bytes)

    if subject == "is":
        passage_instruction = (
            "Each question is associated with a long general reading passage that "
            "precedes it (usually on the previous page). Include this in the "
            "'passages' field, being a list of dictionaries with keys 'id' and 'text'. "
            "Every question should have also have a `passage_id` field that refers to "
            "the ID of the general reading passage it refers to. "
        )
    else:
        passage_instruction = ""

    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "These are pages from an Icelandic primary school exam booklet. "
                "Please extract all multiple-choice questions. For each question "
                "return the full question text, and the text of each option. "
                f"{passage_instruction}"
                "Preserve the original Icelandic text exactly."
            ),
        },
        *_images_to_content(images),
    ]

    if subject == "is":
        response_format = create_model(
            "Questions",
            passages=(list[GeneralReadingPassage], ...),
            **{f"question_{i}": (McQuestionWithPassage, ...) for i in ids},
        )
    else:
        response_format = create_model(
            "Questions", **{f"question_{i}": (McQuestion, ...) for i in ids}
        )

    completion = client.beta.chat.completions.parse(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": content}],
        response_format=response_format,
        max_completion_tokens=128_000,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        return "", []
    return result


def extract_answer_key(pdf_bytes: bytes, client: OpenAI) -> dict[int, str]:
    """Use GPT-4.1 vision to extract the answer key from a marking guide PDF.

    Args:
        pdf_bytes:
            The PDF bytes of the marking guide.
        client:
            An authenticated OpenAI client.

    Returns:
        A dict mapping question number (int) to the correct answer letter (a–d).
    """
    images = pdf_to_images(pdf_bytes=pdf_bytes)

    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "These are pages from an Icelandic primary school exam marking guide "
                "(matsreglur). Please extract the answer key for all multiple-choice "
                "questions. This is the numeric 'Reitur' column (values 1-4), "
                "corresponding to a, b, c, or d. For each question return its "
                "ID number and the single correct letter corresponding to the correct "
                "answer."
            ),
        },
        *_images_to_content(images),
    ]

    completion = client.beta.chat.completions.parse(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": content}],
        response_format=AnswerKey,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        return {}
    return {entry.number: entry.answer.lower() for entry in result.answers}


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

    choices_label = CHOICES_MAPPING["is"]
    text = f"{question}\n{choices_label}:\n\n".join(
        [f"{letter}. {option}" for letter, option in options.items()]
    )
    return text


if __name__ == "__main__":
    main()
