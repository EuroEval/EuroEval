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


AnswerKeyType = (
    Annotated[int, Field(ge=1, le=4)]
    | Annotated[str, Field(min_length=1, max_length=1)]
)


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


# Pydantic models for structured GPT output ---------------------------------


class Option(BaseModel):
    """A single multiple-choice question with four options."""

    answer_key: AnswerKeyType
    answer: str


class McQuestion(BaseModel):
    """A single multiple-choice question with four options."""

    question: Annotated[str, Field(min_length=10)]
    options: list[Option]
    skip: bool = False


class GeneralReadingPassage(BaseModel):
    """A long general reading passage."""

    id: int
    text: Annotated[str, Field(min_length=100)]


class McQuestionWithPassage(BaseModel):
    """A single multiple-choice question with a passage and four options."""

    passage_id: int
    question: Annotated[str, Field(min_length=10)]
    options: list[Option]
    skip: bool = False


class AnswerEntry(BaseModel):
    """A single answer-key entry."""

    question_id: int
    answer_key: AnswerKeyType

    def model_post_init(self, __context: dict) -> None:
        """Post-processing for AnswerEntry."""
        if isinstance(self.answer_key, str):
            self.answer_key = self.answer_key.upper()


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
                questions = extract_questions(
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

            for question_id, question in questions.items():
                if question_id not in answer_key:
                    logger.warning(
                        f"Question {question_id} not found in answer key for "
                        f"{subject} {year}"
                    )
                    continue

                text = format_question_text(
                    question=question.question,
                    options={
                        letter: option.answer
                        for letter, option in zip(
                            "abcd", sorted(question.options, key=lambda x: x.answer_key)
                        )
                    },
                )

                answer_key_to_letter = {
                    option.answer_key: letter
                    for option, letter in zip(
                        sorted(question.options, key=lambda x: x.answer_key), "abcd"
                    )
                }
                try:
                    label = answer_key_to_letter[answer_key[question_id]]
                except KeyError:
                    logger.warning(
                        f"Answer key for question {question_id} not found in answer "
                        f"key for {subject} {year}"
                    )
                    continue

                if text is None:
                    logger.warning(
                        f"Failed to format question {question_id} for {subject} {year}"
                    )
                    continue
                combined_samples.append({"text": text, "label": label, "year": year})

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

        train_samples = 64
        val_samples = 128

        # Create splits based on year: oldest → train, middle → val, newest → test.
        df = df.sort_values(by="year")
        train_df = df.iloc[:train_samples].copy().reset_index(drop=True)
        val_df = (
            df.iloc[train_samples : train_samples + val_samples]
            .copy()
            .reset_index(drop=True)
        )
        test_df = df.iloc[train_samples + val_samples :].copy().reset_index(drop=True)

        logger.info(
            f"Split sizes for {subject}: train={len(train_df)}, val={len(val_df)}, "
            f"test={len(test_df)}"
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
) -> dict[int, McQuestion | McQuestionWithPassage]:
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
        A dict mapping question numbers to the extracted questions.
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
                "If a question has an associated image which is required to be able "
                "to answer the question, skip the question. "
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
    assert result is not None, f"Failed to parse completion for {subject}."

    questions: dict[int, McQuestion | McQuestionWithPassage] = {
        i: getattr(result, f"question_{i}")
        for i in range(1, len(ids) + 1)
        if hasattr(result, f"question_{i}")
        and not getattr(result, f"question_{i}").skip
    }

    if subject == "is":
        questions_to_drop: list[int] = list()
        passages: list[GeneralReadingPassage] = result.passages
        for question_id, question in questions.items():
            passage_candidates = [p for p in passages if p.id == question.passage_id]
            if not passage_candidates:
                logger.warning(
                    f"Passage {question.passage_id} not found for {subject}. Removing "
                    "the question."
                )
                questions_to_drop.append(question_id)
            else:
                passage = passage_candidates[0]
                question.question = f"{passage.text}\n\n{question.question}"
        for question_id in questions_to_drop:
            del questions[question_id]

    return questions


def extract_answer_key(pdf_bytes: bytes, client: OpenAI) -> dict[int, AnswerKeyType]:
    """Use GPT-4.1 vision to extract the answer key from a marking guide PDF.

    Args:
        pdf_bytes:
            The PDF bytes of the marking guide.
        client:
            An authenticated OpenAI client.

    Returns:
        A dict mapping question number (int) to the correct answer key.
    """
    images = pdf_to_images(pdf_bytes=pdf_bytes)

    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "These are pages from an Icelandic primary school exam marking guide "
                "(matsreglur). Please extract the answer key for all multiple-choice "
                "questions. These can be either upper case letters, or numerals. If "
                "both are present, prefer the letter. For each question return its "
                "ID key and the single correct answer key corresponding to the "
                "correct answer."
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
    return {entry.question_id: entry.answer_key for entry in result.answers}


def format_question_text(question: str, options: dict[str, str]) -> str:
    """Format a multiple-choice question as a text string for EuroEval.

    Args:
        question:
            The question text.
        options:
            A dict mapping option letters to their text.

    Returns:
        The formatted text string.
    """
    choices_label = CHOICES_MAPPING["is"]
    text = f"{question}\n{choices_label}:\n" + "\n".join(
        [f"{letter}. {option}" for letter, option in options.items()]
    )
    return text


if __name__ == "__main__":
    main()
