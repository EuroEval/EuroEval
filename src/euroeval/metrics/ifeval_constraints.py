"""IFEval instruction-following constraints and metrics."""

import collections
import collections.abc as c
import functools
import json
import logging
import os
import re
import typing as t
from importlib.metadata import version

import langdetect
import nltk
from langdetect import DetectorFactory
from packaging.version import parse as parse_version

from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig

# make langdetect deterministic
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

NLTK_MIN_VERSION = "3.9.1"
_RANK = os.environ.get("LOCAL_RANK", "0")


def download_nltk_resources() -> None:
    """Download 'punkt' if not already installed."""
    assert (nltk_version := parse_version(version("nltk"))) >= parse_version(
        NLTK_MIN_VERSION
    ), (
        f"`nltk` version {nltk_version} is not >= {NLTK_MIN_VERSION}. "
        "Please update `nltk` before proceeding."
    )
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        if _RANK == "0":
            nltk.download("punkt_tab")


download_nltk_resources()


@functools.lru_cache(maxsize=None)
def get_sentence_tokenizer() -> nltk.tokenize.PunktSentenceTokenizer:
    """Get the cached NLTK sentence tokenizer."""
    return nltk.data.load("nltk:tokenizers/punkt/english.pickle")


# all checker functions accept **kwargs to absorb extra fields from the
# dataset's kwargs dict, which contains all possible parameters for all instructions.


def check_keyword_existence(response: str, *, keywords: list[str], **_) -> bool:
    """Check that all keywords exist in the response."""
    if not keywords:
        raise ValueError("keywords must be provided")
    for keyword in keywords:
        if not re.search(keyword, response, flags=re.IGNORECASE):
            return False
    return True


def check_keyword_frequency(
    response: str, *, keyword: str, frequency: int, relation: str, **_
) -> bool:
    """Check keyword appears with required frequency."""
    if not keyword:
        raise ValueError("keyword must be provided")
    actual = len(re.findall(keyword, response, flags=re.IGNORECASE))
    if relation == "less than":
        return actual < frequency
    return actual >= frequency


def check_forbidden_words(response: str, *, forbidden_words: list[str], **_) -> bool:
    """Check that forbidden words don't appear."""
    if not forbidden_words:
        raise ValueError("forbidden_words must be provided")
    for word in forbidden_words:
        if re.search(r"\b" + word + r"\b", response, flags=re.IGNORECASE):
            return False
    return True


def check_letter_frequency(
    response: str, *, letter: str, let_frequency: int, let_relation: str, **_
) -> bool:
    """Check letter appears with required frequency."""
    if not letter or len(letter) != 1:
        raise ValueError("letter must be a single character")
    counts = collections.Counter(response.lower())
    if let_relation == "less than":
        return counts[letter.lower()] < let_frequency
    return counts[letter.lower()] >= let_frequency


def check_number_sentences(
    response: str, *, num_sentences: int, relation: str, **_
) -> bool:
    """Check number of sentences."""
    tokenizer = get_sentence_tokenizer()
    actual = len(tokenizer.tokenize(response))
    if relation == "less than":
        return actual < num_sentences
    return actual >= num_sentences


def check_number_paragraphs(response: str, *, num_paragraphs: int, **_) -> bool:
    """Check number of paragraphs (separated by ***)."""
    paragraphs = re.split(r"\s?\*\*\*\s?", response)
    count = len(paragraphs)
    for i, p in enumerate(paragraphs):
        if not p.strip():
            if i == 0 or i == len(paragraphs) - 1:
                count -= 1
            else:
                return False
    return count == num_paragraphs


def check_number_words(response: str, *, num_words: int, relation: str, **_) -> bool:
    """Check number of words."""
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    actual = len(tokenizer.tokenize(response))
    if relation == "less than":
        return actual < num_words
    return actual >= num_words


def check_nth_paragraph_first_word(
    response: str, *, num_paragraphs: int, nth_paragraph: int, first_word: str, **_
) -> bool:
    """Check paragraph count and first word of nth paragraph."""
    if first_word is None:
        raise ValueError("first_word must be provided")

    paragraphs = re.split(r"\n\n", response)
    count = sum(1 for p in paragraphs if p.strip())

    if nth_paragraph > count:
        return False

    paragraph = paragraphs[nth_paragraph - 1].strip()
    if not paragraph:
        return False

    # extract first word, stripping punctuation
    word = paragraph.split()[0].strip().lstrip("'\"")
    actual_first = ""
    for char in word:
        if char in ".,?!'\"":
            break
        actual_first += char.lower()

    return count == num_paragraphs and actual_first == first_word.lower()


def check_number_placeholders(response: str, *, num_placeholders: int, **_) -> bool:
    """Check minimum number of [placeholder] brackets."""
    placeholders = re.findall(r"\[.*?\]", response)
    return len(placeholders) >= num_placeholders


def check_postscript(response: str, *, postscript_marker: str, **_) -> bool:
    """Check for postscript marker."""
    response = response.lower()
    if postscript_marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif postscript_marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + postscript_marker.lower() + r".*$"
    return bool(re.findall(pattern, response, flags=re.MULTILINE))


def check_number_bullet_lists(response: str, *, num_bullets: int, **_) -> bool:
    """Check exact number of bullet points."""
    bullets1 = re.findall(r"^\s*\*[^\*].*$", response, flags=re.MULTILINE)
    bullets2 = re.findall(r"^\s*-.*$", response, flags=re.MULTILINE)
    return len(bullets1) + len(bullets2) == num_bullets


def check_constrained_response(response: str, **_) -> bool:
    """Check response contains one of the constrained options."""
    options = ("My answer is yes.", "My answer is no.", "My answer is maybe.")
    return any(opt in response.strip() for opt in options)


def check_number_highlighted_sections(
    response: str, *, num_highlights: int, **_
) -> bool:
    """Check minimum highlighted *sections*."""
    count = 0
    for h in re.findall(r"\*[^\n\*]*\*", response):
        if h.strip("*").strip():
            count += 1
    for h in re.findall(r"\*\*[^\n\*]*\*\*", response):
        if h.removeprefix("**").removesuffix("**").strip():
            count += 1
    return count >= num_highlights


def check_multiple_sections(
    response: str, *, section_spliter: str, num_sections: int, **_
) -> bool:
    """Check for Section X markers."""
    pattern = r"\s?" + section_spliter + r"\s?\d+\s?"
    sections = re.split(pattern, response)
    return len(sections) - 1 >= num_sections


def check_json_format(response: str, **_) -> bool:
    """Check response is valid JSON."""
    value = (
        response.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        json.loads(value)
        return True
    except ValueError:
        return False


def check_title(response: str, **_) -> bool:
    """Check for <<title>> format."""
    for title in re.findall(r"<<[^\n]+>>", response):
        if title.lstrip("<").rstrip(">").strip():
            return True
    return False


def check_two_responses(response: str, **_) -> bool:
    """Check for two different responses separated by ******."""
    parts = response.split("******")
    valid = []
    for i, part in enumerate(parts):
        if not part.strip():
            if i != 0 and i != len(parts) - 1:
                return False
        else:
            valid.append(part)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def check_repeat_prompt(response: str, *, prompt_to_repeat: str, **_) -> bool:
    """Check response starts with the prompt."""
    if not prompt_to_repeat:
        raise ValueError("prompt_to_repeat must be provided")
    return response.strip().lower().startswith(prompt_to_repeat.strip().lower())


def check_end_phrase(response: str, *, end_phrase: str, **_) -> bool:
    """Check response ends with exact phrase."""
    return response.strip().strip('"').lower().endswith(end_phrase.strip().lower())


def check_capital_word_frequency(
    response: str, *, capital_frequency: int, capital_relation: str, **_
) -> bool:
    """Check frequency of ALL CAPS words."""
    words = nltk.word_tokenize(response)
    count = sum(1 for w in words if w.isupper())
    if capital_relation == "less than":
        return count < capital_frequency
    return count >= capital_frequency


def check_english_capital(response: str, **_) -> bool:
    """Check response is English and all caps."""
    try:
        return response.isupper() and langdetect.detect(response) == "en"
    except langdetect.LangDetectException:
        return True


def check_english_lowercase(response: str, **_) -> bool:
    """Check response is English and all lowercase."""
    try:
        return response.islower() and langdetect.detect(response) == "en"
    except langdetect.LangDetectException:
        return True


def check_no_comma(response: str, **_) -> bool:
    """Check response has no commas."""
    return "," not in response


def check_quotation(response: str, **_) -> bool:
    """Check response is wrapped in double quotes."""
    response = response.strip()
    return len(response) > 1 and response[0] == '"' and response[-1] == '"'


instruction_checkers = {
    "keywords:existence": check_keyword_existence,
    "keywords:frequency": check_keyword_frequency,
    "keywords:forbidden_words": check_forbidden_words,
    "keywords:letter_frequency": check_letter_frequency,
    "length_constraints:number_sentences": check_number_sentences,
    "length_constraints:number_paragraphs": check_number_paragraphs,
    "length_constraints:number_words": check_number_words,
    "length_constraints:nth_paragraph_first_word": check_nth_paragraph_first_word,
    "detectable_content:number_placeholders": check_number_placeholders,
    "detectable_content:postscript": check_postscript,
    "detectable_format:number_bullet_lists": check_number_bullet_lists,
    "detectable_format:constrained_response": check_constrained_response,
    "detectable_format:number_highlighted_sections": check_number_highlighted_sections,
    "detectable_format:multiple_sections": check_multiple_sections,
    "detectable_format:json_format": check_json_format,
    "detectable_format:title": check_title,
    "combination:two_responses": check_two_responses,
    "combination:repeat_prompt": check_repeat_prompt,
    "startend:end_checker": check_end_phrase,
    "change_case:capital_word_frequency": check_capital_word_frequency,
    "change_case:english_capital": check_english_capital,
    "change_case:english_lowercase": check_english_lowercase,
    "punctuation:no_comma": check_no_comma,
    "startend:quotation": check_quotation,
}

skipped_instructions = {"language:response_language"}


def check_instruction_following(
    instruction_id_list: list[str], kwargs_list: list[dict], response: str
) -> list[bool]:
    """Check if response follows each instruction."""
    results = []
    for instruction_id, kwargs in zip(instruction_id_list, kwargs_list):
        if instruction_id in skipped_instructions:
            logger.warning(f"Skipping unsupported instruction: {instruction_id}")
            continue

        checker = instruction_checkers[instruction_id]
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        is_following = bool(response.strip() and checker(response, **kwargs))
        results.append(is_following)

    return results


class IFEvalInstructionAccuracy(Metric):
    """Metric for instruction-level accuracy using IFEval methodology."""

    def __init__(
        self,
        name: str = "inst_level_strict_acc",
        pretty_name: str = "Instruction-Level Strict Accuracy",
        postprocessing_fn: t.Callable[[float], tuple[float, str]] | None = None,
    ) -> None:
        """Initialize the metric."""
        super().__init__(
            name=name, pretty_name=pretty_name, postprocessing_fn=postprocessing_fn
        )

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Calculate instruction-level accuracy."""
        all_results: list[bool] = []
        for pred, ref in zip(predictions, references):
            results = check_instruction_following(
                instruction_id_list=ref["instruction_id_list"],
                kwargs_list=ref["kwargs"],
                response=str(pred),
            )
            all_results.extend(results)
        return sum(all_results) / len(all_results) if all_results else 0.0


inst_level_strict_acc_metric = IFEvalInstructionAccuracy()
