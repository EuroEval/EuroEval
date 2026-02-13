"""Constraint functions used for the IFEval metric."""

import collections
import collections.abc as c
import json
import logging
import re
import typing as t
from functools import wraps

import langdetect
import nltk

from ...exceptions import InvalidBenchmark

logger = logging.getLogger(__name__)


class Constraint(t.Protocol):
    """An instruction-following constraint."""

    def __call__(self, response: str, **constraint_kwargs) -> bool:
        """Apply the constraint.

        Args:
            response:
                The output to be checked.
            **constraint_kwargs:
                Extra keyword arguments for the constraint function.

        Returns:
            True if the constraint is satisfied, otherwise False.
        """
        ...


ALL_CONSTRAINTS: dict[str, Constraint] = dict()


def register(
    name: str, **desired_keys_and_types
) -> c.Callable[[Constraint], Constraint]:
    """Decorator that registers a function under the given name.

    Args:
        name:
            The name under which to register the function.
        **desired_keys_and_types:
            The keyword arguments and their types that the function expects.

    Returns:
        The decorator function.
    """

    def decorator(fn: Constraint) -> Constraint:
        """Register the function under the given name.

        Args:
            fn:
                The function to register.

        Returns:
            The function.
        """

        @wraps(fn)
        def wrapper(response: str, **constraint_kwargs) -> bool:
            """Wrapper function that checks the keyword arguments and their types.

            Args:
                response:
                    The response string to be checked.
                **constraint_kwargs:
                    Extra keyword arguments for the constraint function.

            Returns:
                The result of the function.

            Raises:
                InvalidBenchmark:
                    If a required keyword argument is missing or if a keyword argument
                    has the wrong type.
            """
            for key, type_ in desired_keys_and_types.items():
                if key not in constraint_kwargs:
                    raise InvalidBenchmark(
                        f"The function {fn.__name__!r} (registered as {name!r}) "
                        f"requires the keyword argument {key!r}."
                    )

                # Special case for Literal, since it does not support `isinstance`
                elif t.get_origin(type_) == t.Literal:
                    possible_values = t.get_args(type_)
                    if constraint_kwargs[key] not in possible_values:
                        raise InvalidBenchmark(
                            f"The function {fn.__name__!r} (registered as {name!r}) "
                            f"expects the keyword argument {key!r} to be one of "
                            f"{possible_values}, but got {constraint_kwargs[key]!r}."
                        )

                elif not isinstance(constraint_kwargs[key], type_):
                    raise InvalidBenchmark(
                        f"The function {fn.__name__!r} (registered as {name!r}) "
                        f"expects the keyword argument {key!r} to be of type "
                        f"{type_.__name__!r}, but got "
                        f"{type(constraint_kwargs[key]).__name__!r}."
                    )
            return fn(response, **constraint_kwargs)

        ALL_CONSTRAINTS[name] = wrapper  # pyrefly: ignore[unsupported-operation]
        return fn

    return decorator


@register("keywords:existence", keywords=list)
def check_keyword_existence(response: str, **constraint_kwargs) -> bool:
    """Check that all keywords exist in the response.

    Args:
        response:
            The response string to be checked.
        **constraint_kwargs:
            Keyword arguments containing ``keywords`` – a list of keyword patterns
            (case‑insensitive) to search for.

    Returns:
        True if all keywords are found in the response, False otherwise.
    """
    keywords: list[str] = constraint_kwargs["keywords"]

    for keyword in keywords:
        if not re.search(pattern=keyword, string=response, flags=re.IGNORECASE):
            return False
    return True


@register(
    "keywords:frequency",
    keyword=str,
    frequency=int,
    relation=t.Literal["less than", "at least"],
)
def check_keyword_frequency(response: str, **constraint_kwargs) -> bool:
    """Check keyword appears with required frequency.

    Args:
        response:
            The response string to check.
        **constraint_kwargs:
            Keyword arguments containing ``keyword``, ``frequency`` and
            ``relation`` – the keyword pattern (case‑insensitive), the required
            frequency, and the comparison operator ("less than" or "at least").

    Returns:
        True if the keyword appears with the required frequency, False otherwise.
    """
    keyword: str = constraint_kwargs["keyword"]
    frequency: int = constraint_kwargs["frequency"]
    relation: t.Literal["less than", "at least"] = constraint_kwargs["relation"]

    all_keyword_matches = re.findall(
        pattern=keyword, string=response, flags=re.IGNORECASE
    )
    if relation == "less than":
        return len(all_keyword_matches) < frequency
    return len(all_keyword_matches) >= frequency


@register("keywords:forbidden_words", forbidden_words=list)
def check_forbidden_words(response: str, **constraint_kwargs) -> bool:
    """Check that forbidden words don't appear.

    Args:
        response:
            The response string to check.
        **constraint_kwargs:
            Keyword arguments containing ``forbidden_words`` – a list of
            words that must not appear (case‑insensitive, whole‑word match).

    Returns:
        True if none of the forbidden words are found, False otherwise.
    """
    forbidden_words: list[str] = constraint_kwargs["forbidden_words"]

    for word in forbidden_words:
        if re.search(
            pattern=r"\b" + word + r"\b", string=response, flags=re.IGNORECASE
        ):
            return False
    return True


@register(
    "keywords:letter_frequency",
    letter=str,
    let_frequency=int,
    let_relation=t.Literal["less than", "at least"],
)
def check_letter_frequency(response: str, **constraint_kwargs) -> bool:
    """Check letter appears with required frequency.

    Args:
        response:
            The response string to check.
        **constraint_kwargs:
            Keyword arguments containing ``letter``, ``let_frequency`` and
            ``let_relation`` – the single character to count (case‑insensitive),
            the frequency threshold, and the comparison operator ("less than"
            or "at least").

    Returns:
        True if the letter frequency satisfies the relation, False otherwise.

    Raises:
        InvalidBenchmark:
            If letter is not a single character.
    """
    letter: str = constraint_kwargs["letter"]
    let_frequency: int = constraint_kwargs["let_frequency"]
    let_relation: t.Literal["less than", "at least"] = constraint_kwargs["let_relation"]
    if len(letter) != 1:
        raise InvalidBenchmark("letter must be a single character")

    counts = collections.Counter(response.lower())
    if let_relation == "less than":
        return counts[letter.lower()] < let_frequency
    return counts[letter.lower()] >= let_frequency


@register(
    "length_constraints:number_sentences",
    num_sentences=int,
    relation=t.Literal["less than", "at least"],
)
def check_number_sentences(response: str, **constraint_kwargs) -> bool:
    """Check number of sentences.

    Args:
        response:
            The response string to check.
        **constraint_kwargs:
            Keyword arguments containing ``num_sentences`` and ``relation``.

    Returns:
        True if the sentence count satisfies the relation, False otherwise.
    """
    num_sentences: int = constraint_kwargs["num_sentences"]
    relation: t.Literal["less than", "at least"] = constraint_kwargs["relation"]

    actual = len(nltk.tokenize.sent_tokenize(text=response))
    if relation == "less than":
        return actual < num_sentences
    return actual >= num_sentences


@register("length_constraints:number_paragraphs", num_paragraphs=int)
def check_number_paragraphs(response: str, **constraint_kwargs) -> bool:
    """Check number of paragraphs (separated by ***).

    Args:
        response:
            The response string to check.
        **constraint_kwargs:
            Keyword arguments containing ``num_paragraphs`` – the exact number
            of paragraphs expected.

    Returns:
        True if the response contains exactly num_paragraphs non‑empty paragraphs,
        False otherwise.
    """
    num_paragraphs: int = constraint_kwargs["num_paragraphs"]

    paragraphs = re.split(pattern=r"\s?\*\*\*\s?", string=response)
    count = len(paragraphs)
    for i, p in enumerate(paragraphs):
        if not p.strip():
            if i == 0 or i == len(paragraphs) - 1:
                count -= 1
            else:
                return False
    return count == num_paragraphs


@register(
    "length_constraints:number_words",
    num_words=int,
    relation=t.Literal["less than", "at least"],
)
def check_number_words(response: str, **constraint_kwargs) -> bool:
    """Check number of words.

    Args:
        response:
            The response string to check.
        **constraint_kwargs:
            Keyword arguments containing ``num_words`` and ``relation``.

    Returns:
        True if the word count satisfies the relation, False otherwise.
    """
    num_words: int = constraint_kwargs["num_words"]
    relation: t.Literal["less than", "at least"] = constraint_kwargs["relation"]

    words = nltk.tokenize.word_tokenize(text=response)
    actual = len(words)
    if relation == "less than":
        return actual < num_words
    return actual >= num_words


@register(
    "length_constraints:nth_paragraph_first_word",
    num_paragraphs=int,
    nth_paragraph=int,
    first_word=str,
)
def check_nth_paragraph_first_word(response: str, **constraint_kwargs) -> bool:
    """Check paragraph count and first word of nth paragraph.

    Args:
        response:
            The response string to check.
        **constraint_kwargs:
            Keyword arguments containing ``num_paragraphs``, ``nth_paragraph``,
            and ``first_word`` – the expected first word of the nth paragraph
            (case‑insensitive).

    Returns:
        True if the response has exactly num_paragraphs paragraphs and the nth
        paragraph starts with first_word, False otherwise.

    Raises:
        InvalidBenchmark:
            If the n'th paragraph is greater than the number of paragraphs in the
            response.
    """
    num_paragraphs: int = constraint_kwargs["num_paragraphs"]
    nth_paragraph: int = constraint_kwargs["nth_paragraph"]
    first_word: str = constraint_kwargs["first_word"]
    if nth_paragraph > num_paragraphs:
        raise InvalidBenchmark(
            "The n'th paragraph is greater than the number of paragraphs in the "
            "`check_nth_paragraph_first_word` constraint. This should not happen."
        )

    paragraphs = re.split(pattern=r"\n\n", string=response)
    count = sum(1 for p in paragraphs if p.strip())
    if nth_paragraph > count:
        return False

    paragraph = paragraphs[nth_paragraph - 1].strip()
    if not paragraph:
        return False

    word = paragraph.split()[0].strip().lstrip("'\"")
    actual_first = ""
    for char in word:
        if char in ".,?!'\"":
            break
        actual_first += char.lower()

    return count == num_paragraphs and actual_first == first_word.lower()


@register("detectable_content:number_placeholders", num_placeholders=int)
def check_number_placeholders(response: str, **constraint_kwargs) -> bool:
    """Check minimum number of [placeholder] brackets.

    Args:
        response:
            The response string to check.
        **constraint_kwargs:
            Keyword arguments containing ``num_placeholders`` – the minimum
            number of placeholder brackets expected.

    Returns:
        True if the response contains at least num_placeholders placeholders
        of the form [placeholder], False otherwise.
    """
    num_placeholders: int = constraint_kwargs["num_placeholders"]

    placeholders = re.findall(pattern=r"\[.*?\]", string=response)
    return len(placeholders) >= num_placeholders


@register("detectable_content:postscript", postscript_marker=str)
def check_postscript(response: str, **constraint_kwargs) -> bool:
    """Check for postscript marker.

    Args:
        response:
            The response string to check.
        **constraint_kwargs:
            Keyword arguments containing ``postscript_marker`` – the postscript
            label to look for (e.g. "P.S.", "P.P.S").

    Returns:
        True if the postscript marker is found in the response, False otherwise.
    """
    postscript_marker: str = constraint_kwargs["postscript_marker"]

    response = response.lower()
    if postscript_marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif postscript_marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + postscript_marker.lower() + r".*$"
    return bool(re.findall(pattern=pattern, string=response, flags=re.MULTILINE))


@register("detectable_format:number_bullet_lists", num_bullets=int)
def check_number_bullet_lists(response: str, **constraint_kwargs) -> bool:
    """Check exact number of bullet points.

    Args:
        response:
            The response string to check.
        **constraint_kwargs:
            Keyword arguments containing ``num_bullets`` – the exact number
            of bullet points expected.

    Returns:
        True if the response contains exactly num_bullets bullet points, where bullet
        points are lines starting with ``*`` or ``-``, False otherwise.
    """
    num_bullets: int = constraint_kwargs["num_bullets"]

    bullets1 = re.findall(
        pattern=r"^\s*\*[^\*].*$", string=response, flags=re.MULTILINE
    )
    bullets2 = re.findall(pattern=r"^\s*-.*$", string=response, flags=re.MULTILINE)
    return len(bullets1) + len(bullets2) == num_bullets


@register("detectable_format:constrained_response")
def check_constrained_response(response: str, **_) -> bool:
    """Check response contains one of the constrained options.

    Args:
        response:
            The response string to check.

    Returns:
        True if the response contains exactly one of "My answer is yes.", "My answer is
        no.", or "My answer is maybe.", False otherwise.
    """
    options = ("My answer is yes.", "My answer is no.", "My answer is maybe.")
    return any(opt in response.strip() for opt in options)


@register("detectable_format:number_highlighted_sections", num_highlights=int)
def check_number_highlighted_sections(response: str, **constraint_kwargs) -> bool:
    """Check minimum highlighted *sections*.

    Args:
        response:
            The response string to check.
        **constraint_kwargs:
            Keyword arguments containing ``num_highlights`` – the minimum number
            of highlighted sections expected.

    Returns:
        True if the response contains at least num_highlights non‑empty highlighted
        sections, where highlights are text wrapped in single (*text*) or double
        (**text**) asterisks, False otherwise.
    """
    num_highlights: int = constraint_kwargs["num_highlights"]

    count = 0
    for h in re.findall(pattern=r"\*[^\n\*]*\*", string=response):
        if h.strip("*").strip():
            count += 1
    for h in re.findall(pattern=r"\*\*[^\n\*]*\*\*", string=response):
        if h.removeprefix("**").removesuffix("**").strip():
            count += 1
    return count >= num_highlights


@register("detectable_format:multiple_sections", section_spliter=str, num_sections=int)
def check_multiple_sections(response: str, **constraint_kwargs) -> bool:
    """Check for Section X markers.

    Args:
        response:
            The response string to check.
        **constraint_kwargs:
            Keyword arguments containing ``section_spliter`` and ``num_sections``.

    Returns:
        True if the response contains at least num_sections sections delimited
        by markers of the form "<section_spliter> <number>", False otherwise.
    """
    section_spliter: str = constraint_kwargs["section_spliter"]
    num_sections: int = constraint_kwargs["num_sections"]

    pattern = r"\s?" + section_spliter + r"\s?\d+\s?"
    sections = re.split(pattern=pattern, string=response)
    return len(sections) - 1 >= num_sections


@register("detectable_format:json_format")
def check_json_format(response: str, **_) -> bool:
    """Check response is valid JSON.

    Args:
        response:
            The response string to check.

    Returns:
        True if the response (after stripping optional ```json``` fences) is valid JSON,
        False otherwise.
    """
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
    except json.JSONDecodeError:
        return False


@register("detectable_format:title")
def check_title(response: str, **_) -> bool:
    """Check for <<title>> format.

    Args:
        response:
            The response string to check.

    Returns:
        True if the response contains at least one non‑empty title wrapped in
        double angle brackets (e.g. <<My Title>>), False otherwise.
    """
    for title in re.findall(pattern=r"<<[^\n]+>>", string=response):
        if title.lstrip("<").rstrip(">").strip():
            return True
    return False


@register("combination:two_responses")
def check_two_responses(response: str, **_) -> bool:
    """Check for two different responses separated by ******.

    Args:
        response:
            The response string to check.

    Returns:
        True if the response contains exactly two non‑empty, non‑identical sections
        separated by "******", False otherwise.
    """
    parts = response.split("******")
    valid = []
    for i, part in enumerate(parts):
        if not part.strip():
            if i != 0 and i != len(parts) - 1:
                return False
        else:
            valid.append(part)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


@register("combination:repeat_prompt", prompt_to_repeat=str)
def check_repeat_prompt(response: str, **constraint_kwargs) -> bool:
    """Check response starts with the prompt.

    Args:
        response:
            The response string to check.
        **constraint_kwargs:
            Keyword arguments containing ``prompt_to_repeat`` – the exact prompt text
            the response must begin with (case‑insensitive).

    Returns:
        True if the response starts with prompt_to_repeat, False otherwise.
    """
    prompt_to_repeat: str = constraint_kwargs["prompt_to_repeat"]
    return response.strip().lower().startswith(prompt_to_repeat.strip().lower())


@register("startend:end_checker", end_phrase=str)
def check_end_phrase(response: str, **constraint_kwargs) -> bool:
    """Check response ends with exact phrase.

    Args:
        response:
            The response string to check.
        **constraint_kwargs:
            Keyword arguments containing ``end_phrase`` – the exact phrase the
            response must end with (case‑insensitive).

    Returns:
        True if the response ends with end_phrase, False otherwise.
    """
    end_phrase: str = constraint_kwargs["end_phrase"]
    return response.strip().strip('"').lower().endswith(end_phrase.strip().lower())


@register(
    "change_case:capital_word_frequency",
    capital_frequency=int,
    capital_relation=t.Literal["less than", "at least"],
)
def check_capital_word_frequency(response: str, **constraint_kwargs) -> bool:
    """Check frequency of ALL CAPS words.

    Args:
        response:
            The response string to check.
        **constraint_kwargs:
            Keyword arguments containing ``capital_frequency`` and
            ``capital_relation``.

    Returns:
        True if the count of fully uppercased words satisfies the relation,
        False otherwise.
    """
    capital_frequency: int = constraint_kwargs["capital_frequency"]
    capital_relation: t.Literal["less than", "at least"] = constraint_kwargs[
        "capital_relation"
    ]

    words = nltk.word_tokenize(response)
    count = sum(1 for w in words if w.isupper())
    if capital_relation == "less than":
        return count < capital_frequency
    return count >= capital_frequency


@register("change_case:english_capital")
def check_english_capital(response: str, **_) -> bool:
    """Check response is English and all caps.

    Args:
        response:
            The response string to check.

    Returns:
        True if the response is entirely uppercase and detected as English,
        False otherwise. Returns True if language detection fails.
    """
    try:
        return response.isupper() and langdetect.detect(response) == "en"
    except langdetect.LangDetectException:
        return True


@register("change_case:english_lowercase")
def check_english_lowercase(response: str, **_) -> bool:
    """Check response is English and all lowercase.

    Args:
        response:
            The response string to check.

    Returns:
        True if the response is entirely lowercase and detected as English,
        False otherwise. Returns True if language detection fails.
    """
    try:
        return response.islower() and langdetect.detect(response) == "en"
    except langdetect.LangDetectException:
        return True


@register("punctuation:no_comma")
def check_no_comma(response: str, **_) -> bool:
    """Check response has no commas.

    Args:
        response:
            The response string to check.

    Returns:
        True if the response contains no comma characters, False otherwise.
    """
    return "," not in response


@register("startend:quotation")
def check_quotation(response: str, **_) -> bool:
    """Check response is wrapped in double quotes.

    Args:
        response:
            The response string to check.

    Returns:
        True if the response (after stripping whitespace) begins and ends with a double
        quote character, False otherwise.
    """
    response = response.strip()
    return len(response) > 1 and response[0] == '"' and response[-1] == '"'
