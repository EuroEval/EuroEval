"""Utility functions related to string manipulation or structuring."""

import collections.abc as c
import logging
import re
import typing as t

import demjson3
import numpy as np

from .exceptions import InvalidBenchmark, InvalidModel
from .logging_utils import log

if t.TYPE_CHECKING:
    from .data_models import ModelIdComponents


def scramble(text: str) -> str:
    """Scramble a string in a bijective manner.

    Args:
        text:
            The string to scramble.

    Returns:
        The scrambled string.
    """
    rng = np.random.default_rng(seed=4242)
    permutation = rng.permutation(x=len(text))
    scrambled = "".join(text[i] for i in permutation)
    return scrambled


def unscramble(scrambled_text: str) -> str:
    """Unscramble a string in a bijective manner.

    Args:
        scrambled_text:
            The scrambled string to unscramble.

    Returns:
        The unscrambled string.
    """
    rng = np.random.default_rng(seed=4242)
    permutation = rng.permutation(x=len(scrambled_text))
    inverse_permutation = np.argsort(permutation)
    unscrambled = "".join(scrambled_text[i] for i in inverse_permutation)
    return unscrambled


def extract_json_dict_from_string(s: str) -> dict | None:
    """Extract a JSON dictionary from a string.

    Args:
        s:
            The string to extract the JSON dictionary from.

    Returns:
        The extracted JSON dictionary, or None if no JSON dictionary could be found.
    """
    json_regex = r"\{[^{}]*?\}"
    if (json_match := re.search(pattern=json_regex, string=s, flags=re.DOTALL)) is None:
        log(
            "The model output does not contain any JSON dictionary, so cannot parse "
            f"it. Skipping. Here is the output: {s!r}",
            level=logging.DEBUG,
        )
        return None
    json_string = json_match.group()
    try:
        json_output = demjson3.decode(txt=json_string)
    except demjson3.JSONDecodeError:
        log(
            "The model output is not valid JSON, so cannot parse it. Skipping. "
            f"Here is the output: {json_string!r}",
            level=logging.DEBUG,
        )
        return None
    if not isinstance(json_output, dict):
        log(
            "The model output is not a JSON dictionary, so cannot parse "
            f"it. Skipping. Here is the output: {json_string!r}",
            level=logging.DEBUG,
        )
        return None
    elif not all(isinstance(key, str) for key in json_output.keys()):
        log(
            "The model output is not a JSON dictionary with string keys, "
            "so cannot parse it. Skipping. Here is the output: "
            f"{json_string!r}",
            level=logging.DEBUG,
        )
        return None
    return json_output


def extract_multiple_choice_labels(
    prompt: str, candidate_labels: c.Sequence[str]
) -> c.Sequence[str]:
    """Extract multiple choice labels from a prompt.

    Args:
        prompt:
            The prompt to extract the labels from.
        candidate_labels:
            The candidate labels to look for in the prompt.

    Returns:
        The extracted labels.
    """
    sample_candidate_labels: list[str] = list()
    for candidate_label in candidate_labels:
        candidate_label_match = re.search(
            pattern=rf"\b{candidate_label}\. ", string=prompt, flags=re.IGNORECASE
        )
        if candidate_label_match is not None:
            sample_candidate_labels.append(candidate_label)
    if not sample_candidate_labels:
        raise InvalidBenchmark(
            "Could not extract any candidate labels from the prompt. Please ensure "
            "that the candidate labels are present in the prompt, each followed by a "
            "dot and a space (e.g., 'a. '). The candidate labels are: "
            f"{', '.join(candidate_labels)}. Here is the prompt: {prompt!r}"
        )
    return sample_candidate_labels


def split_model_id(model_id: str) -> "ModelIdComponents":
    """Split a model ID into its components.

    Args:
        model_id:
            The model ID to split.

    Returns:
        The split model ID.

    Raises:
        If the model ID is not valid.
    """
    # Importing here to avoid circular imports
    from .data_models import ModelIdComponents

    # Attempt to extract the model ID, revision, and param using regex
    model_id_match = re.match(pattern=r"^[^@#]+", string=model_id)
    revision_match = re.search(pattern=r"@([^@#]+)", string=model_id)
    param_match = re.search(pattern=r"#([^@#]+)", string=model_id)

    # If we cannot extract the model ID, raise an error
    if model_id_match is None:
        raise InvalidModel(f"The model ID {model_id!r} is not valid.")
    model_id = model_id_match.group()

    # Extract the revision and param and return the result
    revision = revision_match.group(1) if revision_match is not None else "main"
    param = param_match.group(1) if param_match is not None else None
    return ModelIdComponents(model_id=model_id, revision=revision, param=param)
