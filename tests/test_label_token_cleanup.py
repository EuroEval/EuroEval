"""Regression tests for Unicode-aware label-token cleanup."""

import collections.abc as c
import typing as t
from dataclasses import dataclass

import pytest

from euroeval import tokenisation_utils
from euroeval.data_models import DatasetConfig
from euroeval.languages import DANISH
from euroeval.task_group_utils.sequence_classification import (
    get_closest_logprobs_labels,
)
from euroeval.tasks import SENT


@pytest.mark.parametrize(
    ("generated_label", "candidate_label", "mapping"),
    [
        ("!Éclair?", "éclair", {"éclair": "éclair"}),
        ("ĠÉclair", "éclair", {"éclair": "éclair"}),
        (" ĠÉclair?", "éclair", {"éclair": " éclair"}),
        ("▁Éclair", "éclair", {"éclair": "éclair"}),
        ("##Éclair", "éclair", {"éclair": "éclair"}),
        ("!É-co-op?", "é-co-op", {"é-co-op": "é-co-op"}),
        ("!123É?", "123é", {"123é": "123é"}),
        ("!ÉCLAIR?", "éclair", True),
    ],
    ids=[
        "punctuation-edges-with-mapping",
        "gpt-prefix-with-mapping",
        "space-before-gpt-prefix-with-mapping",
        "sentencepiece-prefix-with-mapping",
        "wordpiece-prefix-with-mapping",
        "internal-punctuation-with-mapping",
        "digits-and-accented-letter-with-mapping",
        "lowercasing-without-mapping",
    ],
)
def test_get_closest_logprobs_labels_cleans_unicode_edges(
    generated_label: str,
    candidate_label: str,
    mapping: dict[str, str] | t.Literal[True],
) -> None:
    """Match labels after cleaning Unicode and tokenizer marker edges."""
    result = get_closest_logprobs_labels(
        generation_logprobs=[[[("!!!", -0.1), (generated_label, -0.2)]]],
        first_label_token_mapping=mapping,
        candidate_labels=[[candidate_label]],
    )

    assert result == [candidate_label]


def test_get_closest_logprobs_labels_skips_whitespace_only_alternatives() -> None:
    """Skip cleaned junk before matching a Unicode label among alternatives."""
    result = get_closest_logprobs_labels(
        generation_logprobs=[[[(" !!!", -0.1), ("!Éclair?", -0.2)]]],
        first_label_token_mapping=True,
        candidate_labels=[["éclair", "autre"]],
    )

    assert result == ["éclair"]


@pytest.mark.parametrize(
    ("tokens", "label", "expected"),
    [
        (["!!!", "!Éclair?"], "éclair", "éclair"),
        (["ĠÉclair"], "éclair", "éclair"),
        ([" ĠÉclair?"], "éclair", " éclair"),
        (["▁Éclair"], "éclair", "éclair"),
        (["##Éclair"], "éclair", "éclair"),
        (["é-co-op"], "é-co-op", "é-co-op"),
        (["!123é?"], "123é", "123é"),
    ],
    ids=[
        "punctuation-only-alternative",
        "gpt-prefix",
        "space-before-gpt-prefix",
        "sentencepiece-prefix",
        "wordpiece-prefix",
        "internal-punctuation",
        "digits-and-accented-letter",
    ],
)
def test_get_first_label_token_mapping_cleans_unicode_edges(
    monkeypatch: pytest.MonkeyPatch,
    model_config: object,
    tokens: c.Sequence[str],
    label: str,
    expected: str,
) -> None:
    """Clean token edges without damaging Unicode label content."""
    dataset_config = DatasetConfig(
        name=f"unicode-label-{label}-{tokens[0]}",
        task=SENT,
        languages=[DANISH],
        labels=[label],
        prompt_label_mapping={label: label},
    )
    monkeypatch.setattr(
        tokenisation_utils, "has_chat_template", lambda tokeniser: False
    )
    monkeypatch.setattr(
        tokenisation_utils,
        "should_prefix_space_be_added_to_labels",
        lambda labels_to_be_generated, tokeniser: False,
    )

    result = tokenisation_utils.get_first_label_token_mapping(
        dataset_config=dataset_config,
        model_config=model_config,
        tokeniser=FakeTokeniser(tokens),
        generative_type=None,
        log_metadata=False,
    )

    assert result == {label: expected}


@dataclass
class FakeTokeniser:
    """Tokeniser returning configured decoded tokens."""

    tokens: c.Sequence[str]

    def decode(self, token_id: int) -> str:
        """Decode a configured token ID.

        Returns:
            The configured token.
        """
        return self.tokens[token_id]

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        """Return one ID for each configured token."""
        return list(range(len(self.tokens)))
