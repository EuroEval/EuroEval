"""Tests for IFEval instruction-following constraints."""

import nltk
import pytest

from euroeval.metrics.ifeval.constraints import ALL_CONSTRAINTS


@pytest.fixture(scope="module", autouse=True)
def _punkt() -> None:
    """Ensure the NLTK Punkt models are available (incl. Norwegian)."""
    nltk.download("punkt_tab", quiet=True)


class TestNorwegianNumberSentences:
    """Tests for the `no:length_constraints:number_sentences` variant.

    The default English sentence tokenizer over-counts Norwegian sentences that
    contain common abbreviations (e.g. ``f.eks.``, ``bl.a.``), treating them as
    sentence boundaries. The Norwegian variant uses the Norwegian Punkt model.
    """

    constraint = "no:length_constraints:number_sentences"
    # Two real sentences, each containing a Norwegian abbreviation that the
    # English tokenizer splits on (counting three sentences instead of two).
    two_sentences = "Vi bruker mange forkortelser, f.eks. denne. Det er to setninger."

    def test_is_registered(self) -> None:
        """The Norwegian variant is registered."""
        assert self.constraint in ALL_CONSTRAINTS

    def test_abbreviation_not_overcounted(self) -> None:
        """A two-sentence answer satisfies 'less than 3' (English tokenizer fails).

        The English tokenizer counts three sentences here, so the base constraint
        would wrongly reject this answer; the Norwegian variant counts two.
        """
        verifier = ALL_CONSTRAINTS[self.constraint]
        assert verifier(self.two_sentences, num_sentences=3, relation="less than")

    def test_base_constraint_overcounts(self) -> None:
        """Contrast: the base English constraint over-counts the same answer."""
        base = ALL_CONSTRAINTS["length_constraints:number_sentences"]
        assert not base(self.two_sentences, num_sentences=3, relation="less than")

    def test_genuine_violation_still_fails(self) -> None:
        """A one-sentence answer does not satisfy 'at least 3'."""
        verifier = ALL_CONSTRAINTS[self.constraint]
        assert not verifier("Bare én setning.", num_sentences=3, relation="at least")
