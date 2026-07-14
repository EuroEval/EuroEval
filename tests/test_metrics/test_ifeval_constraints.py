"""Tests for IFEval instruction-following constraints."""

import nltk
import pytest

from euroeval.metrics.ifeval.constraints import ALL_CONSTRAINTS


@pytest.fixture(scope="module", autouse=True)
def _punkt() -> None:
    """Ensure the NLTK Punkt models are available (incl. Norwegian)."""
    nltk.download("punkt_tab", quiet=True)


class TestNumberSentencesLanguage:
    """Tests for ``check_number_sentences_with_language``.

    The default English sentence tokenizer over-counts sentences in languages
    whose abbreviations it does not know — e.g. Norwegian ``f.eks.`` and
    ``bl.a.`` are treated as sentence boundaries. The ``language`` argument
    selects the right Punkt model.
    """

    constraint = "length_constraints:number_sentences_with_language"
    # Two real Norwegian sentences, each containing an abbreviation the English
    # tokenizer splits on (counting three sentences instead of two).
    two_sentences = "Vi bruker mange forkortelser, f.eks. denne. Det er to setninger."

    def test_norwegian_not_overcounted(self) -> None:
        """With language='norwegian', a two-sentence answer satisfies 'less than 3'."""
        verifier = ALL_CONSTRAINTS[self.constraint]
        assert verifier(
            self.two_sentences,
            num_sentences=3,
            relation="less than",
            language="norwegian",
        )

    def test_english_overcounts(self) -> None:
        """Contrast: with language='english' the same answer is over-counted."""
        verifier = ALL_CONSTRAINTS[self.constraint]
        assert not verifier(
            self.two_sentences,
            num_sentences=3,
            relation="less than",
            language="english",
        )

    def test_genuine_violation_still_fails(self) -> None:
        """A one-sentence answer does not satisfy 'at least 3'."""
        verifier = ALL_CONSTRAINTS[self.constraint]
        assert not verifier(
            "Bare én setning.",
            num_sentences=3,
            relation="at least",
            language="norwegian",
        )

    def test_unknown_language_falls_back(self) -> None:
        """An unavailable Punkt language falls back to the default tokenizer."""
        verifier = ALL_CONSTRAINTS[self.constraint]
        # Should not raise LookupError; counts with the default model instead.
        assert verifier(
            "Én setning her.",
            num_sentences=2,
            relation="less than",
            language="some-nonexistent-language",
        )

    def test_norwegian_relation_literal(self) -> None:
        """The Norwegian 'færre enn' relation maps to a less-than check.

        The multi-ifeval Norwegian splits carry one ``'færre enn'`` relation
        among otherwise English literals, so the constraint must accept it.
        """
        verifier = ALL_CONSTRAINTS[self.constraint]
        assert verifier(
            self.two_sentences,
            num_sentences=3,
            relation="færre enn",
            language="norwegian",
        )


class TestNumberSentencesBackwardsCompatible:
    """The original constraint stays usable without a ``language`` argument."""

    constraint = "length_constraints:number_sentences"

    def test_no_language_argument_required(self) -> None:
        """Older datasets call number_sentences with only num_sentences/relation."""
        verifier = ALL_CONSTRAINTS[self.constraint]
        assert verifier(
            "First sentence. Second sentence.", num_sentences=3, relation="less than"
        )
