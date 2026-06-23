"""Tests for the `task_group_utils.token_classification` module."""

import json

from euroeval.task_group_utils.token_classification import (
    serialise_ner_tags,
    serialised_ner_content_length,
)


class TestSerialiseNerTags:
    """Tests for `serialise_ner_tags`."""

    prompt_label_mapping = {
        "b-per": "person",
        "i-per": "person",
        "b-loc": "place",
        "i-loc": "place",
        "o": "ingen",
    }

    def test_serialises_entities_as_json(self) -> None:
        """Tagged tokens are grouped by localised label into a JSON object string."""
        tokens = ["John", "Doe", "visited", "New", "York"]
        labels = ["b-per", "i-per", "o", "b-loc", "i-loc"]
        result = serialise_ner_tags(
            tokens=tokens, labels=labels, prompt_label_mapping=self.prompt_label_mapping
        )
        assert json.loads(result) == {
            "person": ["John Doe"],
            "place": ["New York"],
            "ingen": [],
        }

    def test_includes_all_labels_even_when_empty(self) -> None:
        """Every prompt label appears, with an empty list when no entity is tagged."""
        result = serialise_ner_tags(
            tokens=["hello", "world"],
            labels=["o", "o"],
            prompt_label_mapping=self.prompt_label_mapping,
        )
        assert json.loads(result) == {"person": [], "place": [], "ingen": []}

    def test_uses_double_quotes_and_is_valid_json(self) -> None:
        """The output is valid JSON (double quotes), not a Python dict repr."""
        result = serialise_ner_tags(
            tokens=["Paris"],
            labels=["b-loc"],
            prompt_label_mapping=self.prompt_label_mapping,
        )
        assert '"place"' in result
        assert json.loads(result)["place"] == ["Paris"]

    def test_unknown_tag_is_ignored(self) -> None:
        """Tags absent from the mapping are skipped rather than raising."""
        result = serialise_ner_tags(
            tokens=["foo", "bar"],
            labels=["b-misc", "o"],
            prompt_label_mapping=self.prompt_label_mapping,
        )
        assert json.loads(result) == {"person": [], "place": [], "ingen": []}


class TestSerialisedNerContentLength:
    """Tests for `serialised_ner_content_length`."""

    prompt_label_mapping = {
        "b-per": "person",
        "i-per": "person",
        "b-loc": "place",
        "o": "ingen",
    }

    def test_counts_only_entity_characters(self) -> None:
        """The length sums the entity strings, excluding the JSON scaffolding."""
        serialised = serialise_ner_tags(
            tokens=["John", "Doe", "visited", "Paris"],
            labels=["b-per", "i-per", "o", "b-loc"],
            prompt_label_mapping=self.prompt_label_mapping,
        )
        # "John Doe" (8) + "Paris" (5) = 13, regardless of the much longer JSON string.
        assert serialised_ner_content_length(serialised) == 13
        assert len(serialised) > 13

    def test_zero_when_no_entities(self) -> None:
        """An answer with no tagged entities has zero content characters."""
        serialised = serialise_ner_tags(
            tokens=["hello", "world"],
            labels=["o", "o"],
            prompt_label_mapping=self.prompt_label_mapping,
        )
        assert serialised_ner_content_length(serialised) == 0
