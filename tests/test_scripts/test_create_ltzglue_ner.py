"""Tests for the ltzGLUE-NER dataset creation script."""

from src.scripts.dataset_creation.create_ltzglue_ner import _load_split


def test_date_spans_are_relabelled_without_dropping_data() -> None:
    """DATE spans become outside labels while examples and tokens remain intact."""
    data = [
        {
            "tokens": ["Am", "1.", "Mee", "zu", "Lëtzebuerg"],
            "ner_tags": ["O", "B-DATE", "I-DATE", "O", "B-LOC"],
        },
        {"tokens": ["Den", "2.", "Juni"], "ner_tags": ["O", "B-DATE", "I-DATE"]},
    ]

    result = _load_split(data)

    assert len(result) == len(data)
    assert result["tokens"].tolist() == [item["tokens"] for item in data]
    assert result["labels"].tolist() == [["O", "O", "O", "O", "B-LOC"], ["O", "O", "O"]]


def test_non_date_labels_and_tokens_are_preserved() -> None:
    """Non-DATE labels and their token alignment remain unchanged."""
    data = [
        {
            "tokens": ["D'Gemeng", "huet", "d'Gebai"],
            "ner_tags": ["B-ORG", "O", "B-PROD"],
        }
    ]

    result = _load_split(data)

    assert result.iloc[0]["tokens"] == data[0]["tokens"]
    assert result.iloc[0]["labels"] == data[0]["ner_tags"]
    assert len(result.iloc[0]["tokens"]) == len(result.iloc[0]["labels"])
