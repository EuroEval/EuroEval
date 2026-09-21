"""Tests for the SKLEP-derived dataset creation scripts."""

from datasets import Dataset, DatasetDict

from euroeval.dataset_configs.slovak import SKLEP_RTE_CONFIG
from euroeval.tasks import NLI
from src.scripts.dataset_creation.create_reviews3 import (
    process_dataset as process_reviews3,
)
from src.scripts.dataset_creation.create_sk_quad import (
    process_dataset as process_sk_quad,
)
from src.scripts.dataset_creation.create_sklep_nli import process_dataset as process_nli
from src.scripts.dataset_creation.create_sklep_rte import process_dataset as process_rte
from src.scripts.dataset_creation.create_wikigoldsk import (
    process_dataset as process_wikigoldsk,
)


def test_each_split_is_capped_independently() -> None:
    """The cap is applied to each source split, not after combining them."""
    source = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "premise": [f"p{idx}" for idx in range(1025)],
                    "hypothesis": [f"h{idx}" for idx in range(1025)],
                    "label": [0] * 1025,
                }
            ),
            "validation": Dataset.from_dict(
                {"premise": ["p"] * 257, "hypothesis": ["h"] * 257, "label": [0] * 257}
            ),
            "test": Dataset.from_dict(
                {
                    "premise": ["p"] * 2049,
                    "hypothesis": ["h"] * 2049,
                    "label": [0] * 2049,
                }
            ),
        }
    )

    result = process_nli(raw_dataset=source)
    repeated_result = process_nli(raw_dataset=source)

    assert {split: dataset.num_rows for split, dataset in result.items()} == {
        "train": 1024,
        "val": 256,
        "test": 2048,
    }
    assert result["train"].to_list() == repeated_result["train"].to_list()


def test_nli_formats_pairs_and_maps_all_labels() -> None:
    """NLI pairs use EuroEval's standard three-way labels."""
    source = DatasetDict(
        {
            split: Dataset.from_dict(
                {
                    "premise": ["p0", "p1", "p2"],
                    "hypothesis": ["h0", "h1", "h2"],
                    "label": [0, 1, 2],
                }
            )
            for split in ("train", "validation", "test")
        }
    )

    result = process_nli(raw_dataset=source)

    assert result["train"].to_list() == [
        {"text": "Prvé tvrdenie: p0\nDruhé tvrdenie: h0", "label": "entailment"},
        {"text": "Prvé tvrdenie: p1\nDruhé tvrdenie: h1", "label": "neutral"},
        {"text": "Prvé tvrdenie: p2\nDruhé tvrdenie: h2", "label": "contradiction"},
    ]


def test_reviews3_maps_binary_sentiment_labels() -> None:
    """Reviews3 labels are exposed as negative and positive strings."""
    source = DatasetDict(
        {
            split: Dataset.from_dict({"text": ["zlé", "dobré"], "label": [0, 1]})
            for split in ("train", "validation", "test")
        }
    )

    result = process_reviews3(raw_dataset=source)

    assert result["train"]["label"] == ["negative", "positive"]


def test_rte_config_supports_binary_encoder_and_generative_evaluation() -> None:
    """The unofficial RTE config exposes one consistent binary label contract."""
    assert SKLEP_RTE_CONFIG.task == NLI
    assert SKLEP_RTE_CONFIG.labels == ["entailment", "not entailment"]
    assert SKLEP_RTE_CONFIG.prompt_label_mapping == {
        "entailment": "pravda",
        "not entailment": "nepravda",
    }
    assert SKLEP_RTE_CONFIG.unofficial is True
    assert SKLEP_RTE_CONFIG.prompt_prefix is not None
    assert SKLEP_RTE_CONFIG.prompt_template is not None
    assert SKLEP_RTE_CONFIG.instruction_prompt is not None
    assert "neutral" not in SKLEP_RTE_CONFIG.prompt_prefix
    assert "contradiction" not in SKLEP_RTE_CONFIG.instruction_prompt


def test_rte_uses_binary_labels() -> None:
    """RTE remains binary while using the NLI dataset schema."""
    source = DatasetDict(
        {
            split: Dataset.from_dict(
                {"text1": ["p0", "p1"], "text2": ["h0", "h1"], "label": [0, 1]}
            )
            for split in ("train", "validation", "test")
        }
    )

    result = process_rte(raw_dataset=source)

    assert result["train"]["label"] == ["entailment", "not entailment"]


def test_sk_quad_drops_empty_and_invalid_answers() -> None:
    """Only answerable rows with offsets matching their context are retained."""
    source = DatasetDict(
        {
            split: Dataset.from_list(
                [
                    {
                        "context": "Odpoveď je tu.",
                        "question": "Čo je tu?",
                        "answers": {"text": ["Odpoveď"], "answer_start": [0]},
                    },
                    {
                        "context": "Bez odpovede.",
                        "question": "Čo?",
                        "answers": {"text": [], "answer_start": []},
                    },
                    {
                        "context": "Odpoveď je tu.",
                        "question": "Čo?",
                        "answers": {"text": ["odpoveď"], "answer_start": [0]},
                    },
                ]
            )
            for split in ("train", "validation", "test")
        }
    )

    result = process_sk_quad(raw_dataset=source)

    assert result["train"].num_rows == 1
    assert result["train"][0] == {
        "id": "0",
        "context": "Odpoveď je tu.",
        "question": "Čo je tu?",
        "answers": {"text": ["Odpoveď"], "answer_start": [0]},
    }


def test_wikigoldsk_uses_sentence_tokens_and_text_labels() -> None:
    """WikiGoldSK exposes aligned tokens and canonical BIO labels."""
    source = DatasetDict(
        {
            split: Dataset.from_list(
                [
                    {
                        "sentence": "Bratislava je mesto.",
                        "tokens": ["Bratislava", "je", "mesto", "."],
                        "ner_tags_text": ["B-LOC", "O", "O", "O"],
                    }
                ]
            )
            for split in ("train", "validation", "test")
        }
    )

    result = process_wikigoldsk(raw_dataset=source)

    assert result["train"][0] == {
        "text": "Bratislava je mesto.",
        "tokens": ["Bratislava", "je", "mesto", "."],
        "labels": ["B-LOC", "O", "O", "O"],
    }
