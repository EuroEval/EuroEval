# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets>=3.5.0",
#     "huggingface-hub>=0.24.0",
# ]
# ///

"""Create the SKLEP WikiGoldSK Slovak NER mini dataset."""

from datasets import Dataset, DatasetDict, load_dataset

try:
    from dataset_utils import make_dataset_dict, upload_private
except ModuleNotFoundError:
    from src.scripts.dataset_creation.dataset_utils import (
        make_dataset_dict,
        upload_private,
    )

SOURCE_DATASET = "slovak-nlp/sklep"
TARGET_DATASET = "EuroEval/wikigold-sk-mini"
NER_LABELS = (
    "O",
    "B-LOC",
    "I-LOC",
    "B-ORG",
    "I-ORG",
    "B-PER",
    "I-PER",
    "B-MISC",
    "I-MISC",
)


def main() -> None:
    """Download, process, and privately upload WikiGoldSK."""
    source = load_dataset(path=SOURCE_DATASET, name="ner-wikigoldsk")
    assert isinstance(source, DatasetDict)
    dataset = process_dataset(raw_dataset=source)
    upload_private(dataset=dataset, repository_id=TARGET_DATASET)


def process_dataset(raw_dataset: DatasetDict) -> DatasetDict:
    """Convert WikiGoldSK to EuroEval's token-classification schema.

    Returns:
        The processed dataset with independently capped splits.
    """
    return make_dataset_dict(
        train=_process_split(raw_dataset["train"]),
        validation=_process_split(raw_dataset["validation"]),
        test=_process_split(raw_dataset["test"]),
    )


def _process_split(source: Dataset) -> Dataset:
    """Convert one WikiGoldSK source split and validate tag alignment.

    Returns:
        The processed split.

    Raises:
        ValueError:
            If source tokens and labels are not aligned.
    """
    rows = []
    for row in source:
        tokens = list(row["tokens"])
        source_labels = row.get("ner_tags_text")
        if source_labels is None:
            source_labels = row["ner_tags"]
        labels = [_normalise_label(label=label) for label in source_labels]
        if len(tokens) != len(labels):
            raise ValueError("WikiGoldSK tokens and labels must have equal lengths")
        rows.append({"text": row["sentence"], "tokens": tokens, "labels": labels})
    return Dataset.from_list(rows)


def _normalise_label(label: int | str) -> str:
    """Return a canonical WikiGoldSK label.

    Raises:
        ValueError:
            If the source label is not part of the WikiGoldSK schema.
    """
    if isinstance(label, int):
        try:
            return NER_LABELS[label]
        except IndexError as error:
            raise ValueError(f"Unexpected WikiGoldSK label: {label}") from error
    if label not in NER_LABELS:
        raise ValueError(f"Unexpected WikiGoldSK label: {label}")
    return label


if __name__ == "__main__":
    main()
