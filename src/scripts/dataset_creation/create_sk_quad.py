# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets>=3.5.0",
#     "huggingface-hub>=0.24.0",
# ]
# ///

"""Create the answerable-only SK-QuAD Slovak QA mini dataset."""

from datasets import Dataset, DatasetDict, load_dataset

try:
    from dataset_utils import make_dataset_dict, upload_private
except ModuleNotFoundError:
    from src.scripts.dataset_creation.dataset_utils import (
        make_dataset_dict,
        upload_private,
    )

SOURCE_DATASET = "slovak-nlp/sklep"
TARGET_DATASET = "EuroEval/sk-quad-mini"


def main() -> None:
    """Download, process, and privately upload answerable SK-QuAD."""
    source = load_dataset(path=SOURCE_DATASET, name="question-answering")
    assert isinstance(source, DatasetDict)
    dataset = process_dataset(raw_dataset=source)
    upload_private(dataset=dataset, repository_id=TARGET_DATASET)


def process_dataset(raw_dataset: DatasetDict) -> DatasetDict:
    """Filter invalid answers and convert SK-QuAD to EuroEval's QA schema.

    Returns:
        The processed dataset with independently capped splits.
    """
    return make_dataset_dict(
        train=_process_split(raw_dataset["train"]),
        validation=_process_split(raw_dataset["validation"]),
        test=_process_split(raw_dataset["test"]),
    )


def _process_split(source: Dataset) -> Dataset:
    """Filter and convert one SK-QuAD source split.

    Returns:
        The processed split.
    """
    rows = []
    for index, row in enumerate(source):
        answers = row["answers"]
        if not _answers_are_valid(
            context=row["context"],
            answer_texts=answers["text"],
            answer_starts=answers["answer_start"],
        ):
            continue
        rows.append(
            {
                "id": row.get("id", str(index)),
                "context": row["context"],
                "question": row["question"],
                "answers": {
                    "text": list(answers["text"]),
                    "answer_start": list(answers["answer_start"]),
                },
            }
        )
    return Dataset.from_list(rows)


def _answers_are_valid(
    context: str, answer_texts: list[str], answer_starts: list[int]
) -> bool:
    """Return whether every answer is non-empty and has a correct offset."""
    if not answer_texts or len(answer_texts) != len(answer_starts):
        return False
    return all(
        isinstance(text, str)
        and bool(text)
        and isinstance(start, int)
        and 0 <= start <= len(context) - len(text)
        and context[start : start + len(text)] == text
        for text, start in zip(answer_texts, answer_starts, strict=True)
    )


if __name__ == "__main__":
    main()
