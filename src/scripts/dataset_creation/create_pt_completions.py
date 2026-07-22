# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
# ]
# ///

"""Create the PT-PT Completions dataset and upload it to HF Hub."""

from constants import CHOICES_MAPPING
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

SOURCE_ID = "amalia-llm/pt_text_completion"
TARGET_ID = "EuroEval/euroeval-amalia-pt-completions"
TRAIN_SIZE = 8


def _process_split(dataset: Dataset) -> Dataset:
    """Convert a PT-PT Completions split to EuroEval format.

    Returns:
        Processed dataset split.
    """
    rows: list[dict] = []
    for index, row in enumerate(dataset):
        choices = [row["pt-pt"], row["pt-br"]]
        permutation = [0, 1] if index % 2 == 0 else [1, 0]
        shuffled_choices = [choices[choice_index] for choice_index in permutation]
        text = (
            f"Frase: {row['text'].strip()}\n{CHOICES_MAPPING['pt']}:\n"
            f"a. {shuffled_choices[0].strip()}\n"
            f"b. {shuffled_choices[1].strip()}"
        )
        rows.append({"text": text, "label": "a" if permutation[0] == 0 else "b"})
    return Dataset.from_list(mapping=rows)


def main() -> None:
    """Create and upload the PT-PT Completions dataset."""
    dataset = load_dataset(path=SOURCE_ID, name="default")
    assert isinstance(dataset, DatasetDict)

    processed_test = _process_split(dataset=dataset["test"])
    test = concatenate_datasets(
        dsets=[
            processed_test.select(indices=range(TRAIN_SIZE, len(processed_test))),
            _process_split(dataset=dataset["valid"]),
        ]
    )
    output = DatasetDict(
        {"train": processed_test.select(indices=range(TRAIN_SIZE)), "test": test}
    )
    output.push_to_hub(repo_id=TARGET_ID, private=True)
    print(
        f"Uploaded {sum(len(split) for split in output.values()):,} "
        f"PT-PT Completions samples to {TARGET_ID}."
    )


if __name__ == "__main__":
    main()
