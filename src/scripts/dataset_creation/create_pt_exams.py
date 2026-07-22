# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
# ]
# ///

"""Create the PT Exams dataset and upload it to HF Hub."""

import random

from constants import CHOICES_MAPPING
from datasets import Dataset, DatasetDict, load_dataset

SOURCE_ID = "amalia-llm/pt_exams"
TARGET_ID = "EuroEval/euroeval-amalia-pt-exams"
TRAIN_SIZE = 32
VAL_SIZE = 256


def main() -> None:
    """Create and upload the PT Exams dataset."""
    dataset = load_dataset(path=SOURCE_ID, name="default", split="test")
    assert isinstance(dataset, Dataset)

    rows: list[dict] = []
    for index, row in enumerate(dataset):
        choices = list(row["choices"])
        permutation = list(range(len(choices)))
        random.Random(x=f"{row['year']}:{row['subject']}:{index}").shuffle(
            x=permutation
        )
        text = f"{row['question'].strip()}\n{CHOICES_MAPPING['pt']}:\n" + "\n".join(
            f"{'abcd'[label_index]}. {choices[choice_index].strip()}"
            for label_index, choice_index in enumerate(permutation)
        )
        rows.append(
            {
                "text": text,
                "label": "abcd"[permutation.index(row["answer"])],
                "subject": row["subject"],
                "year": row["year"],
            }
        )

    dataset = Dataset.from_list(mapping=rows).shuffle(seed=4242)
    output = DatasetDict(
        {
            "train": dataset.select(indices=range(TRAIN_SIZE)),
            "val": dataset.select(indices=range(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE)),
            "test": dataset.select(indices=range(TRAIN_SIZE + VAL_SIZE, len(dataset))),
        }
    )
    output.push_to_hub(repo_id=TARGET_ID, private=True)
    print(f"Uploaded {len(rows):,} PT Exams samples to {TARGET_ID}.")


if __name__ == "__main__":
    main()
