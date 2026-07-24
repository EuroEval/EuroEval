# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==5.0.0",
# ]
# ///

"""Create the SAUDADE European Portuguese dataset and upload it to HF Hub."""

import random

from constants import CHOICES_MAPPING
from datasets import Dataset, DatasetDict, load_dataset

SOURCE_ID = "amalia-llm/saudade-pt"
TARGET_ID = "EuroEval/euroeval-amalia-saudade-pt-mini"
TRAIN_SIZE = 32
VAL_SIZE = 256
TEST_SIZE = 2048


def main() -> None:
    """Create and upload the SAUDADE-PT dataset."""
    dataset = load_dataset(path=SOURCE_ID, split="test")
    assert isinstance(dataset, Dataset)
    dataset = dataset.shuffle(seed=4242).select(
        indices=range(TRAIN_SIZE + VAL_SIZE + TEST_SIZE)
    )

    rows: list[dict] = []
    for row in dataset:
        choices = list(row["facts_pair"])
        permutation = list(range(len(choices)))
        random.Random(x=f"{row['entity']}:{row['prompt']}").shuffle(x=permutation)
        text = f"{row['prompt'].strip()}\n{CHOICES_MAPPING['pt']}:\n" + "\n".join(
            f"{'ab'[label_index]}. {choices[choice_index].strip()}"
            for label_index, choice_index in enumerate(permutation)
        )
        rows.append(
            {
                "text": text,
                "label": "ab"[permutation.index(row["answer_index"])],
                "entity": row["entity"],
                "category": row["category"],
            }
        )

    dataset = Dataset.from_list(mapping=rows)
    output = DatasetDict(
        {
            "train": dataset.select(indices=range(TRAIN_SIZE)),
            "val": dataset.select(indices=range(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE)),
            "test": dataset.select(
                indices=range(TRAIN_SIZE + VAL_SIZE, TRAIN_SIZE + VAL_SIZE + TEST_SIZE)
            ),
        }
    )
    output.push_to_hub(repo_id=TARGET_ID, private=True)
    print(f"Uploaded {len(rows):,} SAUDADE-PT samples to {TARGET_ID}.")


if __name__ == "__main__":
    main()
