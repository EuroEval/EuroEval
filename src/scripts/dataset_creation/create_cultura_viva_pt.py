# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
# ]
# ///

"""Create the CulturaVivaPT dataset and upload it to HF Hub."""

import random

from constants import CHOICES_MAPPING
from datasets import Dataset, DatasetDict, load_dataset

SOURCE_ID = "amalia-llm/cultura-viva-pt-mcq"
TARGET_ID = "EuroEval/euroeval-amalia-cultura-viva-pt"
TRAIN_SIZE = 32
VAL_SIZE = 128


def main() -> None:
    """Create and upload the CulturaVivaPT dataset."""
    dataset = load_dataset(path=SOURCE_ID, split="train")
    assert isinstance(dataset, Dataset)

    rows: list[dict] = []
    for row in dataset:
        labels = list(row["choices"]["label"])
        choices = list(row["choices"]["text"])
        correct_index = labels.index(row["answerKey"])
        permutation = list(range(len(choices)))
        random.Random(x=row["id"]).shuffle(x=permutation)
        text = f"{row['question'].strip()}\n{CHOICES_MAPPING['pt']}:\n" + "\n".join(
            f"{'abcd'[label_index]}. {choices[choice_index].strip()}"
            for label_index, choice_index in enumerate(permutation)
        )
        rows.append(
            {
                "text": text,
                "label": "abcd"[permutation.index(correct_index)],
                "subject": row["domain"],
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
    print(f"Uploaded {len(rows):,} CulturaVivaPT samples to {TARGET_ID}.")


if __name__ == "__main__":
    main()
