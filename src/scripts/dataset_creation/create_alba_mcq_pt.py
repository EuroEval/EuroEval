# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
# ]
# ///

"""Create the ALBA-MCQ European Portuguese dataset and upload it to HF Hub."""

import random

from constants import CHOICES_MAPPING
from datasets import Dataset, DatasetDict, load_dataset

SOURCE_ID = "amalia-llm/alba_mcq"
TARGET_ID = "EuroEval/euroeval-amalia-alba-mcq-pt"
TRAIN_SIZE = 32
CONFIGS = [
    "culture_bound_semantics",
    "discourse_analysis",
    "language_variety",
    "lexicology",
    "morphology",
    "phonetics_phonology",
    "syntax",
    "word_play",
]


def main() -> None:
    """Create and upload the combined ALBA-MCQ dataset."""
    rows: list[dict] = []
    for config in CONFIGS:
        dataset = load_dataset(path=SOURCE_ID, name=config, split="test")
        assert isinstance(dataset, Dataset)
        for row in dataset:
            choices = list(row["choices"])
            permutation = list(range(len(choices)))
            random.Random(x=f"{config}:{row['id']}").shuffle(x=permutation)
            text = f"{row['question'].strip()}\n{CHOICES_MAPPING['pt']}:\n" + "\n".join(
                f"{'abc'[label_index]}. {choices[choice_index].strip()}"
                for label_index, choice_index in enumerate(permutation)
            )
            rows.append(
                {
                    "text": text,
                    "label": "abc"[permutation.index(row["correct_choice"])],
                    "subject": row["subject"],
                }
            )

    dataset = Dataset.from_list(mapping=rows).shuffle(seed=4242)
    output = DatasetDict(
        {
            "train": dataset.select(indices=range(TRAIN_SIZE)),
            "test": dataset.select(indices=range(TRAIN_SIZE, len(dataset))),
        }
    )
    output.push_to_hub(repo_id=TARGET_ID, private=True)
    print(f"Uploaded {len(rows):,} ALBA-MCQ samples to {TARGET_ID}.")


if __name__ == "__main__":
    main()
