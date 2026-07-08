# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
# ]
# ///

"""Create the MultiWikiQA-lb reading comprehension dataset.

This dataset already exists at alexandrainst/multi-wiki-qa (subset "lb").
This script documents the dataset structure for EuroEval.
"""

from datasets import load_dataset


def main() -> None:
    """Load and verify the MultiWikiQA-lb dataset exists."""
    # The dataset already exists on Hugging Face
    repo_id = "alexandrainst/multi-wiki-qa"
    subset = "lb"

    # Load the dataset to verify it's accessible
    dataset = load_dataset(repo_id, subset, split="train")

    print(f"✓ MultiWikiQA-lb loaded successfully")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Columns: {dataset.column_names}")
    print(f"  Sample: {dataset[0]}")
    print(f"\nFor EuroEval, use: scandeval -m <model_id> -d multi-wiki-qa-lb")


if __name__ == "__main__":
    main()
