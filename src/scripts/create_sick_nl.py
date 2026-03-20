# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.8.3",
#     "huggingface-hub==1.7.2",
# ]
# ///

"""Create the Dutch SICK-NL Entailment dataset and upload it to the HF Hub."""
# TODO use uvx to fix these dependencies

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi


def format(row: dict[str, str]) -> dict[str, str]:
    """Format the raw dataset into the columns needed for EuroEval.

    Args:
        row:
            A row of the dataset as presented by HF Dataset.map().

    Returns:
        the row formatted for EuroEval
    """
    return {
        "text": f"Zin 1: {row['sentence_A']}.\nZin 2: {row['sentence_B']}.",
        "label": row[
            "entailment_label"
        ].lower(),  # labels already in the correct format
        "split": {"TRAIN": "train", "TRIAL": "val", "TEST": "test"}[row["SemEval_set"]],
    }


def main() -> None:
    """Create the Dutch SICK-NL Entailment dataset and upload it to the HF Hub."""
    raw = Dataset.from_csv(
        "https://raw.githubusercontent.com/gijswijnholds/sick_nl/refs/heads/master/data/tasks/sick_nl/SICK_NL.txt",
        sep="\t",
    )
    processed = raw.map(format, remove_columns=raw.column_names)

    dataset = DatasetDict(
        {
            split: processed.filter(lambda s: s == split, input_columns=["split"])
            for split in ("train", "val", "test")
        }
    ).remove_columns(["split"])

    dataset = dataset.shuffle(seed=42)
    dataset["train"] = dataset["train"].select(range(1024))
    dataset["val"] = dataset["val"].select(range(256))
    dataset["test"] = dataset["test"].select(range(1024))

    # Push the dataset to the Hugging Face Hub
    dataset_id = "EuroEval/sick-nl"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
