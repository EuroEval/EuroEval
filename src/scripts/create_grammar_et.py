# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
#     "requests==2.32.5",
# ]
# ///

"""Create the Estonian Grammar dataset and upload to HF Hub."""

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the Estonian Grammar dataset and upload to HF Hub."""
    target_repo_id = "EuroEval/grammar-et"

    # start from the official source
    ds = load_dataset("TalTechNLP/grammar_et")

    # remove a small number of examples where the origal was correct
    ds = ds.filter(lambda row: row["original"] != row["correct"])

    # each row contains a correct/incorrect pair
    train_size = 1024 // 2
    val_size = 256 // 2
    test_size = 2048 // 2

    # rearrange the examples
    train_ds = ds["train"].select(range(train_size))
    val_ds = ds["train"].skip(train_size).select(range(val_size))
    test_ds = concatenate_datasets(
        [
            ds["test"],
            ds["train"]
            .skip(train_size + val_size)
            .select(range(test_size - ds["test"].num_rows)),
        ]
    )
    ds = DatasetDict({"train": train_ds, "val": val_ds, "test": test_ds})

    # separate pairs into individual rows with labels
    new_ds = DatasetDict({})
    for split in ds:
        cur_ds = ds[split]
        original_ds = Dataset.from_dict(
            {
                "text": cur_ds["original"],
                "label": ["incorrect" for _ in range(cur_ds.num_rows)],
            }
        )
        corrected_ds = Dataset.from_dict(
            {
                "text": cur_ds["correct"],
                "label": ["correct" for _ in range(cur_ds.num_rows)],
            }
        )
        new_ds[split] = concatenate_datasets([original_ds, corrected_ds])
    new_ds = new_ds.shuffle(seed=42)

    try:
        api = HfApi()
        api.delete_repo(target_repo_id, repo_type="dataset")
    except HTTPError:
        pass

    new_ds.push_to_hub(target_repo_id, private=True)


if __name__ == "__main__":
    main()
