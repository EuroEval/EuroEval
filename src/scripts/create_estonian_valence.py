# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
#     "requests==2.32.5",
# ]
# ///

"""Create the Estonian valence dataset and upload to HF Hub."""

from datasets import DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the Estonian valence dataset and upload to HF Hub."""
    target_repo_id = "EuroEval/estonian-valence"

    # Use the reupload available on HuggingFace
    ds = load_dataset("kardosdrur/estonian-valence")

    # Standardize the columns
    ds = ds.rename_columns({"paragraph": "text", "valence": "label"})
    ds = ds.select_columns(["text", "label"])

    # Remove examples with the fourth label for consistency with the other sentiment
    # datasets
    ds = ds.filter(
        lambda rows: [el != "vastuoluline" for el in rows["label"]], batched=True
    )

    # Convert the labels to English labels
    label_mapping = {
        "negatiivne": "negative",
        "neutraalne": "neutral",
        "positiivne": "positive",
    }
    ds = ds.map(
        lambda rows: {"label": [label_mapping[el] for el in rows["label"]]},
        batched=True,
    )

    # Target split sizes
    train_size = 1024
    val_size = 256
    test_size = 2048

    # How many samples are missing from the test split?
    missing_test_size = test_size - len(ds["test"])

    # Reallocate the samples to the splits
    new_train = ds["train"].select(range(train_size))
    new_val = ds["train"].skip(train_size).select(range(val_size))
    new_test = concatenate_datasets(
        [
            ds["test"],
            ds["train"]
            .skip(train_size)
            .skip(val_size)
            .select(range(missing_test_size)),
        ]
    )

    new_ds = DatasetDict({"train": new_train, "val": new_val, "test": new_test})

    # Sanity check the labels
    expected_labels = {"negative", "neutral", "positive"}
    for key in new_ds:
        cur_labels = set(new_ds[key].unique("label"))
        if cur_labels != expected_labels:
            raise ValueError(f"Incorrect labels for {key}: {cur_labels}")

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(target_repo_id, repo_type="dataset")
    except HTTPError:
        pass

    new_ds.push_to_hub(target_repo_id, private=True)


if __name__ == "__main__":
    main()
