# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
#     "requests==2.32.5",
# ]
# ///

"""Create the Estonian Winogrande dataset and upload to HF Hub."""

from typing import MutableMapping

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the Estonian Winogrande dataset and upload to HF Hub."""
    target_repo_id = "EuroEval/winogrande-et"

    # start from the official source
    human_ds = load_dataset("tartuNLP/winogrande_et", "human_translated")
    mt_ds = load_dataset("tartuNLP/winogrande_et", "machine_translated")

    # target split sizes
    train_size = 1024
    val_size = 256
    test_size = 2048

    # we don't have human translations for train and dev
    ds = DatasetDict(
        {
            "train": mt_ds["train"].select(range(train_size)),
            "val": mt_ds["dev"].select(range(val_size)),
            "test": human_ds["test"].select(
                range(min(test_size, len(human_ds["test"])))
            ),
        }
    )

    # please don't share the answers explicitly though
    ds["test"] = ds["test"].map(lambda row: {"answer": row["qID"][-1]})

    # add options to the text and transform labels to letters
    ds = ds.map(add_options_and_label)

    # retain only used columns
    ds = ds.select_columns(["text", "label"])

    try:
        api = HfApi()
        api.delete_repo(target_repo_id, repo_type="dataset")
    except HTTPError:
        pass

    ds.push_to_hub(target_repo_id, private=True)


def add_options_and_label(row: MutableMapping) -> MutableMapping:
    """Add options to the text and transform labels to letters.

    Args:
        row:
            A row from the dataset.

    Returns:
        A dictionary with the modified text and label.
    """
    letter_mapping = {"1": "a", "2": "b"}

    original_text = row["sentence"]
    option_1 = row["option1"]
    option_2 = row["option2"]

    new_text = f"{original_text}\nVastusevariandid:\na. {option_1}\nb. {option_2}"

    answer = row["answer"]
    if answer not in letter_mapping.keys():
        raise ValueError(
            f"Invalid answer: {answer}, possible values are {letter_mapping.keys()}"
        )
    label = letter_mapping[answer]

    return {"text": new_text, "label": label}


if __name__ == "__main__":
    main()
