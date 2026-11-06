# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the CoNLL-NL-mini NER dataset and upload it to the HF Hub."""

from datasets import DatasetDict, load_dataset

if __package__:
    from .dataset_utils import preflight_private_upload
else:
    from dataset_utils import preflight_private_upload

if __package__:
    from .dataset_processing import (
        dataset_dict_from_frames,
        process_ner_frame,
        process_source_splits,
    )
else:
    from dataset_processing import (
        dataset_dict_from_frames,
        process_ner_frame,
        process_source_splits,
    )


def main() -> None:
    """Create the CoNLL-NL-mini NER dataset and uploads it to the HF Hub."""
    # Define dataset ID
    repo_id = "eriktks/conll2002"

    # Download the dataset
    dataset = load_dataset(path=repo_id, name="nl", token=True)
    assert isinstance(dataset, DatasetDict)

    ner_mapping = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
        7: "B-MISC",
        8: "I-MISC",
    }
    frames = {
        "train": process_ner_frame(
            frame=dataset["train"].to_pandas(), label_mapping=ner_mapping
        ),
        "val": process_ner_frame(
            frame=dataset["validation"].to_pandas(), label_mapping=ner_mapping
        ),
        "test": process_ner_frame(
            frame=dataset["test"].to_pandas(), label_mapping=ner_mapping
        ),
    }
    limits = {"train": 1024, "val": 256, "test": 1024}
    frames = process_source_splits(frames=frames, limits=limits)
    dataset = dataset_dict_from_frames(
        frames=frames, columns=["tokens", "text", "labels"]
    )

    # Create dataset ID
    dataset_id = "EuroEval/conll-nl-mini"

    # Push the dataset to the Hugging Face Hub
    preflight_private_upload(repository_id=dataset_id)
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
