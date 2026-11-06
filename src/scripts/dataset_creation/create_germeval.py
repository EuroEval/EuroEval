# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the GermEval-mini NER dataset and upload it to the HF Hub."""

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
    """Create the GermEval-mini NER dataset and uploads it to the HF Hub."""
    # Define dataset ID
    repo_id = "germeval_14"

    # Download the dataset
    dataset = load_dataset(path=repo_id, token=True)
    assert isinstance(dataset, DatasetDict)

    ner_mapping = {
        0: "O",  # Original: O
        1: "B-LOC",  # Original: B-LOC
        2: "I-LOC",  # Original: I-LOC
        3: "O",  # Original: B-LOCderiv
        4: "O",  # Original: I-LOCderiv
        5: "B-LOC",  # Original: B-LOCpart
        6: "I-LOC",  # Original: I-LOCpart
        7: "B-ORG",  # Original: B-ORG
        8: "I-ORG",  # Original: I-ORG
        9: "O",  # Original: B-ORGderiv
        10: "O",  # Original: I-ORGderiv
        11: "B-ORG",  # Original: B-ORGpart
        12: "I-ORG",  # Original: I-ORGpart
        13: "B-MISC",  # Original: B-OTH
        14: "I-MISC",  # Original: I-OTH
        15: "O",  # Original: B-OTHderiv
        16: "O",  # Original: I-OTHderiv
        17: "B-MISC",  # Original: B-OTHpart
        18: "I-MISC",  # Original: I-OTHpart
        19: "B-PER",  # Original: B-PER
        20: "I-PER",  # Original: I-PER
        21: "O",  # Original: B-PERderiv
        22: "O",  # Original: I-PERderiv
        23: "B-PER",  # Original: B-PERpart
        24: "I-PER",  # Original: I-PERpart
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
    dataset_id = "EuroEval/germeval-mini"

    # Push the dataset to the Hugging Face Hub
    preflight_private_upload(repository_id=dataset_id)
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
