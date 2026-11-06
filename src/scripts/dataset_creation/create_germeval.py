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

from datasets import load_dataset

if __package__:
    from .dataset_processing import prepare_ner_dataset
    from .dataset_utils import create_and_upload_dataset
else:
    from dataset_processing import prepare_ner_dataset
    from dataset_utils import create_and_upload_dataset


def main() -> None:
    """Create the germeval NER dataset from its original source splits."""
    label_mapping = {
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
    create_and_upload_dataset(
        load_source=lambda: load_dataset(path="germeval_14", token=True),
        process_source=lambda source: prepare_ner_dataset(
            source=source,
            label_mapping=label_mapping,
            limits={"train": 1024, "val": 256, "test": 1024},
        ),
        repository_id="EuroEval/germeval-mini",
    )


if __name__ == "__main__":
    main()
