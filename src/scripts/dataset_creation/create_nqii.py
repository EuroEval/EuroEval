# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the NQiI-mini dataset and upload them to the HF Hub."""

from datasets import load_dataset

if __package__:
    from .dataset_utils import create_and_upload_dataset
    from .qa_processing import prepare_squad_dataset
else:
    from dataset_utils import create_and_upload_dataset
    from qa_processing import prepare_squad_dataset


def main() -> None:
    """Create nqii without mixing source training and heldout rows."""
    create_and_upload_dataset(
        load_source=lambda: load_dataset(path="vesteinn/icelandic-qa-NQiI", token=True),
        process_source=lambda source: prepare_squad_dataset(
            source=source,
            source_splits={"train": "train", "val": "validation", "test": "test"},
            answer_mode="provided",
            limits={"train": 1024, "val": 256, "test": 1024},
        ),
        repository_id="EuroEval/nqii-mini",
    )


if __name__ == "__main__":
    main()
