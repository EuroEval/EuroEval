# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the old SQuAD-nl-mini dataset and upload them to the HF Hub."""

from datasets import load_dataset

if __package__:
    from .dataset_utils import create_and_upload_dataset
    from .qa_processing import prepare_squad_dataset
else:
    from dataset_utils import create_and_upload_dataset
    from qa_processing import prepare_squad_dataset


def main() -> None:
    """Create squad-nl without mixing source training and heldout rows."""
    create_and_upload_dataset(
        load_source=lambda: load_dataset(path="yhavinga/squad_v2_dutch", token=True),
        process_source=lambda source: prepare_squad_dataset(
            source=source,
            source_splits={"train": "train", "val": "validation", "test": "validation"},
            answer_mode="infer",
        ),
        repository_id="EuroEval/squad-nl-mini",
    )


if __name__ == "__main__":
    main()
