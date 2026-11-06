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

from datasets import load_dataset

if __package__:
    from .dataset_processing import CONLL_LABEL_MAPPING, prepare_ner_dataset
    from .dataset_utils import create_and_upload_dataset
else:
    from dataset_processing import CONLL_LABEL_MAPPING, prepare_ner_dataset
    from dataset_utils import create_and_upload_dataset


def main() -> None:
    """Create the conll-nl NER dataset from its original source splits."""
    label_mapping = CONLL_LABEL_MAPPING
    create_and_upload_dataset(
        load_source=lambda: load_dataset(
            path="eriktks/conll2002", token=True, name="nl"
        ),
        process_source=lambda source: prepare_ner_dataset(
            source=source,
            label_mapping=label_mapping,
            limits={"train": 1024, "val": 256, "test": 1024},
        ),
        repository_id="EuroEval/conll-nl-mini",
    )


if __name__ == "__main__":
    main()
