# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the NorQuAD-mini dataset and upload them to the HF Hub."""

import requests
from datasets import DatasetDict

if __package__:
    from .dataset_utils import create_and_upload_dataset
    from .qa_processing import prepare_squad_dataset, squad_articles_to_dataset
else:
    from dataset_utils import create_and_upload_dataset
    from qa_processing import prepare_squad_dataset, squad_articles_to_dataset


def main() -> None:
    """Create NorQuAD while preserving its training and evaluation partitions."""
    create_and_upload_dataset(
        load_source=load_source,
        process_source=lambda source: prepare_squad_dataset(
            source=source,
            source_splits={"train": "train", "val": "validation", "test": "test"},
        ),
        repository_id="EuroEval/norquad-mini",
    )


def load_source() -> DatasetDict:
    """Download and flatten the three original NorQuAD splits separately.

    Returns:
        Dataset dictionary retaining every original source split.
    """
    url = (
        "https://raw.githubusercontent.com/ltgoslo/NorQuAD/main/data/evaluation"
        "/all/{}_dataset_flattened.json"
    )
    source = {}
    for split, name in {
        "train": "training",
        "validation": "validation",
        "test": "test",
    }.items():
        response = requests.get(url=url.format(name))
        response.raise_for_status()
        source[split] = squad_articles_to_dataset(articles=response.json()["data"])
    return DatasetDict(source)


if __name__ == "__main__":
    main()
