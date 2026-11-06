# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "click==8.1.3",
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the FQuAD-mini NER dataset and upload it to the HF Hub."""

import json
from pathlib import Path

import click
from datasets import DatasetDict

if __package__:
    from .dataset_utils import upload_private
    from .qa_processing import prepare_squad_dataset, squad_articles_to_dataset
else:
    from dataset_utils import upload_private
    from qa_processing import prepare_squad_dataset, squad_articles_to_dataset


@click.command(
    help="Create FQuAD-mini from the raw data requested from https://fquad.illuin.tech."
)
@click.argument("data_dir", type=str)
def main(data_dir: str) -> None:
    """Create FQuAD from local source files, inferring valid answer offsets.

    Args:
        data_dir:
            Directory containing the original train.json and valid.json files.
    """
    source = DatasetDict(
        {
            split: squad_articles_to_dataset(
                articles=json.loads((Path(data_dir) / filename).read_text())["data"]
            )
            for split, filename in {
                "train": "train.json",
                "validation": "valid.json",
            }.items()
        }
    )
    dataset = prepare_squad_dataset(
        source=source,
        source_splits={"train": "train", "val": "validation", "test": "validation"},
        answer_mode="infer",
    )
    upload_private(dataset=dataset, repository_id="EuroEval/fquad-mini")


if __name__ == "__main__":
    main()
