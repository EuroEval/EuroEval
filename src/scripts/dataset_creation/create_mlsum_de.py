# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the German MLSum-de-mini summarisation dataset."""

from datasets import load_dataset

if __package__:
    from .dataset_processing import prepare_mlsum_dataset
    from .dataset_utils import create_and_upload_dataset
else:
    from dataset_processing import prepare_mlsum_dataset
    from dataset_utils import create_and_upload_dataset


def main() -> None:
    """Create MLSum-de using independently filtered source splits."""
    create_and_upload_dataset(
        load_source=lambda: load_dataset(path="GEM/mlsum", name="de", token=True),
        process_source=lambda source: prepare_mlsum_dataset(
            source=source, target_column="target"
        ),
        repository_id="EuroEval/mlsum-de-mini",
    )


if __name__ == "__main__":
    main()
