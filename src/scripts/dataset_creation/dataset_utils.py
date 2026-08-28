"""Shared helpers for capping dataset splits and uploading them to the Hub."""

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

SEED = 4242
MAX_TRAIN = 1024
MAX_VAL = 256
MAX_TEST = 2048


def make_dataset_dict(
    train: Dataset, validation: Dataset, test: Dataset
) -> DatasetDict:
    """Cap source splits and expose validation under EuroEval's ``val`` name.

    Returns:
        A dataset dictionary with independently capped splits.
    """
    return DatasetDict(
        {
            "train": cap_split(dataset=train, maximum=MAX_TRAIN),
            "val": cap_split(dataset=validation, maximum=MAX_VAL),
            "test": cap_split(dataset=test, maximum=MAX_TEST),
        }
    )


def cap_split(dataset: Dataset, maximum: int) -> Dataset:
    """Sample at most ``maximum`` rows without crossing source split boundaries.

    Returns:
        The capped dataset.
    """
    if dataset.num_rows <= maximum:
        return dataset
    return dataset.shuffle(seed=SEED).select(range(maximum))


def upload_private(dataset: DatasetDict, repository_id: str) -> None:
    """Replace a Hub dataset and fail unless the resulting repository is private.

    Raises:
        RuntimeError:
            If the resulting repository is public.
    """
    api = HfApi()
    api.delete_repo(repo_id=repository_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(repo_id=repository_id, private=True)
    repository = api.repo_info(repo_id=repository_id, repo_type="dataset")
    if not repository.private:
        raise RuntimeError(f"Dataset repository {repository_id} is not private")
