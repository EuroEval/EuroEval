"""Shared helpers for capping dataset splits and uploading them to the Hub."""

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

SEED = 4242
MAX_TRAIN = 1024
MAX_VAL = 256
MAX_TEST = 2048


def make_dataset_dict(
    train: Dataset, validation: Dataset, test: Dataset
) -> DatasetDict:
    """Cap source splits and expose validation under EuroEval's ``val`` name.

    Args:
        train:
            Source training split.
        validation:
            Source validation split.
        test:
            Source test split.

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

    Args:
        dataset:
            Source dataset split to cap.
        maximum:
            Maximum number of rows to retain.

    Returns:
        The capped dataset.
    """
    if dataset.num_rows <= maximum:
        return dataset
    return dataset.shuffle(seed=SEED).select(range(maximum))


def upload_private(dataset: DatasetDict, repository_id: str) -> None:
    """Replace a Hub dataset and fail unless the resulting repository is private.

    Args:
        dataset:
            Dataset to upload.
        repository_id:
            Hub dataset repository ID.

    Raises:
        RuntimeError:
            If the resulting repository is public.
    """
    preflight_private_upload(repository_id=repository_id)
    dataset.push_to_hub(repo_id=repository_id, private=True)
    repository = HfApi().repo_info(repo_id=repository_id, repo_type="dataset")
    if not repository.private:
        raise RuntimeError(f"Dataset repository {repository_id} is not private")


def preflight_private_upload(repository_id: str) -> None:
    """Reject an existing public Hub dataset before uploading any bytes.

    A missing repository is allowed because ``push_to_hub`` creates it as private.

    Args:
        repository_id:
            Hub dataset repository ID.

    Raises:
        RuntimeError:
            If the existing repository is public.
    """
    api = HfApi()
    try:
        repository = api.repo_info(repo_id=repository_id, repo_type="dataset")
    except RepositoryNotFoundError:
        return
    if not repository.private:
        raise RuntimeError(f"Dataset repository {repository_id} is not private")
