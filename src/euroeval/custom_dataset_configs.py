"""Load custom dataset configs."""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

from huggingface_hub import HfApi

from .data_models import DatasetConfig
from .logging_utils import log_once
from .utils import get_hf_token


def load_custom_datasets_module(custom_datasets_file: Path) -> ModuleType | None:
    """Load the custom datasets module if it exists.

    Args:
        custom_datasets_file:
            The path to the custom datasets module.

    Returns:
        The custom datasets module, or None if it does not exist.
    """
    if custom_datasets_file.exists():
        spec = importlib.util.spec_from_file_location(
            name="custom_datasets_module", location=str(custom_datasets_file.resolve())
        )
        if spec is None:
            log_once(
                "Could not load the spec for the custom datasets file from "
                f"{custom_datasets_file.resolve()}.",
                level=logging.ERROR,
            )
            return None
        module = importlib.util.module_from_spec(spec=spec)
        if spec.loader is None:
            log_once(
                "Could not load the module for the custom datasets file from "
                f"{custom_datasets_file.resolve()}.",
                level=logging.ERROR,
            )
            return None
        spec.loader.exec_module(module)
        return module
    return None


def try_get_dataset_config_from_repo(
    dataset_id: str,
    api_key: str | None,
    cache_dir: Path,
    trust_remote_code: bool,
    run_with_cli: bool,
) -> DatasetConfig | None:
    """Try to get a dataset config from a Hugging Face dataset repository.

    Args:
        dataset_id:
            The ID of the dataset to get the config for.
        api_key:
            The Hugging Face API key to use to check if the repositories have custom
            dataset configs.
        cache_dir:
            The directory to store the cache in.
        trust_remote_code:
            Whether to trust remote code. If this is not set to True, then we will not
            load the dataset config.
        run_with_cli:
            Whether the code is being run with the CLI.

    Returns:
        The dataset config if it exists, otherwise None.
    """
    # Check if the dataset ID is a Hugging Face dataset ID, abort if it isn't
    token = get_hf_token(api_key=api_key)
    hf_api = HfApi(token=token)
    if not hf_api.repo_exists(repo_id=dataset_id, repo_type="dataset"):
        return None

    # Check if the repository has a euroeval_config.py file, abort if it doesn't
    repo_files = hf_api.list_repo_files(
        repo_id=dataset_id, repo_type="dataset", revision="main"
    )
    if "euroeval_config.py" not in repo_files:
        log_once(
            f"Dataset {dataset_id} does not have a euroeval_config.py file, so we "
            "cannot load it. Skipping.",
            level=logging.WARNING,
        )
        return None

    # At this point we know that the config exists in the repo, so we now check if the
    # user has allowed running code from remote repositories, and abort if not. We abort
    # the entire evaluation here to avoid a double error message, and since it requires
    # the user to explicitly allow it before continuing.
    if not trust_remote_code:
        rerunning_msg = (
            "the --trust-remote-code flag"
            if run_with_cli
            else "`trust_remote_code=True`"
        )
        log_once(
            f"The dataset {dataset_id} exists on the Hugging Face Hub and has a "
            "euroeval_config.py file, but remote code is not allowed. Please rerun "
            f"this with {rerunning_msg} if you trust the code in this repository.",
            level=logging.ERROR,
        )
        sys.exit(1)

    # Fetch the euroeval_config.py file, abort if loading failed
    external_config_path = cache_dir / "external_dataset_configs" / dataset_id
    external_config_path.mkdir(parents=True, exist_ok=True)
    hf_api.hf_hub_download(
        repo_id=dataset_id,
        repo_type="dataset",
        filename="euroeval_config.py",
        local_dir=external_config_path,
        local_dir_use_symlinks=False,
    )
    module = load_custom_datasets_module(
        custom_datasets_file=external_config_path / "euroeval_config.py"
    )
    if module is None:
        return None

    # Check that there is exactly one dataset config, abort if there isn't
    repo_dataset_configs = [
        cfg for cfg in vars(module).values() if isinstance(cfg, DatasetConfig)
    ]
    if not repo_dataset_configs:
        return None  # Already warned the user in this case, so we just skip
    elif len(repo_dataset_configs) > 1:
        log_once(
            f"Dataset {dataset_id} has multiple dataset configurations. Please ensure "
            "that only a single DatasetConfig is defined in the `euroeval_config.py` "
            "file.",
            level=logging.WARNING,
        )
        return None

    # Get the dataset split names
    splits = [
        split["name"]
        for split in hf_api.dataset_info(repo_id=dataset_id).card_data.dataset_info[
            "splits"
        ]
    ]
    train_split_candidates = sorted(
        [split for split in splits if "train" in split.lower()], key=len
    )
    val_split_candidates = sorted(
        [split for split in splits if "val" in split.lower()], key=len
    )
    test_split_candidates = sorted(
        [split for split in splits if "test" in split.lower()], key=len
    )
    train_split = train_split_candidates[0] if train_split_candidates else None
    val_split = val_split_candidates[0] if val_split_candidates else None
    test_split = test_split_candidates[0] if test_split_candidates else None
    if test_split is None:
        log_once(
            f"Dataset {dataset_id} does not have a test split, so we cannot load it. "
            "Please ensure that the dataset has a test split.",
            level=logging.ERROR,
        )
        return None

    # Set up the config with the repo information
    repo_dataset_config = repo_dataset_configs[0]
    repo_dataset_config.name = dataset_id
    repo_dataset_config.pretty_name = dataset_id
    repo_dataset_config.source = dataset_id
    repo_dataset_config.train_split = train_split
    repo_dataset_config.val_split = val_split
    repo_dataset_config.test_split = test_split

    return repo_dataset_config
