"""Load custom dataset configs.

This module provides the main entry point for loading dataset configurations from
Hugging Face repositories, including Python-based configs. YAML-specific loading
logic lives in the `yaml_config` module.
"""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

from huggingface_hub import HfApi

from .data_models import DatasetConfig
from .logging_utils import log_once
from .utils import get_hf_token
from .yaml_config import get_repo_splits, load_yaml_config


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
                message=(
                    "Could not load the spec for the custom datasets file from "
                    f"{custom_datasets_file.resolve()}."
                ),
                level=logging.ERROR,
            )
            return None
        module = importlib.util.module_from_spec(spec=spec)
        if spec.loader is None:
            log_once(
                message=(
                    "Could not load the module for the custom datasets file from "
                    f"{custom_datasets_file.resolve()}."
                ),
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

    The function first looks for a YAML config file (`eval.yaml`) which can be
    loaded without executing any remote code. If no YAML file is present the
    function falls back to `euroeval_config.py`, which requires
    `trust_remote_code=True`.

    Args:
        dataset_id:
            The ID of the dataset to get the config for.
        api_key:
            The Hugging Face API key to use to check if the repositories have
            custom dataset configs.
        cache_dir:
            The directory to store the cache in.
        trust_remote_code:
            Whether to trust remote code. Only required when loading a Python
            config (`euroeval_config.py`). YAML configs never require this flag.
        run_with_cli:
            Whether the code is being run with the CLI.

    Returns:
        The dataset config if it exists, otherwise None.
    """
    token = get_hf_token(api_key=api_key)
    hf_api = HfApi(token=token)
    if not hf_api.repo_exists(repo_id=dataset_id, repo_type="dataset"):
        return None

    repo_files = list(
        hf_api.list_repo_files(repo_id=dataset_id, repo_type="dataset", revision="main")
    )

    if "eval.yaml" in repo_files:
        return load_yaml_config(
            hf_api=hf_api, dataset_id=dataset_id, cache_dir=cache_dir
        )

    return load_python_config(
        hf_api=hf_api,
        dataset_id=dataset_id,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        run_with_cli=run_with_cli,
    )


def load_python_config(
    hf_api: HfApi,
    dataset_id: str,
    cache_dir: Path,
    trust_remote_code: bool,
    run_with_cli: bool,
) -> DatasetConfig | None:
    """Load a dataset config from a euroeval_config.py file in a Hugging Face repo.

    Args:
        hf_api:
            The Hugging Face API object.
        dataset_id:
            The ID of the dataset to get the config for.
        cache_dir:
            The directory to store the cache in.
        trust_remote_code:
            Whether to trust remote code.
        run_with_cli:
            Whether the code is being run with the CLI.

    Returns:
        The dataset config if it exists, otherwise None.
    """
    repo_files = list(
        hf_api.list_repo_files(repo_id=dataset_id, repo_type="dataset", revision="main")
    )

    if "euroeval_config.py" not in repo_files:
        log_once(
            message=(
                f"Dataset {dataset_id} does not have a euroeval_config.py or a YAML "
                "config file (eval.yaml), so we cannot load it. Skipping."
            ),
            level=logging.WARNING,
        )
        return None

    if not trust_remote_code:
        rerunning_msg = (
            "the --trust-remote-code flag"
            if run_with_cli
            else "`trust_remote_code=True`"
        )
        log_once(
            message=(
                f"The dataset {dataset_id} exists on the Hugging Face Hub and has a "
                "euroeval_config.py file, but remote code is not allowed. Please "
                f"rerun this with {rerunning_msg} if you trust the code in this "
                "repository."
            ),
            level=logging.ERROR,
        )
        sys.exit(1)

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

    repo_dataset_configs = [
        cfg for cfg in vars(module).values() if isinstance(cfg, DatasetConfig)
    ]
    if not repo_dataset_configs:
        return None
    if len(repo_dataset_configs) > 1:
        log_once(
            message=(
                f"Dataset {dataset_id} has multiple dataset configurations. Please "
                "ensure that only a single DatasetConfig is defined in the "
                "`euroeval_config.py` file."
            ),
            level=logging.WARNING,
        )
        return None

    train_split, val_split, test_split = get_repo_splits(
        hf_api=hf_api, dataset_id=dataset_id
    )
    if test_split is None:
        log_once(
            message=(
                f"Dataset {dataset_id} does not have a test split, so we cannot load "
                "it. Please ensure that the dataset has a test split."
            ),
            level=logging.ERROR,
        )
        return None

    if train_split is None and val_split is not None:
        log_once(
            message=(
                f"Dataset {dataset_id!r} has no training split. Using the validation "
                f"split {val_split!r} as the training split instead."
            ),
            level=logging.DEBUG,
        )
        train_split = val_split
        val_split = None

    repo_dataset_config = repo_dataset_configs[0]
    repo_dataset_config.name = dataset_id
    repo_dataset_config.pretty_name = dataset_id
    repo_dataset_config.source = dataset_id
    repo_dataset_config.train_split = train_split
    repo_dataset_config.val_split = val_split
    repo_dataset_config.test_split = test_split

    return repo_dataset_config
