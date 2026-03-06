"""Load custom dataset configs."""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml
from huggingface_hub import HfApi

from .data_models import DatasetConfig
from .logging_utils import log_once
from .utils import get_hf_token

# Names of the YAML config files that EuroEval recognises in a Hugging Face dataset
# repository. The list is ordered by priority: the first match wins. `eval.yaml` is the
# standard name (aligned with the HuggingFace community-evals convention), while
# `euroeval_config.yaml` is accepted as an alternative.
YAML_CONFIG_FILENAMES = ["eval.yaml", "euroeval_config.yaml"]


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


def load_dataset_config_from_yaml(yaml_path: Path) -> DatasetConfig | None:
    """Load a dataset config from a YAML file.

    The YAML file must contain at least the ``task`` and ``languages`` keys. All other
    keys are optional and correspond to the parameters of
    :class:`~euroeval.data_models.DatasetConfig`.

    Example YAML config::

        task: classification
        languages:
          - en
        labels:
          - positive
          - negative

    Supported task names are the ``name`` attributes of all task objects defined in
    :mod:`euroeval.tasks` (e.g. ``classification``, ``sentiment-classification``,
    ``named-entity-recognition``, etc.).

    Args:
        yaml_path:
            Path to the YAML config file.

    Returns:
        A :class:`~euroeval.data_models.DatasetConfig` built from the YAML data, or
        ``None`` if the file could not be parsed or contains invalid values.
    """
    # Import here to avoid circular imports at module level
    from . import languages as all_languages
    from . import tasks as all_tasks
    from .data_models import Task
    from .languages import Language

    # Build a lookup table: task name → Task object
    task_map: dict[str, Task] = {
        obj.name: obj
        for obj in vars(all_tasks).values()
        if isinstance(obj, Task)
    }

    # Build a lookup table: ISO language code → Language object
    language_map: dict[str, Language] = {
        obj.code: obj
        for obj in vars(all_languages).values()
        if isinstance(obj, Language)
    }

    try:
        with yaml_path.open(encoding="utf-8") as fh:
            raw: Any = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        log_once(
            f"Could not parse YAML config from {yaml_path}: {exc}",
            level=logging.ERROR,
        )
        return None

    if not isinstance(raw, dict):
        log_once(
            f"YAML config at {yaml_path} must be a mapping at the top level.",
            level=logging.ERROR,
        )
        return None

    # --- required fields ---

    task_name: Any = raw.get("task")
    if not isinstance(task_name, str):
        log_once(
            f"YAML config at {yaml_path} must contain a string 'task' field.",
            level=logging.ERROR,
        )
        return None
    task_obj = task_map.get(task_name)
    if task_obj is None:
        log_once(
            f"Unknown task '{task_name}' in YAML config at {yaml_path}. "
            f"Valid task names are: {sorted(task_map)}.",
            level=logging.ERROR,
        )
        return None

    raw_languages: Any = raw.get("languages")
    if not isinstance(raw_languages, list) or not raw_languages:
        log_once(
            f"YAML config at {yaml_path} must contain a non-empty list of language "
            "codes under the 'languages' key.",
            level=logging.ERROR,
        )
        return None
    language_objs: list[Language] = []
    for code in raw_languages:
        lang = language_map.get(str(code))
        if lang is None:
            log_once(
                f"Unknown language code '{code}' in YAML config at {yaml_path}.",
                level=logging.ERROR,
            )
            return None
        language_objs.append(lang)

    # --- optional fields ---

    kwargs: dict[str, Any] = {}

    for str_field in (
        "prompt_prefix",
        "prompt_template",
        "instruction_prompt",
        "input_column",
        "target_column",
    ):
        value = raw.get(str_field)
        if value is not None:
            if not isinstance(value, str):
                log_once(
                    f"Field '{str_field}' in YAML config at {yaml_path} must be a "
                    "string.",
                    level=logging.ERROR,
                )
                return None
            kwargs[str_field] = value

    for int_field in ("num_few_shot_examples", "max_generated_tokens"):
        value = raw.get(int_field)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int):
                log_once(
                    f"Field '{int_field}' in YAML config at {yaml_path} must be an "
                    "integer.",
                    level=logging.ERROR,
                )
                return None
            kwargs[int_field] = value

    labels_raw: Any = raw.get("labels")
    if labels_raw is not None:
        if not isinstance(labels_raw, list):
            log_once(
                f"Field 'labels' in YAML config at {yaml_path} must be a list.",
                level=logging.ERROR,
            )
            return None
        kwargs["labels"] = [str(lbl) for lbl in labels_raw]

    prompt_label_mapping_raw: Any = raw.get("prompt_label_mapping")
    if prompt_label_mapping_raw is not None:
        if not isinstance(prompt_label_mapping_raw, dict):
            log_once(
                f"Field 'prompt_label_mapping' in YAML config at {yaml_path} must be "
                "a mapping.",
                level=logging.ERROR,
            )
            return None
        kwargs["prompt_label_mapping"] = {
            str(k): str(v) for k, v in prompt_label_mapping_raw.items()
        }

    choices_column_raw: Any = raw.get("choices_column")
    if choices_column_raw is not None:
        if isinstance(choices_column_raw, list):
            kwargs["choices_column"] = [str(c) for c in choices_column_raw]
        elif isinstance(choices_column_raw, str):
            kwargs["choices_column"] = choices_column_raw
        else:
            log_once(
                f"Field 'choices_column' in YAML config at {yaml_path} must be a "
                "string or a list of strings.",
                level=logging.ERROR,
            )
            return None

    return DatasetConfig(task=task_obj, languages=language_objs, **kwargs)


def _find_split(splits: list[str], keyword: str) -> str | None:
    """Return the shortest split name containing ``keyword``, or ``None``."""
    candidates = sorted([s for s in splits if keyword in s.lower()], key=len)
    return candidates[0] if candidates else None


def _get_repo_splits(
    hf_api: HfApi,
    dataset_id: str,
) -> tuple[str | None, str | None, str | None]:
    """Return the (train, val, test) split names for a Hugging Face dataset repo.

    Returns a 3-tuple ``(train_split, val_split, test_split)`` where each element is
    either the name of the matching split or ``None`` if no such split exists.
    """
    splits = [
        split["name"]
        for split in hf_api.dataset_info(repo_id=dataset_id).card_data.dataset_info[
            "splits"
        ]
    ]
    return (
        _find_split(splits, "train"),
        _find_split(splits, "val"),
        _find_split(splits, "test"),
    )


def try_get_dataset_config_from_repo(
    dataset_id: str,
    api_key: str | None,
    cache_dir: Path,
    trust_remote_code: bool,
    run_with_cli: bool,
) -> DatasetConfig | None:
    """Try to get a dataset config from a Hugging Face dataset repository.

    The function first looks for a YAML config file (``euroeval_config.yaml`` or
    ``eval.yaml``) which can be loaded without executing any remote code.  If neither
    YAML file is present the function falls back to ``euroeval_config.py``, which
    requires ``trust_remote_code=True``.

    Args:
        dataset_id:
            The ID of the dataset to get the config for.
        api_key:
            The Hugging Face API key to use to check if the repositories have custom
            dataset configs.
        cache_dir:
            The directory to store the cache in.
        trust_remote_code:
            Whether to trust remote code. Only required when loading a Python config
            (``euroeval_config.py``).  YAML configs never require this flag.
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

    repo_files = list(
        hf_api.list_repo_files(
            repo_id=dataset_id, repo_type="dataset", revision="main"
        )
    )

    # ------------------------------------------------------------------
    # 1. Try YAML configs first — no trust_remote_code needed.
    # ------------------------------------------------------------------
    yaml_filename: str | None = next(
        (fn for fn in YAML_CONFIG_FILENAMES if fn in repo_files), None
    )
    if yaml_filename is not None:
        external_config_path = cache_dir / "external_dataset_configs" / dataset_id
        external_config_path.mkdir(parents=True, exist_ok=True)
        hf_api.hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            filename=yaml_filename,
            local_dir=external_config_path,
            local_dir_use_symlinks=False,
        )
        repo_dataset_config = load_dataset_config_from_yaml(
            yaml_path=external_config_path / yaml_filename
        )
        if repo_dataset_config is None:
            return None

        train_split, val_split, test_split = _get_repo_splits(hf_api, dataset_id)
        if test_split is None:
            log_once(
                f"Dataset {dataset_id} does not have a test split, so we cannot load "
                "it. Please ensure that the dataset has a test split.",
                level=logging.ERROR,
            )
            return None

        repo_dataset_config.name = dataset_id
        repo_dataset_config.pretty_name = dataset_id
        repo_dataset_config.source = dataset_id
        repo_dataset_config.train_split = train_split
        repo_dataset_config.val_split = val_split
        repo_dataset_config.test_split = test_split
        return repo_dataset_config

    # ------------------------------------------------------------------
    # 2. Fall back to the Python config — requires trust_remote_code.
    # ------------------------------------------------------------------
    if "euroeval_config.py" not in repo_files:
        log_once(
            f"Dataset {dataset_id} does not have a euroeval_config.py or a YAML config "
            "file (euroeval_config.yaml / eval.yaml), so we cannot load it. Skipping.",
            level=logging.WARNING,
        )
        return None

    # At this point we know that the Python config exists in the repo, so we now check
    # if the user has allowed running code from remote repositories, and abort if not.
    # We abort the entire evaluation here to avoid a double error message, and since it
    # requires the user to explicitly allow it before continuing.
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

    train_split, val_split, test_split = _get_repo_splits(hf_api, dataset_id)
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
