"""Load custom dataset configs.

This module provides utilities for loading dataset configurations from YAML files,
Hugging Face repositories, and custom Python modules. It supports both EuroEval-specific
and Inspect AI-compatible formats.
"""

import importlib.util
import logging
import sys
import typing as t
from pathlib import Path
from types import ModuleType

import yaml
from huggingface_hub import HfApi

from .data_models import DatasetConfig, Task
from .languages import Language, get_all_languages
from .logging_utils import log_once
from .tasks import get_all_tasks
from .utils import get_hf_token

if t.TYPE_CHECKING:
    from .data_models import Task


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


def infer_task_from_inspect_ai(
    raw: dict[str, t.Any], task_map: dict[str, Task]
) -> Task | None:
    """Try to infer the EuroEval task from Inspect AI YAML fields.

    Currently detects:

    * A solver with `name: multiple_choice` in `tasks[0].solvers`
      → `multiple-choice`
    * A `choices` key in `tasks[0].field_spec` → `multiple-choice`

    Args:
        raw:
            The raw YAML data.
        task_map:
            The mapping from task names to task objects.

    Returns:
        The inferred task, or None if the task cannot be inferred.
    """
    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        return None
    first_task = tasks_raw[0]
    if not isinstance(first_task, dict):
        return None

    solvers = first_task.get("solvers")
    if isinstance(solvers, list):
        for solver in solvers:
            if isinstance(solver, dict) and solver.get("name") == "multiple_choice":
                return task_map.get("multiple-choice")

    field_spec = first_task.get("field_spec")
    if isinstance(field_spec, dict) and "choices" in field_spec:
        return task_map.get("multiple-choice")

    return None


def load_yaml_file(yaml_path: Path) -> dict[str, t.Any] | None:
    """Load a YAML file and return its contents as a dictionary.

    Args:
        yaml_path:
            Path to the YAML config file.

    Returns:
        The parsed YAML content as a dictionary, or None if parsing failed.
    """
    try:
        with yaml_path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        log_once(
            f"Could not parse YAML config from {yaml_path}: {exc}", level=logging.ERROR
        )
        return None

    if not isinstance(raw, dict):
        log_once(
            f"YAML config at {yaml_path} must be a mapping at the top level.",
            level=logging.ERROR,
        )
        return None

    return raw


def promote_field_spec_fields(raw: dict[str, t.Any]) -> None:
    """Promote column names from field_spec to top-level keys.

    Promotes the following mappings when the top-level key is not already set:
    - field_spec.input → input_column
    - field_spec.target → target_column (only if plain, not literal/int)
    - field_spec.choices → choices_column
    - tasks[0].split → test_split

    Args:
        raw:
            The parsed YAML data to modify in place.
    """
    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        return

    first_task = tasks_raw[0]
    if not isinstance(first_task, dict):
        return

    field_spec = first_task.get("field_spec")
    if isinstance(field_spec, dict):
        # input → input_column
        if "input" in field_spec and "input_column" not in raw:
            raw["input_column"] = field_spec["input"]

        # target → target_column (skip literals and integers)
        if "target" in field_spec and "target_column" not in raw:
            target = field_spec["target"]
            if isinstance(target, str) and not target.startswith("literal:"):
                raw["target_column"] = target

        # choices → choices_column
        if "choices" in field_spec and "choices_column" not in raw:
            raw["choices_column"] = field_spec["choices"]

    # tasks[0].split → test_split
    split_val = first_task.get("split")
    if isinstance(split_val, str) and split_val and "test_split" not in raw:
        raw["test_split"] = split_val


def validate_and_get_task(raw: dict[str, t.Any], yaml_path: Path) -> Task | None:
    """Validate the task field or infer it from Inspect AI hints.

    Args:
        raw:
            The parsed YAML data.
        yaml_path:
            Path to the YAML config file (for error messages).

    Returns:
        A valid Task object, or None if validation failed.
    """
    task_map = get_all_tasks()
    task_name = raw.get("task")

    if isinstance(task_name, str):
        task_obj = task_map.get(task_name)
        if task_obj is None:
            log_once(
                f"Unknown task '{task_name}' in YAML config at {yaml_path}. "
                f"Valid task names are: {sorted(task_map)}.",
                level=logging.ERROR,
            )
            return None
    else:
        task_obj = infer_task_from_inspect_ai(raw, task_map)
        if task_obj is None:
            log_once(
                f"YAML config at {yaml_path} does not contain a 'task' field and the "
                "task could not be inferred from the Inspect AI 'tasks' block. "
                "Add a top-level 'task' key (e.g. 'task: classification') or include "
                "a 'multiple_choice' solver / a 'choices' field_spec entry so that "
                "the task can be detected automatically.",
                level=logging.ERROR,
            )
            return None

    return task_obj


def parse_languages(
    raw: dict[str, t.Any], fallback_codes: list[str] | None, yaml_path: Path
) -> list[Language] | None:
    """Parse language codes from YAML or use fallbacks.

    Args:
        raw:
            The parsed YAML data.
        fallback_codes:
            ISO 639-1 language codes to use as a fallback.
        yaml_path:
            Path to the YAML config file (for error messages).

    Returns:
        A list of Language objects, or None if validation failed.
    """
    language_map = get_all_languages()
    raw_languages = raw.get("languages")

    if isinstance(raw_languages, list) and raw_languages:
        language_codes: list[str] = [str(c) for c in raw_languages]
    elif fallback_codes:
        log_once(
            f"YAML config at {yaml_path} does not contain a 'languages' key. "
            "Using language(s) from the repository metadata: "
            f"{fallback_codes}.",
            level=logging.DEBUG,
        )
        language_codes = fallback_codes
    else:
        log_once(
            f"YAML config at {yaml_path} does not contain a 'languages' key and no "
            "language metadata could be found for this repository. Defaulting to "
            "English. Add a top-level 'languages' key to the YAML file "
            "(e.g. 'languages:\\n  - en') to override this.",
            level=logging.WARNING,
        )
        language_codes = ["en"]

    language_objs: list[Language] = []
    for code in language_codes:
        lang = language_map.get(code)
        if lang is None:
            log_once(
                f"Unknown language code '{code}' in YAML config at {yaml_path}.",
                level=logging.ERROR,
            )
            return None
        language_objs.append(lang)

    return language_objs


def parse_string_field(
    raw: dict[str, t.Any], field_name: str, yaml_path: Path
) -> str | None:
    """Parse and validate a string field from YAML.

    Args:
        raw:
            The parsed YAML data.
        field_name:
            The name of the field to parse.
        yaml_path:
            Path to the YAML config file (for error messages).

    Returns:
        The field value as a string, or None if validation failed.
    """
    value = raw.get(field_name)
    if value is not None:
        if not isinstance(value, str):
            log_once(
                f"Field '{field_name}' in YAML config at {yaml_path} must be a string.",
                level=logging.ERROR,
            )
            return None
        return value
    return None


def parse_int_field(
    raw: dict[str, t.Any], field_name: str, yaml_path: Path
) -> int | None:
    """Parse and validate an integer field from YAML.

    Args:
        raw:
            The parsed YAML data.
        field_name:
            The name of the field to parse.
        yaml_path:
            Path to the YAML config file (for error messages).

    Returns:
        The field value as an integer, or None if validation failed.
    """
    value = raw.get(field_name)
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int):
            log_once(
                f"Field '{field_name}' in YAML config at {yaml_path} must be "
                "an integer.",
                level=logging.ERROR,
            )
            return None
        return value
    return None


def build_kwargs(
    raw: dict[str, t.Any], yaml_path: Path
) -> dict[str, str | int | list[str] | dict[str, str]] | None:
    """Build the kwargs dictionary from optional YAML fields.

    Args:
        raw:
            The parsed YAML data.
        yaml_path:
            Path to the YAML config file (for error messages).

    Returns:
        A dictionary of keyword arguments for DatasetConfig,
            or `None` if validation failed.
    """
    kwargs: dict[str, str | int | list[str] | dict[str, str]] = {}

    # String fields
    for field_name in (
        "prompt_prefix",
        "prompt_template",
        "instruction_prompt",
        "input_column",
        "target_column",
        "test_split",
    ):
        value = parse_string_field(raw, field_name, yaml_path)
        if value is not None:
            kwargs[field_name] = value

    # Integer fields
    for field_name in ("num_few_shot_examples", "max_generated_tokens"):
        value = parse_int_field(raw, field_name, yaml_path)
        if value is not None:
            kwargs[field_name] = value

    # labels: list of strings
    labels_raw = raw.get("labels")
    if labels_raw is not None:
        if not isinstance(labels_raw, list):
            log_once(
                f"Field 'labels' in YAML config at {yaml_path} must be a list.",
                level=logging.ERROR,
            )
            return None
        kwargs["labels"] = [str(lbl) for lbl in labels_raw]

    # prompt_label_mapping: dict of string to string
    prompt_label_mapping_raw = raw.get("prompt_label_mapping")
    if prompt_label_mapping_raw is not None:
        if not isinstance(prompt_label_mapping_raw, dict):
            log_once(
                "Field 'prompt_label_mapping' in YAML config at"
                f" {yaml_path} must be a mapping.",
                level=logging.ERROR,
            )
            return None
        kwargs["prompt_label_mapping"] = {
            str(k): str(v) for k, v in prompt_label_mapping_raw.items()
        }

    # choices_column: string or list of strings
    choices_column_raw = raw.get("choices_column")
    if choices_column_raw is not None:
        if isinstance(choices_column_raw, list):
            kwargs["choices_column"] = [str(c) for c in choices_column_raw]
        elif isinstance(choices_column_raw, str):
            kwargs["choices_column"] = choices_column_raw
        else:
            log_once(
                "Field 'choices_column' in YAML config at"
                f" {yaml_path} must be a string or a list of strings.",
                level=logging.ERROR,
            )
            return None

    return kwargs


def load_dataset_config_from_yaml(
    yaml_path: Path, fallback_language_codes: list[str] | None = None
) -> DatasetConfig | None:
    """Load a dataset config from a YAML file.

    The file is fully compatible with the `Inspect AI eval.yaml format
    <https://inspect.aisi.org.uk/tasks.html#hugging-face>`_.  The EuroEval-specific
    `task` and `languages` keys are **optional**:

    * **`task`** — if absent, the task is inferred from Inspect AI hints: a solver
      with `name: multiple_choice` or a `field_spec.choices` entry both map to the
      `multiple-choice` task.  If the task cannot be inferred an error is logged and
      `None` is returned.
    * **`languages`** — if absent, the `fallback_language_codes` argument (a list
      of ISO 639-1 codes) is used.  When called from
      `try_get_dataset_config_from_repo`, the Hugging Face Hub repo metadata
      supplies this fallback automatically.  If neither source provides a language list,
      English (`"en"`) is used as the final fallback and a warning is logged.

    Column mappings may be specified either as flat top-level keys
    (`input_column` / `target_column` / `choices_column`) **or** via a
    `tasks[0].field_spec` block using the Inspect AI `input` / `target` /
    `choices` sub-keys.  Top-level keys take precedence when both are present.

    `tasks[0].split` is used as the test split, replacing the default `"test"`
    value.  `try_get_dataset_config_from_repo` still auto-detects the train and val
    splits from the repository, and also uses `tasks[0].config` as the HuggingFace
    dataset config/subset name when loading the dataset.

    When reading `field_spec`:

    * `field_spec.input` is used as `input_column`.
    * `field_spec.target` is used as `target_column` **only when it is a plain
      column name**.  Inspect AI also allows `"literal:<value>"` (a hard-coded
      answer string) and bare integers (which Inspect AI maps to letters A, B, C …);
      both are silently skipped because they are not column names.
    * `field_spec.choices` is used as `choices_column` (a single column name or a
      list of column names).

    Inspect AI silently ignores keys it does not recognise, so a single file can serve
    both frameworks.

    Example — EuroEval flat format:

        task: classification
        languages:
          - en
        labels:
          - positive
          - negative

    Example — pure Inspect AI format (task and languages are inferred automatically):

        # eval.yaml — no EuroEval-specific keys required
        name: My Dataset
        tasks:
          - id: my_dataset
            split: test
            field_spec:
              input: question
              target: answer
              choices: options
            solvers:
              - name: multiple_choice
            scorers:
              - name: choice

    Example — Inspect AI format with optional EuroEval overrides:

        # eval.yaml
        name: My Dataset
        tasks:
          - id: my_dataset
            split: test
            field_spec:
              input: text
              target: label
            solvers:
              - name: multiple_choice
            scorers:
              - name: choice
        # EuroEval-specific keys (optional; ignored by Inspect AI)
        task: multiple-choice
        languages:
          - en

    Supported task names are the `name` attributes of all task objects defined in
    `euroeval.tasks` (e.g. `classification`, `sentiment-classification`,
    `named-entity-recognition`, etc.).

    Args:
        yaml_path:
            Path to the YAML config file.
        fallback_language_codes:
            ISO 639-1 language codes to use when the YAML file does not contain a
            `languages` key.  Typically supplied from HuggingFace Hub repo metadata
            by `try_get_dataset_config_from_repo`.

    Returns:
        A `DatasetConfig` built from the YAML data, or `None` if the file could not
        be parsed or contains invalid values.
    """
    raw = load_yaml_file(yaml_path=yaml_path)
    if raw is None:
        return None

    promote_field_spec_fields(raw=raw)

    task_obj = validate_and_get_task(raw=raw, yaml_path=yaml_path)
    if task_obj is None:
        return None

    language_objs = parse_languages(
        raw=raw, fallback_codes=fallback_language_codes, yaml_path=yaml_path
    )
    if language_objs is None:
        return None

    kwargs = build_kwargs(raw=raw, yaml_path=yaml_path)
    if kwargs is None:
        return None

    return DatasetConfig(
        task=task_obj,
        languages=language_objs,
        **kwargs,  # pyrefly: ignore[bad-argument-type]
    )


def find_split(splits: list[str], keyword: str) -> str | None:
    """Return the shortest split name containing `keyword`, or `None`.

    Args:
        splits:
            A list of split names.
        keyword:
            The keyword to search for.

    Returns:
        The shortest split name containing `keyword`, or `None` if no such split exists.
    """
    candidates = sorted([s for s in splits if keyword in s.lower()], key=len)
    return candidates[0] if candidates else None


def get_repo_splits(
    hf_api: HfApi, dataset_id: str
) -> tuple[str | None, str | None, str | None]:
    """Return the (train, val, test) split names for a Hugging Face dataset repo.

    Args:
        hf_api:
            The Hugging Face API object.
        dataset_id:
            The ID of the dataset to get the split names for.

    Returns:
        A 3-type (train_split, val_split, test_split) where each element is either the
        name of the matching split or `None` if no such split exists.
    """
    splits = get_repo_split_names(hf_api=hf_api, dataset_id=dataset_id)
    return (
        find_split(splits=splits, keyword="train"),
        find_split(splits=splits, keyword="val"),
        find_split(splits=splits, keyword="test"),
    )


def get_repo_split_names(hf_api: HfApi, dataset_id: str) -> list[str]:
    """Extract split names from a Hugging Face dataset repo.

    Args:
        hf_api:
            The Hugging Face API object.
        dataset_id:
            The ID of the dataset to get the split names for.

    Returns:
        A list of split names.
    """
    return [
        split["name"]
        for split in hf_api.dataset_info(repo_id=dataset_id).card_data.dataset_info[
            "splits"
        ]
    ]


def try_get_dataset_config_from_repo(
    dataset_id: str,
    api_key: str | None,
    cache_dir: Path,
    trust_remote_code: bool,
    run_with_cli: bool,
) -> DatasetConfig | None:
    """Try to get a dataset config from a Hugging Face dataset repository.

    The function first looks for a YAML config file (`euroeval_config.yaml` or
    `eval.yaml`) which can be loaded without executing any remote code.  If neither
    YAML file is present the function falls back to `euroeval_config.py`, which
    requires `trust_remote_code=True`.

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
            (`euroeval_config.py`).  YAML configs never require this flag.
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

    # ------------------------------------------------------------------
    # 1. Try YAML configs first — no trust_remote_code needed.
    # ------------------------------------------------------------------
    if "eval.yaml" in repo_files:
        return load_yaml_config(hf_api, dataset_id, cache_dir)

    # ------------------------------------------------------------------
    # 2. Fall back to the Python config — requires trust_remote_code.
    # ------------------------------------------------------------------
    return load_python_config(
        hf_api=hf_api,
        dataset_id=dataset_id,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        run_with_cli=run_with_cli,
    )


def load_yaml_config(
    hf_api: HfApi, dataset_id: str, cache_dir: Path
) -> DatasetConfig | None:
    """Load a dataset config from an eval.yaml file in a Hugging Face repo.

    Args:
        hf_api:
            The Hugging Face API object.
        dataset_id:
            The ID of the dataset to get the config for.
        cache_dir:
            The directory to store the cache in.

    Returns:
        The dataset config if it exists, otherwise None.
    """
    external_config_path = cache_dir / "external_dataset_configs" / dataset_id
    external_config_path.mkdir(parents=True, exist_ok=True)
    hf_api.hf_hub_download(
        repo_id=dataset_id,
        repo_type="dataset",
        filename="eval.yaml",
        local_dir=external_config_path,
        local_dir_use_symlinks=False,
    )

    # Fetch repo metadata for language fallback and split detection
    repo_dataset_info = hf_api.dataset_info(repo_id=dataset_id)
    fallback_language_codes: list[str] | None = None
    if repo_dataset_info.card_data is not None:
        lang_meta = getattr(repo_dataset_info.card_data, "language", None)
        if isinstance(lang_meta, list) and lang_meta:
            fallback_language_codes = [str(c) for c in lang_meta if c]

    # Peek at the raw YAML to extract Inspect AI fields that need dataset-level context
    inspect_ai_config: str | None = None
    inspect_ai_split: str | None = None
    yaml_file_path = external_config_path / "eval.yaml"
    try:
        with yaml_file_path.open(encoding="utf-8") as fh:
            raw_peek = yaml.safe_load(fh)
        if isinstance(raw_peek, dict):
            tasks_peek = raw_peek.get("tasks")
            if isinstance(tasks_peek, list) and tasks_peek:
                first_task_peek = tasks_peek[0]
                if isinstance(first_task_peek, dict):
                    config_val = first_task_peek.get("config")
                    if isinstance(config_val, str) and config_val:
                        inspect_ai_config = config_val
                    split_val = first_task_peek.get("split")
                    if isinstance(split_val, str) and split_val:
                        inspect_ai_split = split_val
    except (yaml.YAMLError, OSError):
        pass  # errors will be handled inside load_dataset_config_from_yaml()

    repo_dataset_config = load_dataset_config_from_yaml(
        yaml_path=yaml_file_path, fallback_language_codes=fallback_language_codes
    )
    if repo_dataset_config is None:
        return None

    train_split, val_split, auto_test_split = get_repo_splits(hf_api, dataset_id)
    test_split = inspect_ai_split if inspect_ai_split is not None else auto_test_split
    if test_split is None:
        log_once(
            f"Dataset {dataset_id} does not have a test split, so we cannot load "
            "it. Please ensure that the dataset has a test split.",
            level=logging.ERROR,
        )
        return None

    # When there is no training split, promote the validation split to training
    if train_split is None and val_split is not None:
        log_once(
            f"Dataset {dataset_id!r} has no training split. Using the validation "
            f"split {val_split!r} as the training split instead.",
            level=logging.DEBUG,
        )
        train_split = val_split
        val_split = None

    source = f"{dataset_id}::{inspect_ai_config}" if inspect_ai_config else dataset_id

    repo_dataset_config.name = dataset_id
    repo_dataset_config.pretty_name = dataset_id
    repo_dataset_config.source = source
    repo_dataset_config.train_split = train_split
    repo_dataset_config.val_split = val_split
    repo_dataset_config.test_split = test_split
    return repo_dataset_config


def load_python_config(
    hf_api: HfApi,
    dataset_id: str,
    cache_dir: Path,
    trust_remote_code: bool,
    run_with_cli: bool,
) -> DatasetConfig | None:
    """Load a dataset config from an euroeval_config.py file in a Hugging Face repo.

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
            f"Dataset {dataset_id} does not have a euroeval_config.py or a YAML config "
            "file (euroeval_config.yaml / eval.yaml), so we cannot load it. Skipping.",
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
            f"The dataset {dataset_id} exists on the Hugging Face Hub and has a "
            "euroeval_config.py file, but remote code is not allowed. Please rerun "
            f"this with {rerunning_msg} if you trust the code in this repository.",
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
        return None  # Already warned the user in this case
    elif len(repo_dataset_configs) > 1:
        log_once(
            f"Dataset {dataset_id} has multiple dataset configurations. Please ensure "
            "that only a single DatasetConfig is defined in the `euroeval_config.py` "
            "file.",
            level=logging.WARNING,
        )
        return None

    train_split, val_split, test_split = get_repo_splits(hf_api, dataset_id)
    if test_split is None:
        log_once(
            f"Dataset {dataset_id} does not have a test split, so we cannot load it. "
            "Please ensure that the dataset has a test split.",
            level=logging.ERROR,
        )
        return None

    # When there is no training split, promote the validation split to training
    if train_split is None and val_split is not None:
        log_once(
            f"Dataset {dataset_id!r} has no training split. Using the validation "
            f"split {val_split!r} as the training split instead.",
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
