"""Load dataset configurations from YAML files.

This module handles all YAML-related functionality for loading dataset
configurations, including Inspect AI-compatible `eval.yaml` files from
Hugging Face Hub repositories.
"""

import dataclasses
import logging
import string
from pathlib import Path
from typing import cast

import yaml
from huggingface_hub import HfApi

from .data_models import DatasetConfig, Task
from .languages import Language, get_all_languages, get_language, is_language_code
from .logging_utils import log_once
from .metrics.llm_as_a_judge import create_model_graded_fact_metric
from .split_utils import get_repo_splits
from .tasks import REFERENCE_FREE_QA, get_all_tasks


def load_yaml_config(
    hf_api: HfApi, dataset_id: str, cache_dir: Path
) -> list[DatasetConfig] | None:
    """Load dataset configs from an eval.yaml file in a Hugging Face repo.

    Each task entry declared by the `eval.yaml` becomes its own dataset config, so a
    repository declaring a configuration per language expands into one dataset per
    language.

    Args:
        hf_api:
            The Hugging Face API object.
        dataset_id:
            The ID of the dataset to get the configs for, optionally suffixed by the
            split to select, e.g. `repo::test`.
        cache_dir:
            The directory to store the cache in.

    Returns:
        The dataset configs, or None if the repository has no loadable eval.yaml or
        the selector does not refer to any task entry.
    """
    parsed_selector = parse_dataset_selector(dataset_id=dataset_id)
    if parsed_selector is None:
        return None
    repo_id, subset_config, subset_split = parsed_selector

    external_config_path = cache_dir / "external_dataset_configs" / repo_id
    external_config_path.mkdir(parents=True, exist_ok=True)
    hf_api.hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="eval.yaml",
        local_dir=external_config_path,
        local_dir_use_symlinks=False,
    )

    repo_dataset_info = hf_api.dataset_info(repo_id=repo_id)
    fallback_language_codes: list[str] | None = None
    if repo_dataset_info.card_data is not None:
        lang_meta = getattr(repo_dataset_info.card_data, "language", None)
        if isinstance(lang_meta, list) and lang_meta:
            fallback_language_codes = [str(c) for c in lang_meta if c]

    yaml_file_path = external_config_path / "eval.yaml"
    try:
        with yaml_file_path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except (yaml.YAMLError, OSError):
        raw = None
    if not isinstance(raw, dict):
        # Re-read through the shared loader so parse and top-level-shape errors are
        # reported consistently instead of being swallowed here.
        load_dataset_config_from_yaml(
            yaml_path=yaml_file_path,
            fallback_language_codes=fallback_language_codes,
            task_index=0,
        )
        return None

    tasks_raw = raw.get("tasks")
    declared_configs = sorted(
        {
            str(task["config"])
            for task in (tasks_raw if isinstance(tasks_raw, list) else [])
            if isinstance(task, dict) and task.get("config")
        }
    )
    config_languages = resolve_config_languages(configs=declared_configs)

    selected = select_inspect_ai_tasks(
        raw=raw,
        subset_split=subset_split,
        dataset_id=dataset_id,
        subset_config=subset_config,
    )
    if selected is None:
        return None

    dataset_configs: list[DatasetConfig] = []
    for task_index, inspect_ai_config, inspect_ai_split in selected:
        task_entry = tasks_raw[task_index] if isinstance(tasks_raw, list) else {}
        language_selection_supplied = "languages" in raw or (
            isinstance(task_entry, dict) and "languages" in task_entry
        )
        entry_fallback_codes = fallback_language_codes
        if inspect_ai_config is not None and not language_selection_supplied:
            language = (
                config_languages.get(inspect_ai_config) if config_languages else None
            )
            if language is not None:
                entry_fallback_codes = [language.code]
            elif config_languages is not None and is_language_code(
                inspect_ai_config.partition("_")[0]
            ):
                log_once(
                    message=(
                        f"Config {inspect_ai_config!r} in dataset {repo_id!r} refers "
                        "to a language that EuroEval does not support, so it is not "
                        "benchmark."
                    ),
                    level=logging.WARNING,
                )
                continue
            elif (
                fallback_language_codes is not None and len(fallback_language_codes) > 1
            ):
                log_once(
                    message=(
                        f"The language of subset {inspect_ai_config!r} in dataset "
                        f"{repo_id!r} could not be determined. Results are attributed "
                        f"to all {len(fallback_language_codes)} languages of the "
                        "repository; add a per-entry `languages` key to the "
                        "`eval.yaml`."
                    ),
                    level=logging.WARNING,
                )

        repo_dataset_config = load_dataset_config_from_yaml(
            yaml_path=yaml_file_path,
            fallback_language_codes=entry_fallback_codes,
            task_index=task_index,
        )
        if repo_dataset_config is None:
            return None

        train_split, val_split, auto_test_split = get_repo_splits(
            hf_api=hf_api, dataset_id=repo_id, config_name=inspect_ai_config
        )
        test_split = (
            inspect_ai_split if inspect_ai_split is not None else auto_test_split
        )
        if test_split is None:
            log_once(
                message=(
                    f"Dataset {dataset_id} does not have a test split, so we cannot "
                    "load it. Please ensure that the dataset has a test split."
                ),
                level=logging.ERROR,
            )
            return None

        if train_split is None and val_split is not None:
            log_once(
                message=(
                    f"Dataset {dataset_id!r} has no training split. Using the "
                    "validation split "
                    f"{val_split!r} as the training split instead."
                ),
                level=logging.DEBUG,
            )
            train_split = val_split
            val_split = None

        if inspect_ai_config is None:
            name = repo_id
            source = repo_id
        else:
            name = f"{repo_id}::{inspect_ai_config}"
            if inspect_ai_split is not None:
                name = f"{name}::{inspect_ai_split}"
            source = f"{repo_id}::{inspect_ai_config}"

        repo_dataset_config.name = name
        repo_dataset_config.pretty_name = name
        repo_dataset_config.source = source
        repo_dataset_config.train_split = train_split
        repo_dataset_config.val_split = val_split
        repo_dataset_config.test_split = test_split
        dataset_configs.append(repo_dataset_config)

    if not dataset_configs:
        return None
    return dataset_configs


def load_dataset_config_from_yaml(
    yaml_path: Path,
    fallback_language_codes: list[str] | None = None,
    task_index: int = 0,
) -> DatasetConfig | None:
    """Load a dataset config from a YAML file.

    The file is fully compatible with the Inspect AI `eval.yaml` format
    (https://inspect.aisi.org.uk/tasks.html#hugging-face). The EuroEval-specific
    `task` and `languages` keys are optional:

    * `task` -- if absent, the task is inferred from Inspect AI hints: a solver
      with `name: multiple_choice` or a `field_spec.choices` entry both map to the
      `multiple-choice` task, while a `math` scorer maps to the `math` task. If the
      task cannot be inferred an error is logged and None is returned.
    * `languages` -- if absent, the `fallback_language_codes` argument (a list
      of ISO 639-1 codes) is used. When called from
      `try_get_dataset_config_from_repo`, the Hugging Face Hub repo metadata
      supplies this fallback automatically. If neither source provides a language
      list, English (`"en"`) is used as the final fallback and a warning is logged.

    Column mappings may be specified either as flat top-level keys
    (`input_column` / `target_column` / `choices_column`) or via the selected task's
    `field_spec` block using the Inspect AI `input` / `target` /
    `choices` sub-keys. Top-level keys take precedence when both are present.

    The selected task's `split` is used as the test split. A dataset selector uses
    `repo::config[::split]`; `try_get_dataset_config_from_repo` auto-detects the train
    and val splits from the repository, and uses the selected task's `config` as the
    HuggingFace dataset config/subset name.

    When reading `field_spec`:

    * `field_spec.input` is used as `input_column`.
    * `field_spec.target` is used as `target_column` only when it is a plain
      column name. Inspect AI also allows `"literal:<value>"` (a hard-coded
      answer string) and bare integers (which Inspect AI maps to letters A, B, C
      ...); both are silently skipped because they are not column names.
    * `field_spec.choices` is used as `choices_column` (a single column name or
      a list of column names).

    Example -- EuroEval flat format:

        task: classification
        languages:
          - en
        labels:
          - positive
          - negative

    Example -- pure Inspect AI format (task and languages are inferred automatically):

        # eval.yaml -- no EuroEval-specific keys required
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

    Example -- Inspect AI format with optional EuroEval overrides:

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

    Args:
        yaml_path:
            Path to the YAML config file.
        fallback_language_codes:
            ISO 639-1 language codes to use when the YAML file does not contain a
            `languages` key. Typically supplied from HuggingFace Hub repo metadata
            by `try_get_dataset_config_from_repo`.
        task_index (optional):
            The Inspect AI task entry to load. Defaults to 0.

    Returns:
        A `DatasetConfig` built from the YAML data, or None if the file could not
            be parsed or contains invalid values.
    """
    raw = load_yaml_file(yaml_path=yaml_path)
    if raw is None:
        return None

    tasks_raw = raw.get("tasks")
    if isinstance(tasks_raw, list) and 0 <= task_index < len(tasks_raw):
        selected_task = tasks_raw[task_index]
        if isinstance(selected_task, dict) and "languages" in selected_task:
            raw["languages"] = selected_task["languages"]
    promote_field_spec_fields(raw=raw, task_index=task_index)

    task_obj = validate_and_get_task(
        raw=raw, yaml_path=yaml_path, task_index=task_index
    )
    if task_obj is None:
        return None
    promote_inspect_ai_prompt_template(raw=raw, task=task_obj, task_index=task_index)

    prompt = str(raw.get("instruction_prompt", ""))
    if task_obj.name == "math" and not any(marker in prompt for marker in _BOX_MARKERS):
        log_once(
            message=(
                "The math task expects the model to answer in \\boxed{...}, but the "
                "instruction prompt does not ask it to. Add a `prompt_template` "
                "solver, or a top-level `instruction_prompt` key, telling the model "
                "to put its final answer in `\\boxed{...}`."
            ),
            level=logging.WARNING,
        )

    language_objs = parse_languages(
        raw=raw, fallback_codes=fallback_language_codes, yaml_path=yaml_path
    )
    if language_objs is None:
        return None

    kwargs = build_kwargs(raw=raw, yaml_path=yaml_path)
    if kwargs is None:
        return None

    kwargs.setdefault("test_split", "test")
    kwargs.setdefault("bootstrap_samples", True)
    kwargs.setdefault("unofficial", False)
    kwargs.setdefault("input_column", "text")
    return DatasetConfig(task=task_obj, languages=language_objs, **kwargs)  # ty: ignore[invalid-argument-type]


def parse_languages(
    raw: dict[str, object], fallback_codes: list[str] | None, yaml_path: Path
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
        from_repo_metadata = False
    elif fallback_codes:
        log_once(
            message=(
                f"YAML config at {yaml_path} does not contain a 'languages' key. "
                "Using language(s) from the repository metadata: "
                f"{fallback_codes}."
            ),
            level=logging.DEBUG,
        )
        language_codes = fallback_codes
        from_repo_metadata = True
    else:
        log_once(
            message=(
                f"YAML config at {yaml_path} does not contain a 'languages' key and "
                "no language metadata could be found for this repository. Defaulting "
                "to English. Add a top-level 'languages' key to the YAML file "
                "(e.g. 'languages: [en]') to override this."
            ),
            level=logging.WARNING,
        )
        language_codes = ["en"]
        from_repo_metadata = False

    language_objs: list[Language] = []
    for code in language_codes:
        lang = language_map.get(code)
        if lang is None:
            if from_repo_metadata:
                # The Hub card lists the languages covered by the underlying data,
                # not the EuroEval-supported ones, so an unsupported code here is
                # not a problem with the configuration itself
                log_once(
                    message=(
                        f"Language code '{code}' from the repository metadata is not "
                        "supported by EuroEval, so it is ignored (YAML config at "
                        f"{yaml_path})."
                    ),
                    level=logging.DEBUG,
                )
                continue
            log_once(
                message=(
                    f"Unknown language code '{code}' in YAML config at {yaml_path}."
                ),
                level=logging.ERROR,
            )
            return None
        language_objs.append(lang)

    if not language_objs:
        log_once(
            message=(
                f"None of the language codes {language_codes} from the repository "
                f"metadata are supported by EuroEval, so we cannot determine the "
                f"languages of the YAML config at {yaml_path}. Please add a top-level "
                "'languages' key to the YAML file (e.g. 'languages: [en]')."
            ),
            level=logging.ERROR,
        )
        return None

    return language_objs


_BOX_MARKERS: tuple[str, ...] = ("boxed", "fbox")
"""Substrings an instruction prompt may use to ask for a boxed answer.

`boxed` covers `\boxed{...}`, `\fbox` is spelled without it, and `beginboxed` contains
`boxed`; these are the spellings `euroeval.metrics.math` extracts an answer from.
"""


def parse_dataset_selector(
    dataset_id: str,
) -> tuple[str, str | None, str | None] | None:
    """Parse a dataset selector and report malformed selectors.

    A selector either names one of the expanded subsets directly, `repo::config::split`,
    or narrows the repository down to one of its splits, `repo::split`. A configuration
    is not selected on its own: it is commonly named after the language it contains,
    which reads as a language code rather than a split, so naming one is reported as an
    unknown split that lists the subsets it stands for.

    Args:
        dataset_id:
            The requested dataset ID.

    Returns:
        The repository ID, the configuration of a named subset, and the requested split,
        or None if the selector is malformed.
    """
    parts = dataset_id.split("::")
    if len(parts) > 3 or any(not part for part in parts):
        log_once(
            message=(
                f"Invalid dataset selector {dataset_id!r}. Use the syntax "
                "<repo>[::<split>] to select a split, or the name of one of the "
                "expanded subsets, <repo>::<config>::<split>."
            ),
            level=logging.ERROR,
        )
        return None
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return parts[0], None, parts[1] if len(parts) > 1 else None


def promote_field_spec_fields(raw: dict[str, object], task_index: int = 0) -> None:
    """Promote column names from field_spec to top-level keys.

    Promotes the following mappings when the top-level key is not already set:

    * `field_spec.input` -> `input_column`
    * `field_spec.target` -> `target_column` (only if plain, not literal/int)
    * `field_spec.choices` -> `choices_column`
    * `tasks[task_index].split` -> `test_split`

    Prompt templates are promoted separately by `promote_inspect_ai_prompt_template`.

    Args:
        raw:
            The parsed YAML data to modify in place.
        task_index (optional):
            The Inspect AI task entry to use. Defaults to 0.
    """
    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        return

    if not 0 <= task_index < len(tasks_raw):
        return
    first_task: dict[str, object] = cast(dict[str, object], tasks_raw[task_index])
    if not isinstance(first_task, dict):
        return

    field_spec = first_task.get("field_spec")
    if isinstance(field_spec, dict):
        _fs: dict[str, object] = cast(dict[str, object], field_spec)
        if "input" in _fs and "input_column" not in raw:
            raw["input_column"] = _fs["input"]

        if "target" in _fs and "target_column" not in raw:
            target = _fs["target"]
            if isinstance(target, str) and not target.startswith("literal:"):
                raw["target_column"] = target

        if "choices" in _fs and "choices_column" not in raw:
            raw["choices_column"] = _fs["choices"]

    split_val = first_task.get("split")
    if isinstance(split_val, str) and split_val and "test_split" not in raw:
        raw["test_split"] = split_val


def promote_inspect_ai_prompt_template(
    raw: dict[str, object], task: Task, task_index: int = 0
) -> None:
    r"""Promote an Inspect AI prompt template to EuroEval's instruction prompt.

    Inspect AI's `{prompt}` placeholder is replaced by EuroEval's `{text}`
    placeholder, and a template without the placeholder gets `\\n\\n{text}`
    appended so that the input is still included. Otherwise the replacement is
    deliberately literal: doubled braces such as `\\boxed{{}}` in the Inspect AI
    template are Python format escapes and must remain doubled until EuroEval
    formats the prompt for a sample. An explicit `instruction_prompt` key in the
    YAML file takes precedence over the solver template.

    Args:
        raw:
            The parsed YAML data to modify in place.
        task:
            The resolved EuroEval task.
        task_index (optional):
            The Inspect AI task entry to use. Defaults to 0.
    """
    if "instruction_prompt" in raw or task.uses_logprobs:
        return

    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        return
    if not 0 <= task_index < len(tasks_raw):
        return
    first_task = tasks_raw[task_index]
    if not isinstance(first_task, dict):
        return

    solvers = first_task.get("solvers")
    if not isinstance(solvers, list):
        return
    for solver in solvers:
        if not isinstance(solver, dict) or solver.get("name") != "prompt_template":
            continue
        args = solver.get("args")
        if not isinstance(args, dict):
            continue
        template = args.get("template")
        if not isinstance(template, str):
            continue
        substituted = (
            template.replace("{prompt}", "{text}")
            if "{prompt}" in template
            else f"{template}\n\n{{text}}"
        )
        unsupported = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(substituted)
            if field_name not in (None, "", "text")
        }
        if unsupported:
            log_once(
                message=(
                    "Inspect AI prompt template ignored because it uses unsupported "
                    f"placeholders: {sorted(unsupported)}."
                ),
                level=logging.DEBUG,
            )
            return
        raw["instruction_prompt"] = substituted
        return


def resolve_config_languages(configs: list[str]) -> dict[str, Language] | None:
    """Map dataset configurations onto languages, if they name languages.

    Dataset configurations are commonly named after the language they contain, using
    either the ISO 639-1 or the ISO 639-3 code, and may carry a further suffix such as
    `eng_metric`. Configuration names are only trusted when most of them resolve to a
    language, as otherwise configurations named e.g. `default` or `train` would be
    mistaken for language codes.

    Args:
        configs:
            The configurations declared by a dataset, in arbitrary order.

    Returns:
        A mapping from configuration name to language, or None when the names do not
        look like languages. A configuration that looks like a language but is not
        supported by EuroEval is missing from the mapping.
    """
    resolved = {config: get_language(config.partition("_")[0]) for config in configs}
    named = [config for config, language in resolved.items() if language is not None]
    if 2 * len(named) < len(configs):
        return None
    return {config: language for config, language in resolved.items() if language}


def select_inspect_ai_tasks(
    raw: dict[str, object],
    subset_split: str | None,
    *,
    dataset_id: str,
    subset_config: str | None = None,
) -> list[tuple[int, str | None, str | None]] | None:
    """Select the Inspect AI task entries a dataset selector refers to.

    Every task entry declared by the `eval.yaml` is a dataset of its own, so with no
    selector all of them are selected. Since configurations are commonly named after
    the language they contain, they are not told apart by the
    dataset ID, which leaves `::<split>` to select a single split across all the
    configurations.

    Args:
        raw:
            The parsed YAML data.
        subset_split:
            The requested split, if any.
        subset_config:
            The configuration of a directly named subset, if the full subset name was
            requested. Not usable on its own, as configurations are language names
            named in the selector as `config::split`.
        dataset_id:
            The requested dataset ID, used for error messages.

    Returns:
        A list of task entry indices, configurations and splits, or None if the
        selector does not refer to any entry.
    """
    tasks = raw.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        if subset_split is not None or subset_config is not None:
            log_once(
                message=(
                    f"Dataset {dataset_id!r} eval.yaml declares no task entries, so "
                    "split selection is unsupported."
                ),
                level=logging.ERROR,
            )
            return None
        return [(0, None, None)]
    entries = [
        (index, task) for index, task in enumerate(tasks) if isinstance(task, dict)
    ]
    if not any(task.get("config") for _, task in entries):
        # Entries without configurations are not subsets, so only the first one is
        # used, as it was before the subsets existed
        if subset_split is not None or subset_config is not None:
            log_once(
                message=(
                    f"Dataset {dataset_id!r} eval.yaml declares no configurations, so "
                    "subset selection is unsupported."
                ),
                level=logging.ERROR,
            )
            return None
        first = entries[0][1]
        return [
            (entries[0][0], None, str(first["split"]) if first.get("split") else None)
        ]
    if subset_config is not None:
        configs = sorted(
            {str(task["config"]) for _, task in entries if task.get("config")}
        )
        entries = [
            (index, task)
            for index, task in entries
            if str(task.get("config")) == subset_config
        ]
        if not entries:
            log_once(
                message=(
                    f"Unknown config {subset_config!r} for dataset {dataset_id!r}. "
                    f"Available configs are: {configs}."
                ),
                level=logging.ERROR,
            )
            return None
    splits = sorted({str(task["split"]) for _, task in entries if task.get("split")})
    if subset_split is not None:
        if subset_split not in splits:
            configs = sorted(
                {str(task["config"]) for _, task in entries if task.get("config")}
            )
            message = (
                f"Unknown split {subset_split!r} for dataset {dataset_id!r}. "
                f"Available splits are: {splits}."
            )
            if subset_split in configs:
                subsets = sorted(
                    f"{dataset_id.partition('::')[0]}::{subset_split}::{task.get('split')}"
                    for _, task in entries
                    if task.get("config") == subset_split and task.get("split")
                )
                message += (
                    f" {subset_split!r} is a configuration, not a split; name one of "
                    f"its subsets instead{': ' if subsets else '.'}"
                    f"{', '.join(subsets)}"
                )
            log_once(message=message, level=logging.ERROR)
            return None
        entries = [
            (index, task)
            for index, task in entries
            if str(task.get("split")) == subset_split
        ]
    if not entries:
        log_once(
            message=(
                f"No task entry matches split {subset_split!r} in dataset "
                f"{dataset_id!r}."
            ),
            level=logging.ERROR,
        )
        return None
    return [
        (
            index,
            str(task["config"]) if task.get("config") else None,
            str(task["split"]) if task.get("split") else None,
        )
        for index, task in entries
    ]


def validate_and_get_task(
    raw: dict[str, object], yaml_path: Path, task_index: int = 0
) -> Task | None:
    """Validate the task field or infer it from Inspect AI hints.

    Args:
        raw:
            The parsed YAML data.
        yaml_path:
            Path to the YAML config file (for error messages).
        task_index (optional):
            The Inspect AI task entry to inspect. Defaults to 0.

    Returns:
        A valid Task object, or None if validation failed.
    """
    task_map = get_all_tasks()
    task_name = raw.get("task")

    if isinstance(task_name, str):
        task_obj = task_map.get(task_name)
        if task_obj is None:
            log_once(
                message=(
                    f"Unknown task '{task_name}' in YAML config at {yaml_path}. "
                    f"Valid task names are: {sorted(task_map)}."
                ),
                level=logging.ERROR,
            )
            return None
    else:
        task_obj = infer_task_from_inspect_ai(
            raw=raw, task_map=task_map, task_index=task_index
        )
        if task_obj is None:
            log_once(
                message=(
                    f"YAML config at {yaml_path} does not contain a 'task' field and "
                    "the task could not be inferred from the Inspect AI 'tasks' block. "
                    "Add a top-level 'task' key (e.g. 'task: classification') or "
                    "include a 'multiple_choice' solver / a 'choices' field_spec entry "
                    "so that the task can be detected automatically."
                ),
                level=logging.ERROR,
            )
            return None

    return task_obj


def infer_task_from_inspect_ai(
    raw: dict[str, object], task_map: dict[str, Task], task_index: int = 0
) -> Task | None:
    """Try to infer the EuroEval task from Inspect AI YAML fields.

    Currently detects:

    * A solver with `name: multiple_choice` in `tasks[0].solvers`
      -> `multiple-choice`
    * A `choices` key in `tasks[0].field_spec` -> `multiple-choice`
    * A scorer with `name: math` in `tasks[0].scorers` -> `math`
    * A scorer with `name: model_graded_fact` in `tasks[0].scorers`
      -> `reference-free-qa` task with an LLM-as-a-judge metric.
      The judge model is read from `scorers[0].args.model`; when absent, the
      default judge defined in `REFERENCE_FREE_QA` is used.

    Prompt templates are not handled here; see `promote_inspect_ai_prompt_template`.

    Args:
        raw:
            The raw YAML data.
        task_map:
            The mapping from task names to task objects.
        task_index (optional):
            The Inspect AI task entry to inspect. Defaults to 0.

    Returns:
        The inferred task, or None if the task cannot be inferred.
    """
    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        return None
    if not 0 <= task_index < len(tasks_raw):
        return None
    first_task: dict[str, object] = cast(dict[str, object], tasks_raw[task_index])
    if not isinstance(first_task, dict):
        return None

    solvers = first_task.get("solvers")
    if isinstance(solvers, list):
        for solver in solvers:
            if isinstance(solver, dict):
                _s: dict[str, object] = cast(dict[str, object], solver)
                if _s.get("name") == "multiple_choice":
                    return task_map.get("multiple-choice")

    scorers = first_task.get("scorers")
    if isinstance(scorers, list):
        for scorer in scorers:
            if isinstance(scorer, dict):
                _sc: dict[str, object] = cast(dict[str, object], scorer)
                if _sc.get("name") == "math":
                    return task_map.get("math")
                if _sc.get("name") == "model_graded_fact":
                    judge_id: str | None = None
                    args = _sc.get("args") or {}
                    if isinstance(args, dict):
                        model_val = args.get("model")
                        if isinstance(model_val, str) and model_val:
                            judge_id = model_val
                    if judge_id is not None:
                        metric = create_model_graded_fact_metric(judge_id=judge_id)
                        return dataclasses.replace(REFERENCE_FREE_QA, metrics=[metric])
                    return REFERENCE_FREE_QA

    field_spec = first_task.get("field_spec")
    if isinstance(field_spec, dict) and "choices" in field_spec:
        return task_map.get("multiple-choice")

    return None


DatasetKwargs = dict[str, str | int | bool | list[str] | dict[str, str]]


def build_kwargs(raw: dict[str, object], yaml_path: Path) -> DatasetKwargs | None:
    """Build keyword arguments for `DatasetConfig` from YAML fields.

    Reads the following optional fields from `raw` and maps them to the
    corresponding `DatasetConfig` constructor arguments:

    * String fields: `prompt_prefix`, `prompt_template`, `instruction_prompt`,
      `input_column`, `target_column`, `test_split`.
    * Integer fields: `num_few_shot_examples`, `max_generated_tokens`.
    * `labels` -- a list of strings.
    * `prompt_label_mapping` -- a mapping from strings to strings.
    * `choices_column` -- a string or list of strings.

    Args:
        raw:
            The parsed YAML data.
        yaml_path:
            Path to the YAML config file (for error messages).

    Returns:
        A dictionary suitable for unpacking into `DatasetConfig(...)`, or None
            if any field fails validation.
    """
    kwargs: DatasetKwargs = {}

    for field_name in (
        "prompt_prefix",
        "prompt_template",
        "instruction_prompt",
        "input_column",
        "target_column",
        "test_split",
    ):
        value = parse_string_field(raw=raw, field_name=field_name, yaml_path=yaml_path)
        if value is not None:
            kwargs[field_name] = value

    for field_name in ("num_few_shot_examples", "max_generated_tokens"):
        value = parse_int_field(raw=raw, field_name=field_name, yaml_path=yaml_path)
        if value is not None:
            kwargs[field_name] = value
        elif raw.get(field_name) is not None:
            return None

    labels_raw = raw.get("labels")
    if labels_raw is not None:
        if not isinstance(labels_raw, list):
            log_once(
                message=f"Field 'labels' in YAML config at {yaml_path} must be a list.",
                level=logging.ERROR,
            )
            return None
        kwargs["labels"] = [str(lbl) for lbl in labels_raw]

    prompt_label_mapping_raw = raw.get("prompt_label_mapping")
    if prompt_label_mapping_raw is not None:
        if not isinstance(prompt_label_mapping_raw, dict):
            log_once(
                message=(
                    "Field 'prompt_label_mapping' in YAML config at"
                    f" {yaml_path} must be a mapping."
                ),
                level=logging.ERROR,
            )
            return None
        kwargs["prompt_label_mapping"] = {
            str(k): str(v) for k, v in prompt_label_mapping_raw.items()
        }

    choices_column_raw = raw.get("choices_column")
    if choices_column_raw is not None:
        if isinstance(choices_column_raw, list):
            kwargs["choices_column"] = [str(c) for c in choices_column_raw]
        elif isinstance(choices_column_raw, str):
            kwargs["choices_column"] = choices_column_raw
        else:
            log_once(
                message=(
                    "Field 'choices_column' in YAML config at"
                    f" {yaml_path} must be a string or a list of strings."
                ),
                level=logging.ERROR,
            )
            return None

    return kwargs


def parse_int_field(
    raw: dict[str, object], field_name: str, yaml_path: Path
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
                message=(
                    f"Field '{field_name}' in YAML config at {yaml_path} must be an "
                    "integer."
                ),
                level=logging.ERROR,
            )
            return None
        return value
    return None


def parse_string_field(
    raw: dict[str, object], field_name: str, yaml_path: Path
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
                message=(
                    f"Field '{field_name}' in YAML config at {yaml_path} must be a "
                    "string."
                ),
                level=logging.ERROR,
            )
            return None
        return value
    return None


def load_yaml_file(yaml_path: Path) -> dict[str, object] | None:
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
            message=f"Could not parse YAML config from {yaml_path}: {exc}",
            level=logging.ERROR,
        )
        return None

    if not isinstance(raw, dict):
        log_once(
            message=f"YAML config at {yaml_path} must be a mapping at the top level.",
            level=logging.ERROR,
        )
        return None

    return raw
