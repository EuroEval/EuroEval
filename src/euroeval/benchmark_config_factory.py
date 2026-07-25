"""Factory class for creating dataset configurations."""

import collections.abc as c
import logging
import sys
import typing as t
from pathlib import Path

import torch

from .closest_match import get_closest_match
from .data_models import BenchmarkConfig, BenchmarkConfigParams, DatasetConfig, Task
from .dataset_configs import get_all_dataset_configs
from .enums import Device
from .languages import get_all_languages, get_correct_language_codes
from .logging_utils import log

if t.TYPE_CHECKING:
    from .data_models import Language


def build_benchmark_config(
    benchmark_config_params: BenchmarkConfigParams,
) -> BenchmarkConfig:
    """Create a benchmark configuration.

    Args:
        benchmark_config_params:
            The parameters for creating the benchmark configuration.

    Returns:
        The benchmark configuration.
    """
    language_codes = get_correct_language_codes(
        language_codes=benchmark_config_params.language
    )
    languages = prepare_languages(
        language_codes=benchmark_config_params.language,
        default_language_codes=language_codes,
    )

    dataset_configs = prepare_dataset_configs(
        task=benchmark_config_params.task,
        dataset=benchmark_config_params.dataset,
        languages=languages,
        custom_datasets_file=benchmark_config_params.custom_datasets_file,
        api_key=benchmark_config_params.api_key,
        cache_dir=Path(benchmark_config_params.cache_dir),
        trust_remote_code=benchmark_config_params.trust_remote_code,
        run_with_cli=benchmark_config_params.run_with_cli,
    )

    return BenchmarkConfig(
        datasets=dataset_configs,
        languages=languages,
        finetuning_batch_size=benchmark_config_params.finetuning_batch_size,
        raise_errors=benchmark_config_params.raise_errors,
        cache_dir=benchmark_config_params.cache_dir,
        api_key=benchmark_config_params.api_key,
        force=benchmark_config_params.force,
        progress_bar=benchmark_config_params.progress_bar,
        save_results=benchmark_config_params.save_results,
        verbose=benchmark_config_params.verbose or benchmark_config_params.debug,
        device=prepare_device(device=benchmark_config_params.device),
        trust_remote_code=benchmark_config_params.trust_remote_code,
        clear_model_cache=benchmark_config_params.clear_model_cache,
        evaluate_test_split=benchmark_config_params.evaluate_test_split,
        few_shot=benchmark_config_params.few_shot,
        num_iterations=(
            1
            if hasattr(sys, "_called_from_test")
            else benchmark_config_params.num_iterations
        ),
        api_base=benchmark_config_params.api_base,
        api_version=benchmark_config_params.api_version,
        gpu_memory_utilization=benchmark_config_params.gpu_memory_utilization,
        attention_backend=benchmark_config_params.attention_backend,
        generative_type=benchmark_config_params.generative_type,
        use_bits_per_character=benchmark_config_params.use_bits_per_character,
        debug=benchmark_config_params.debug,
        run_with_cli=benchmark_config_params.run_with_cli,
        requires_safetensors=benchmark_config_params.requires_safetensors,
        download_only=benchmark_config_params.download_only,
        max_context_length=benchmark_config_params.max_context_length,
        vocabulary_size=benchmark_config_params.vocabulary_size,
    )


def prepare_languages(
    language_codes: str | c.Sequence[str] | None,
    default_language_codes: c.Sequence[str],
) -> c.Sequence["Language"]:
    """Prepare language(s) for benchmarking.

    Args:
        language_codes:
            The language codes of the languages to include for models or datasets.
            If specified then this overrides the `language` parameter for model or
            dataset languages.
        default_language_codes:
            The default language codes of the languages to include.

    Returns:
        The prepared dataset languages.
    """
    language_mapping = get_all_languages()

    languages_str: c.Sequence[str]
    if language_codes is None:
        languages_str = default_language_codes
    elif isinstance(language_codes, str):
        languages_str = [language_codes]
    else:
        languages_str = language_codes

    if "all" in languages_str:
        prepared_languages = list(language_mapping.values())
    else:
        prepared_languages = [language_mapping[language] for language in languages_str]

    return prepared_languages


def prepare_dataset_configs(
    task: "str | Task | c.Sequence[str | Task] | None",
    languages: c.Sequence["Language"],
    dataset: "str | DatasetConfig | c.Sequence[str | DatasetConfig] | None",
    custom_datasets_file: Path,
    api_key: str | None,
    cache_dir: Path,
    trust_remote_code: bool,
    run_with_cli: bool,
) -> list["DatasetConfig"]:
    """Prepare dataset config(s) for benchmarking.

    Args:
        task:
            The tasks to include for dataset. If None then datasets will not be
            filtered based on their task.
        languages:
            The languages of the datasets in the benchmark.
        dataset:
            The datasets to include for task. If None then all datasets will be
            included, limited by the `task` and `languages` parameters.
        custom_datasets_file:
            A path to a Python file containing custom dataset configurations.
        api_key:
            The API key to use for accessing the Hugging Face Hub.
        cache_dir:
            The directory to store the cache in.
        trust_remote_code:
            Whether to trust remote code.
        run_with_cli:
            Whether to run the benchmark with the CLI.

    Returns:
        The prepared dataset configs.
    """
    dataset_ids = _extract_dataset_ids(dataset=dataset)

    all_dataset_configs = get_all_dataset_configs(
        custom_datasets_file=custom_datasets_file,
        dataset_ids=dataset_ids,
        api_key=api_key,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        run_with_cli=run_with_cli,
    )
    all_official_dataset_configs: c.Sequence[DatasetConfig] = [
        dataset_config
        for dataset_config in all_dataset_configs.values()
        if not dataset_config.unofficial
    ]

    datasets = _get_datasets_list(
        dataset=dataset,
        all_dataset_configs=all_dataset_configs,
        all_official_dataset_configs=all_official_dataset_configs,
    )

    task_mapping = {cfg.task.name: cfg.task for cfg in all_dataset_configs.values()}
    tasks = _get_tasks_list(task=task, task_mapping=task_mapping)

    return [
        ds
        for ds in datasets
        if (tasks is None or ds.task in tasks)
        and any(lang in languages for lang in ds.languages)
    ]


def prepare_device(device: Device | None) -> torch.device:
    """Prepare device for benchmarking.

    Args:
        device:
            The device to use for running the models. If None then the device will be
            set automatically.

    Returns:
        The prepared device.
    """
    device_mapping = {
        Device.CPU: torch.device("cpu"),
        Device.CUDA: torch.device("cuda"),
        Device.MPS: torch.device("mps"),
    }
    if isinstance(device, Device):
        return device_mapping[device]

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _extract_dataset_ids(
    dataset: "str | DatasetConfig | c.Sequence[str | DatasetConfig] | None",
) -> list[str]:
    """Extract dataset IDs from the dataset argument.

    Args:
        dataset:
            The dataset argument to extract IDs from.

    Returns:
        List of dataset IDs.
    """
    dataset_ids: list[str] = list()
    if isinstance(dataset, str):
        dataset_ids.append(dataset)
    elif isinstance(dataset, DatasetConfig):
        dataset_ids.append(dataset.name)
    elif isinstance(dataset, list):
        for d in dataset:
            if isinstance(d, str):
                dataset_ids.append(d)
            elif isinstance(d, DatasetConfig):
                dataset_ids.append(d.name)
    return dataset_ids


def _get_datasets_list(
    dataset: "str | DatasetConfig | c.Sequence[str | DatasetConfig] | None",
    all_dataset_configs: dict[str, DatasetConfig],
    all_official_dataset_configs: c.Sequence[DatasetConfig],
) -> c.Sequence[DatasetConfig]:
    """Get the list of datasets based on the dataset argument.

    Args:
        dataset:
            The dataset argument specifying which datasets to include.
        all_dataset_configs:
            Mapping of dataset IDs to DatasetConfig objects.
        all_official_dataset_configs:
            List of official dataset configs.

    Returns:
        List of dataset configs.

    Raises:
        SystemExit: If dataset lookup fails.
    """
    try:
        if dataset is None:
            return all_official_dataset_configs
        elif isinstance(dataset, str):
            return [all_dataset_configs[dataset]]
        elif isinstance(dataset, DatasetConfig):
            return [dataset]
        else:
            return [
                all_dataset_configs[d] if isinstance(d, str) else d for d in dataset
            ]
    except KeyError as e:
        _handle_dataset_lookup_error(
            error=e, options=list(all_dataset_configs.keys()), entity_type="dataset"
        )
        # Unreachable: _handle_dataset_lookup_error calls sys.exit(1)
        raise SystemExit(1)


def _get_tasks_list(
    task: "str | Task | c.Sequence[str | Task] | None", task_mapping: dict[str, Task]
) -> list[Task] | None:
    """Get the list of tasks based on the task argument.

    Args:
        task:
            The task argument specifying which tasks to include.
        task_mapping:
            Mapping of task names to Task objects.

    Returns:
        List of tasks, or None if no task filtering.
    """
    try:
        if task is None:
            return None
        elif isinstance(task, str):
            return [task_mapping[task]]
        elif isinstance(task, Task):
            return [task]
        else:
            return [task_mapping[t] if isinstance(t, str) else t for t in task]
    except KeyError as e:
        _handle_dataset_lookup_error(
            error=e, options=list(task_mapping.keys()), entity_type="task"
        )


def _handle_dataset_lookup_error(
    error: KeyError, options: list[str], entity_type: str
) -> None:
    """Handle a KeyError during dataset or task lookup.

    Args:
        error:
            The KeyError that was raised.
        options:
            List of valid options to search for closest match.
        entity_type:
            Type of entity ("dataset" or "task") for error message.
    """
    closest_match, closest_distance = get_closest_match(
        string=error.args[0], options=options, case_sensitive=False
    )
    msg = f"The {entity_type} {error} was not found."
    if closest_distance < 5:
        msg += f" Maybe you meant to use {closest_match!r}?"
    log(msg, level=logging.ERROR)
    sys.exit(1)
