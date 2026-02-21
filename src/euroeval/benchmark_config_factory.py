"""Factory class for creating dataset configurations."""

import collections.abc as c
import importlib.util
import logging
import sys
import typing as t
from pathlib import Path

import torch

from .closest_match import get_closest_match
from .data_models import BenchmarkConfig, BenchmarkConfigParams, DatasetConfig, Task
from .dataset_configs import get_all_dataset_configs
from .enums import Device
from .languages import get_all_languages
from .logging_utils import log

if importlib.util.find_spec("vllm") is not None:
    pass

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
        debug=benchmark_config_params.debug,
        run_with_cli=benchmark_config_params.run_with_cli,
        requires_safetensors=benchmark_config_params.requires_safetensors,
        download_only=benchmark_config_params.download_only,
    )


def get_correct_language_codes(
    language_codes: str | c.Sequence[str],
) -> c.Sequence[str]:
    """Get correct language code(s).

    Args:
        language_codes:
            The language codes of the languages to include, both for models and
            datasets. Here 'no' means both Bokmål (nb) and Nynorsk (nn). Set this
            to 'all' if all languages should be considered.

    Returns:
        The correct language codes.
    """
    # Create a dictionary that maps languages to their associated language objects
    language_mapping = get_all_languages()

    # Create the list `languages`
    if "all" in language_codes:
        languages = list(language_mapping.keys())
    elif isinstance(language_codes, str):
        languages = [language_codes]
    else:
        languages = list(language_codes)

    # If `languages` contains 'no' then also include 'nb' and 'nn'. Conversely, if
    # either 'nb' or 'nn' are specified then also include 'no'.
    if "no" in languages:
        languages = list(set(languages) | {"nb", "nn"})
    elif "nb" in languages or "nn" in languages:
        languages = list(set(languages) | {"no"})

    return languages


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
    # Create a dictionary that maps languages to their associated language objects
    language_mapping = get_all_languages()

    # Create the list `languages_str` of language codes to use for models or datasets
    languages_str: c.Sequence[str]
    if language_codes is None:
        languages_str = default_language_codes
    elif isinstance(language_codes, str):
        languages_str = [language_codes]
    else:
        languages_str = language_codes

    # Convert the model languages to language objects
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
    # Extract the dataset IDs from the `dataset` argument
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

    # Create the list of dataset configs
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
    try:
        if dataset is None:
            datasets = all_official_dataset_configs
        elif isinstance(dataset, str):
            datasets = [all_dataset_configs[dataset]]
        elif isinstance(dataset, DatasetConfig):
            datasets = [dataset]
        else:
            datasets = [
                all_dataset_configs[d] if isinstance(d, str) else d for d in dataset
            ]
    except KeyError as e:
        closest_match, closest_distance = get_closest_match(
            string=e.args[0],
            options=list(all_dataset_configs.keys()),
            case_sensitive=False,
        )
        msg = f"The dataset {e} was not found in the benchmark datasets."
        if closest_distance < 5:
            msg += f" Maybe you meant to use {closest_match!r}?"
        log(msg, level=logging.ERROR)
        sys.exit(1)

    # Create the list of dataset tasks
    task_mapping = {cfg.task.name: cfg.task for cfg in all_dataset_configs.values()}
    try:
        if task is None:
            tasks = None
        elif isinstance(task, str):
            tasks = [task_mapping[task]]
        elif isinstance(task, Task):
            tasks = [task]
        else:
            tasks = [task_mapping[t] if isinstance(t, str) else t for t in task]
    except KeyError as e:
        closest_match, closest_distance = get_closest_match(
            string=e.args[0], options=list(task_mapping.keys()), case_sensitive=False
        )
        msg = f"Task {e} not found in the benchmark tasks."
        if closest_distance < 5:
            msg += f" Maybe you meant to use {closest_match!r}?"
        log(msg, level=logging.ERROR)
        sys.exit(1)

    # Filter the dataset configs based on the specified tasks and languages
    datasets = [
        ds
        for ds in datasets
        if (tasks is None or ds.task in tasks)
        and any(lang in languages for lang in ds.languages)
    ]

    return datasets


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
