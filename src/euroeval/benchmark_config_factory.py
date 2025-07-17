"""Factory class for creating dataset configurations."""

import logging
import sys
import typing as t

import torch

from .data_models import BenchmarkConfig
from .dataset_configs import get_all_dataset_configs
from .enums import Device
from .exceptions import InvalidBenchmark
from .languages import get_all_languages
from .tasks import SPEED, get_all_tasks

if t.TYPE_CHECKING:
    from .data_models import Language, Task


logger = logging.getLogger("euroeval")


def build_benchmark_config(
    progress_bar: bool,
    save_results: bool,
    task: str | list[str] | None,
    dataset: str | list[str] | None,
    language: str | list[str],
    model_language: str | list[str] | None,
    dataset_language: str | list[str] | None,
    device: Device | None,
    batch_size: int,
    raise_errors: bool,
    cache_dir: str,
    api_key: str | None,
    force: bool,
    verbose: bool,
    trust_remote_code: bool,
    clear_model_cache: bool,
    evaluate_test_split: bool,
    few_shot: bool,
    num_iterations: int,
    api_base: str | None,
    api_version: str | None,
    gpu_memory_utilization: float,
    debug: bool,
    run_with_cli: bool,
    only_allow_safetensors: bool,
    first_time: bool = False,
) -> BenchmarkConfig:
    """Create a benchmark configuration.

    Args:
        progress_bar:
            Whether to show a progress bar when running the benchmark.
        save_results:
            Whether to save the benchmark results to a file.
        task:
            The tasks to include for dataset. If None then datasets will not be
            filtered based on their task.
        dataset:
            The datasets to include for task. If None then all datasets will be
            included, limited by the `task` parameter.
        language:
            The language codes of the languages to include, both for models and
            datasets. Here 'no' means both Bokmål (nb) and Nynorsk (nn). Set this
            to 'all' if all languages should be considered.
        model_language:
            The language codes of the languages to include for models. If None then
            the `language` parameter will be used.
        dataset_language:
            The language codes of the languages to include for datasets. If None then
            the `language` parameter will be used.
        device:
            The device to use for running the models. If None then the device will be
            set automatically.
        batch_size:
            The batch size to use for running the models.
        raise_errors:
            Whether to raise errors when running the benchmark.
        cache_dir:
            The directory to use for caching the models.
        api_key:
            The API key to use for a given inference server.
        force:
            Whether to force the benchmark to run even if the results are already
            cached.
        verbose:
            Whether to print verbose output when running the benchmark. This is
            automatically set if `debug` is True.
        trust_remote_code:
            Whether to trust remote code when running the benchmark.
        clear_model_cache:
            Whether to clear the model cache before running the benchmark.
        evaluate_test_split:
            Whether to use the test split for the datasets.
        few_shot:
            Whether to use few-shot learning for the models.
        num_iterations:
            The number of iterations each model should be evaluated for.
        api_base:
            The base URL for a given inference API. Only relevant if `model` refers to a
            model on an inference API.
        api_version:
            The version of the API to use for a given inference API.
        gpu_memory_utilization:
            The GPU memory utilization to use for vLLM. A larger value will result in
            faster evaluation, but at the risk of running out of GPU memory. Only reduce
            this if you are running out of GPU memory. Only relevant if the model is
            generative.
        debug:
            Whether to run the benchmark in debug mode.
        run_with_cli:
            Whether the benchmark is being run with the CLI.
        only_allow_safetensors:
            Whether to only allow evaluations of models stored as safetensors.
        first_time:
            Whether this is the first time the benchmark configuration is being created.
            Defaults to False.

    Returns:
        The benchmark configuration.
    """
    language_codes = get_correct_language_codes(language_codes=language)
    model_languages = prepare_languages(
        language_codes=model_language, default_language_codes=language_codes
    )
    dataset_languages = prepare_languages(
        language_codes=dataset_language, default_language_codes=language_codes
    )

    tasks, datasets = prepare_tasks_and_datasets(
        task=task, dataset=dataset, dataset_languages=dataset_languages
    )

    torch_device = prepare_device(device=device)

    # Set variable with number of iterations
    if hasattr(sys, "_called_from_test"):
        num_iterations = 1

    return BenchmarkConfig(
        model_languages=model_languages,
        dataset_languages=dataset_languages,
        tasks=tasks,
        datasets=datasets,
        batch_size=batch_size,
        raise_errors=raise_errors,
        cache_dir=cache_dir,
        api_key=api_key,
        force=force,
        progress_bar=progress_bar,
        save_results=save_results,
        verbose=verbose or debug,
        device=torch_device,
        trust_remote_code=trust_remote_code,
        clear_model_cache=clear_model_cache,
        evaluate_test_split=evaluate_test_split,
        few_shot=few_shot,
        num_iterations=num_iterations,
        api_base=api_base,
        api_version=api_version,
        gpu_memory_utilization=gpu_memory_utilization,
        debug=debug,
        run_with_cli=run_with_cli,
        only_allow_safetensors=only_allow_safetensors,
    )


def get_correct_language_codes(language_codes: str | list[str]) -> list[str]:
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
        languages = language_codes

    # If `languages` contains 'no' then also include 'nb' and 'nn'. Conversely, if
    # either 'nb' or 'nn' are specified then also include 'no'.
    if "no" in languages:
        languages = list(set(languages) | {"nb", "nn"})
    elif "nb" in languages or "nn" in languages:
        languages = list(set(languages) | {"no"})

    return languages


def prepare_languages(
    language_codes: str | list[str] | None, default_language_codes: list[str]
) -> list["Language"]:
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
    languages_str: list[str]
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


def prepare_tasks_and_datasets(
    task: str | list[str] | None,
    dataset_languages: list["Language"],
    dataset: str | list[str] | None,
) -> tuple[list["Task"], list[str]]:
    """Prepare task(s) and dataset(s) for benchmarking.

    Args:
        task:
            The tasks to include for dataset. If None then datasets will not be
            filtered based on their task.
        dataset_languages:
            The languages of the datasets in the benchmark.
        dataset:
            The datasets to include for task. If None then all datasets will be
            included, limited by the `task` and `dataset_languages` parameters.

    Returns:
        The prepared tasks and datasets.

    Raises:
        InvalidBenchmark:
            If the task or dataset is not found in the benchmark tasks or datasets.
    """
    # Create a dictionary that maps benchmark tasks to their associated benchmark
    # task objects, and a dictionary that maps dataset names to their associated
    # dataset configuration objects
    task_mapping = get_all_tasks()
    all_dataset_configs = get_all_dataset_configs()

    # Create the list of dataset tasks
    try:
        if task is None:
            tasks = [t for t in task_mapping.values() if t != SPEED]
        elif isinstance(task, str):
            tasks = [task_mapping[task]]
        else:
            tasks = [task_mapping[t] for t in task]
    except KeyError as e:
        raise InvalidBenchmark(f"Task {e} not found in the benchmark tasks.") from e

    all_official_datasets = [
        dataset_name
        for dataset_name, dataset_config in all_dataset_configs.items()
        if not dataset_config.unofficial
    ]
    if dataset is None:
        dataset = all_official_datasets
    elif isinstance(dataset, str):
        dataset = [dataset]

    all_datasets = list(all_dataset_configs.keys())
    invalid_datasets = set(dataset) - set(all_datasets)
    if invalid_datasets:
        raise InvalidBenchmark(
            f"Dataset(s) {', '.join(invalid_datasets)} not found in the benchmark "
            "datasets."
        )

    datasets = [
        dataset_name
        for dataset_name, dataset_config in all_dataset_configs.items()
        if dataset_name in dataset
        and dataset_config.task in tasks
        and set(dataset_config.languages).intersection(dataset_languages)
    ]

    return tasks, datasets


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
