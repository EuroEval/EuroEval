"""Command-line interface for benchmarking."""

import collections.abc as c
import importlib.util
import logging
from pathlib import Path

import click

from euroeval.dataset_configs import get_all_dataset_configs

from .benchmarker import Benchmarker
from .data_models import DatasetConfig, Task
from .enums import Device, GenerativeType
from .languages import get_all_languages
from .logging_utils import log


@click.command()
@click.option(
    "--model",
    "-m",
    required=True,
    multiple=True,
    help="The ID of the model to benchmark.",
)
@click.option(
    "--task",
    "-t",
    default=None,
    show_default=True,
    multiple=True,
    help="The dataset tasks to benchmark the model(s) on.",
)
@click.option(
    "--language",
    "-l",
    default=["all"],
    show_default=True,
    multiple=True,
    metavar="ISO 639-1 LANGUAGE CODE",
    type=click.Choice(["all"] + list(get_all_languages().keys())),
    help="""The languages to benchmark, both for models and datasets. If "all" then all
    models will be benchmarked on all datasets.""",
)
@click.option(
    "--model-language",
    "-ml",
    default=None,
    show_default=True,
    multiple=True,
    metavar="ISO 639-1 LANGUAGE CODE",
    type=click.Choice(["all"] + list(get_all_languages().keys())),
    help="""The model languages to benchmark. If not specified then this will use the
    `language` value.""",
)
@click.option(
    "--dataset-language",
    "-dl",
    default=None,
    show_default=True,
    multiple=True,
    metavar="ISO 639-1 LANGUAGE CODE",
    type=click.Choice(["all"] + list(get_all_languages().keys())),
    help="""The dataset languages to benchmark. If "all" then the models will be
    benchmarked on all datasets. If not specified then this will use the `language`
    value.""",
)
@click.option(
    "--dataset",
    default=None,
    show_default=True,
    multiple=True,
    help="""The name of the benchmark dataset. We recommend to use the `task` and
    `language` options instead of this option.""",
)
@click.option(
    "--batch-size",
    default="32",
    type=click.Choice(["1", "2", "4", "8", "16", "32"]),
    help="The batch size to use.",
)
@click.option(
    "--progress-bar/--no-progress-bar",
    default=True,
    show_default=True,
    help="Whether to show a progress bar.",
)
@click.option(
    "--raise-errors/--no-raise-errors",
    default=False,
    show_default=True,
    help="Whether to raise errors instead of skipping the evaluation.",
)
@click.option(
    "--verbose/--no-verbose",
    "-v/-nv",
    default=False,
    show_default=True,
    help="Whether extra input should be outputted during benchmarking",
)
@click.option(
    "--save-results/--no-save-results",
    "-s/-ns",
    default=True,
    show_default=True,
    help="Whether results should not be stored to disk.",
)
@click.option(
    "--cache-dir",
    "-c",
    default=".euroeval_cache",
    show_default=True,
    help="The directory where models are datasets are cached.",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    show_default=True,
    help="""The API key to use for a given inference API. If you are benchmarking an "
    "OpenAI model then this would be the OpenAI API key, if you are benchmarking a "
    "model on the Hugging Face inference API then this would be the Hugging Face API "
    "key, and so on.""",
)
@click.option(
    "--force/--no-force",
    "-f",
    default=False,
    show_default=True,
    help="""Whether to force evaluation of models which have already been evaluated,
    with scores lying in the 'euroeval_benchmark_results.jsonl' file.""",
)
@click.option(
    "--device",
    default=None,
    show_default=True,
    type=click.Choice([device.lower() for device in Device.__members__]),
    help="""The device to use for evaluation. If not specified then the device will be
    set automatically.""",
)
@click.option(
    "--trust-remote-code/--no-trust-remote-code",
    default=False,
    show_default=True,
    help="""Whether to trust remote code. Only set this flag if you trust the supplier
    of the model.""",
)
@click.option(
    "--clear-model-cache/--no-clear-model-cache",
    default=False,
    show_default=True,
    help="""Whether to clear the model cache after benchmarking each model. Note that
    this will only remove the model files, and not the cached model outputs (which
    don't take up a lot of disk space). This is useful when benchmarking many models,
    to avoid running out of disk space.""",
)
@click.option(
    "--evaluate-test-split/--evaluate-val-split",
    default=False,
    show_default=True,
    help="""Whether to only evaluate on the test split. Only use this for your final
    evaluation, as the test split should not be used for development.""",
)
@click.option(
    "--few-shot/--zero-shot",
    default=True,
    show_default=True,
    help="Whether to only evaluate the model using few-shot evaluation. Only relevant "
    "if the model is generative.",
)
@click.option(
    "--num-iterations",
    default=10,
    show_default=True,
    help="""The number of times each model should be evaluated. This is only meant to
    be used for power users, and scores will not be allowed on the leaderboards if this
    is changed.""",
)
@click.option(
    "--api-base",
    default=None,
    show_default=True,
    help="The base URL for a given inference API. Only relevant if `model` refers to a "
    "model on an inference API.",
)
@click.option(
    "--api-version",
    default=None,
    show_default=True,
    help="The version of the API to use. Only relevant if `model` refers to a model on "
    "an inference API.",
)
@click.option(
    "--gpu-memory-utilization",
    default=0.8,
    show_default=True,
    help="The GPU memory utilization to use for vLLM. A larger value will result in "
    "faster evaluation, but at the risk of running out of GPU memory. Only reduce this "
    "if you are running out of GPU memory. Only relevant if the model is generative.",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    show_default=True,
    help="Whether to run the benchmark in debug mode. This prints out extra "
    "information and stores all outputs to the current working directory. Only "
    "relevant if the model is generative.",
)
@click.option(
    "--requires-safetensors",
    is_flag=True,
    help="Only allow loading models that have safetensors weights available",
    default=False,
)
@click.option(
    "--generative-type",
    type=click.Choice(["base", "instruction_tuned", "reasoning"]),
    default=None,
    show_default=True,
    help="The type of generative model. Only relevant if the model is generative. If "
    "not specified, the type will be inferred automatically.",
)
@click.option(
    "--download-only",
    is_flag=True,
    help="Only download the requested model weights and datasets, and exit.",
    default=False,
)
@click.option(
    "--custom-datasets-file",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default="custom_datasets.py",
    show_default=True,
    help="A path to a Python file containing DatasetConfig definitions for custom "
    "datasets.",
)
def benchmark(
    model: tuple[str],
    dataset: tuple[str | DatasetConfig],
    language: tuple[str],
    model_language: tuple[str],
    dataset_language: tuple[str],
    raise_errors: bool,
    task: tuple[str],
    batch_size: str,
    progress_bar: bool,
    save_results: bool,
    cache_dir: str,
    api_key: str | None,
    force: bool,
    verbose: bool,
    device: str | None,
    trust_remote_code: bool,
    clear_model_cache: bool,
    evaluate_test_split: bool,
    few_shot: bool,
    num_iterations: int,
    api_base: str | None,
    api_version: str | None,
    gpu_memory_utilization: float,
    debug: bool,
    requires_safetensors: bool,
    generative_type: str | None,
    download_only: bool,
    custom_datasets_file: Path,
) -> None:
    """Benchmark pretrained language models on language tasks."""
    models = list(model)
    datasets: c.Sequence[str | DatasetConfig] | None = (
        None if len(dataset) == 0 else list(dataset)
    )
    languages: list[str] = list(language)
    model_languages = None if len(model_language) == 0 else list(model_language)
    dataset_languages = None if len(dataset_language) == 0 else list(dataset_language)
    tasks: c.Sequence[str | Task] | None = None if len(task) == 0 else list(task)
    batch_size_int = int(batch_size)
    device = Device[device.upper()] if device is not None else None
    generative_type_obj = (
        GenerativeType[generative_type.upper()] if generative_type else None
    )

    # Load all defined DatasetConfig and Task objects from the custom datasets file
    if custom_datasets_file.exists():
        # Load the custom module
        spec = importlib.util.spec_from_file_location(
            name="custom_datasets_module", location=str(custom_datasets_file.resolve())
        )
        if spec is None:
            raise RuntimeError(
                "Could not load the spec for the custom datasets file from "
                f"{custom_datasets_file.resolve()}."
            )
        module = importlib.util.module_from_spec(spec=spec)
        if spec.loader is None:
            raise RuntimeError(
                "Could not load the module for the custom datasets file from "
                f"{custom_datasets_file.resolve()}."
            )
        spec.loader.exec_module(module)

        # Load all the custom dataset configurations and tasks from the module
        custom_dataset_configs: list[DatasetConfig] = [
            obj for obj in vars(module).values() if isinstance(obj, DatasetConfig)
        ]
        custom_task_objects: list[Task] = [
            obj for obj in vars(module).values() if isinstance(obj, Task)
        ]

        # If the user has not specified any datasets or tasks, we just use all the usual
        # datasets as well as all the custom ones that we loaded
        if datasets is None and tasks is None:
            datasets = custom_dataset_configs + list(get_all_dataset_configs().keys())

        # If the user has specified only datasets, then we replace the custom dataset
        # names that the user specified (if any) with the corresponding dataset configs
        # that we loaded
        elif datasets is not None and tasks is None:
            dataset_name_to_config = {
                config.name: config for config in custom_dataset_configs
            }
            datasets = [
                dataset_name_to_config.get(ds, ds) if isinstance(ds, str) else ds
                for ds in datasets
            ]

        # If the user has specified only tasks, then we replace the custom task names
        # that the user specified (if any) with the corresponding task objects that we
        # loaded
        elif datasets is None and tasks is not None:
            task_name_to_object = {
                task_obj.name: task_obj for task_obj in custom_task_objects
            }
            tasks = [
                task_name_to_object.get(t, t) if isinstance(t, str) else t
                for t in tasks
            ]

        # Log the loaded custom datasets and tasks
        dataset_str = (
            "the custom dataset"
            if len(custom_dataset_configs) == 1
            else f"{len(custom_dataset_configs):,} custom datasets"
        )
        task_str = (
            "the custom task"
            if len(custom_task_objects) == 1
            else f"{len(custom_task_objects):,} custom tasks"
        )
        log(
            f"Loaded {dataset_str} and {task_str} from "
            f"{custom_datasets_file.as_posix()!r}.\n",
            level=logging.INFO,
        )

    benchmarker = Benchmarker(
        language=languages,
        model_language=model_languages,
        dataset_language=dataset_languages,
        task=tasks,  # type: ignore[arg-type]
        dataset=datasets,  # type: ignore[arg-type]
        batch_size=batch_size_int,
        progress_bar=progress_bar,
        save_results=save_results,
        raise_errors=raise_errors,
        verbose=verbose,
        api_key=api_key,
        force=force,
        cache_dir=cache_dir,
        device=device,
        trust_remote_code=trust_remote_code,
        clear_model_cache=clear_model_cache,
        evaluate_test_split=evaluate_test_split,
        few_shot=few_shot,
        num_iterations=num_iterations,
        api_base=api_base,
        api_version=api_version,
        gpu_memory_utilization=gpu_memory_utilization,
        generative_type=generative_type_obj,
        debug=debug,
        run_with_cli=True,
        requires_safetensors=requires_safetensors,
        download_only=download_only,
    )

    # Perform the benchmark evaluation
    benchmarker.benchmark(model=models)


if __name__ == "__main__":
    benchmark()
