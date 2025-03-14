"""Command-line interface for benchmarking."""

import click

from .benchmarker import Benchmarker
from .dataset_configs import get_all_dataset_configs
from .enums import Device
from .languages import get_all_languages
from .tasks import get_all_tasks


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
    type=click.Choice(list(get_all_tasks().keys())),
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
    type=click.Choice(list(get_all_dataset_configs().keys())),
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
    "--use-flash-attention/--no-use-flash-attention",
    default=None,
    show_default=True,
    help="""Whether to use Flash Attention. If not specified then the model will use
    Flash Attention for generative models if a CUDA GPU is available and `flash-attn`
    or `vllm-flash-attn` are installed.""",
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
    "--debug/--no-debug",
    default=False,
    show_default=True,
    help="Whether to run the benchmark in debug mode. This prints out extra "
    "information and stores all outputs to the current working directory. Only "
    "relevant if the model is generative.",
)
@click.option(
    "--only-allow-safetensors",
    is_flag=True,
    help="Only allow loading models that have safetensors weights available",
    default=False,
)
def benchmark(
    model: tuple[str],
    dataset: tuple[str],
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
    use_flash_attention: bool | None,
    clear_model_cache: bool,
    evaluate_test_split: bool,
    few_shot: bool,
    num_iterations: int,
    api_base: str | None,
    api_version: str | None,
    debug: bool,
    only_allow_safetensors: bool,
) -> None:
    """Benchmark pretrained language models on language tasks."""
    models = list(model)
    datasets = None if len(dataset) == 0 else list(dataset)
    languages: list[str] = list(language)
    model_languages = None if len(model_language) == 0 else list(model_language)
    dataset_languages = None if len(dataset_language) == 0 else list(dataset_language)
    tasks = None if len(task) == 0 else list(task)
    batch_size_int = int(batch_size)
    device = Device[device.upper()] if device is not None else None

    benchmarker = Benchmarker(
        language=languages,
        model_language=model_languages,
        dataset_language=dataset_languages,
        task=tasks,
        dataset=datasets,
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
        use_flash_attention=use_flash_attention,
        clear_model_cache=clear_model_cache,
        evaluate_test_split=evaluate_test_split,
        few_shot=few_shot,
        num_iterations=num_iterations,
        api_base=api_base,
        api_version=api_version,
        debug=debug,
        run_with_cli=True,
        only_allow_safetensors=only_allow_safetensors,
    )

    # Perform the benchmark evaluation
    benchmarker.benchmark(model=models)


if __name__ == "__main__":
    benchmark()
