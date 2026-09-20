"""Class that benchmarks language models."""

import collections.abc as c
import contextlib
import datetime as dt
import logging
import os
import re
import typing as t
from dataclasses import replace
from pathlib import Path
from shutil import rmtree
from time import sleep

from huggingface_hub import snapshot_download
from torch.distributed import destroy_process_group

from .benchmark_config_factory import build_benchmark_config
from .constants import ATTENTION_BACKENDS, GENERATIVE_PIPELINE_TAGS, ORTHOGONAL_TASKS
from .data_loading import load_data, load_raw_data
from .data_models import BenchmarkConfigParams, BenchmarkResult, get_package_version
from .enums import Device, GenerativeType, InferenceBackend, ModelType, ShotMode
from .exceptions import HuggingFaceHubDown, InvalidBenchmark, InvalidModel
from .finetuning import finetune
from .generation import generate
from .logging_utils import adjust_logging_level, get_pbar, log, log_once
from .metrics.bpc import bpc_metric
from .model_config import get_model_config
from .model_loading import load_model
from .scores import log_scores
from .speed_benchmark import benchmark_speed
from .string_utils import split_model_id
from .tasks import SPEED
from .utils import enforce_reproducibility, get_hf_token, internet_connection_available

if t.TYPE_CHECKING:
    from .benchmark_modules import BenchmarkModule
    from .data_models import BenchmarkConfig, DatasetConfig, ModelConfig, Task


class Benchmarker:
    """Benchmarking all the language models.

    Attributes:
        benchmark_config_default_params:
            The default parameters for the benchmark configuration.
        benchmark_config:
            The benchmark configuration.
        force:
            Whether to force evaluations of models, even if they have been benchmarked
            already.
        results_path:
            The path to the results file.
        benchmark_results:
            The benchmark results.
    """

    def __init__(
        self,
        progress_bar: bool = True,
        save_results: bool = True,
        task: "str | Task | c.Sequence[str | Task] | None" = None,
        dataset: "str | DatasetConfig | c.Sequence[str | DatasetConfig] | None" = None,
        language: str | c.Sequence[str] = "all",
        device: Device | None = None,
        finetuning_batch_size: int = 32,
        raise_errors: bool = False,
        cache_dir: str = ".euroeval_cache",
        api_key: str | None = None,
        force: bool = False,
        verbose: bool = False,
        trust_remote_code: bool = False,
        clear_model_cache: bool = False,
        evaluate_test_split: bool = False,
        few_shot: bool | None = None,
        num_iterations: int = 10,
        api_base: str | None = None,
        api_version: str | None = None,
        gpu_memory_utilization: float = 0.8,
        attention_backend: t.Literal[
            *ATTENTION_BACKENDS  # ty: ignore[invalid-type-form]
        ]
        | None = None,
        generative_type: GenerativeType | None = None,
        use_bits_per_character: bool = False,
        custom_datasets_file: Path | str = Path("custom_datasets.py"),
        debug: bool = False,
        run_with_cli: bool = False,
        requires_safetensors: bool = False,
        download_only: bool = False,
        max_context_length: int | None = None,
        vocabulary_size: int | None = None,
    ) -> None:
        """Initialise the benchmarker.

        Args:
            progress_bar:
                Whether progress bars should be shown. Defaults to True.
            save_results:
                Whether to save the benchmark results to
                'euroeval_benchmark_results.jsonl'. Defaults to True.
            task:
                The tasks benchmark the model(s) on. Mutually exclusive with `dataset`.
                If both `task` and `dataset` are None then all datasets will be
                benchmarked.
            dataset:
                The datasets to benchmark on. Mutually exclusive with `task`. If both
                `task` and `dataset` are None then all datasets will be
                benchmarked. Note that `language` still filters the datasets a
                `dataset` names, so a `language` alongside a `dataset` narrows the run
                rather than widening it; the CLI rejects the combination.
            language:
                The language codes of the languages to include, both for models and
                datasets. Set this to 'all' if all languages should be considered.
                Defaults to "all".
            device:
                The device to use for benchmarking. Defaults to None.
            finetuning_batch_size:
                The batch size to use when finetuning. Defaults to 32.
            raise_errors:
                Whether to raise errors instead of skipping the model evaluation.
                Defaults to False.
            cache_dir:
                Directory to store cached models. Defaults to '.euroeval_cache'.
            api_key:
                The API key to use for a given inference API.
            force:
                Whether to force evaluations of models, even if they have been
                benchmarked already. Defaults to False.
            verbose:
                Whether to output additional output. This is automatically set if
                `debug` is True. Defaults to False.
            trust_remote_code:
                Whether to trust remote code when loading models. Defaults to False.
            clear_model_cache:
                Whether to clear the model cache after benchmarking each model.
                Defaults to False.
            evaluate_test_split:
                Whether to evaluate the test split of the datasets. Defaults to False.
            few_shot:
                Explicitly select few-shot (True) or zero-shot (False) evaluation.
                When None, select automatically from the model type. Only relevant if
                the model is generative. Defaults to None.
            num_iterations:
                The number of times each model should be evaluated. This is only meant
                to be used for power users, and scores will not be allowed on the
                leaderboards if this is changed. Defaults to 10.
            api_base:
                The base URL for a given inference API. Only relevant if `model` refers
                to a model on an inference API. Defaults to None.
            api_version:
                The version of the API to use. Defaults to None.
            gpu_memory_utilization:
                The GPU memory utilization to use for vLLM. Only relevant if the model
                is generative. A larger value will result in faster evaluation, but at
                the risk of running out of GPU memory. Only reduce this if you are
                running out of GPU memory. Defaults to 0.9.
            attention_backend:
                The attention backend to use for vLLM. Only relevant if the model is
                generative. If None then vLLM will automatically choose the best
                backend. Defaults to None.
            generative_type:
                The type of generative model to benchmark. Only relevant if the model is
                generative. If not specified, then the type will be inferred based on
                the tags of the model. Defaults to None.
            use_bits_per_character:
                Whether to compute bits-per-character (BPC) on the ground-truth answer.
                For multiple-choice tasks, treats benchmark as text-to-text with bare
                question → full answer text. Only supported for base decoder models.
                Defaults to False.
            custom_datasets_file:
                Path to a Python file defining custom datasets. Defaults to
                'custom_datasets.py'.
            debug:
                Whether to output debug information. Defaults to False.
            run_with_cli:
                Whether the benchmarker is being run from the command-line interface.
                Defaults to False.
            requires_safetensors:
                Whether to only allow models that use the safetensors format. Defaults
                to False.
            download_only:
                Whether to only download models and datasets without performing any
                benchmarking. Defaults to False.
            max_context_length:
                Override for the maximum context length of the model. If None, the value
                will be inferred automatically from the model. Defaults to None.
            vocabulary_size:
                Override for the vocabulary size of the model. If None, the value will
                be inferred automatically from the model. Defaults to None.

        Raises:
            ValueError:
                If both `task` and `dataset` are specified, or if `download_only` is
                True and we have no internet connection.
        """
        if task is not None and dataset is not None:
            raise ValueError("Only one of `task` and `dataset` can be specified.")

        if not internet_connection_available() and download_only:
            msg = "It appears you do not have an internet connection, but "
            if run_with_cli:
                msg += "the --download-only flag was set."
            else:
                msg += "the argument `download_only` was set to True."
            raise ValueError(msg)

        # If FULL_LOG has been set, then force verbose mode
        if os.getenv("FULL_LOG", "0") == "1":
            verbose = True

        adjust_logging_level(verbose=verbose)

        self.benchmark_config_default_params = BenchmarkConfigParams(
            task=task,
            dataset=dataset,
            progress_bar=progress_bar,
            save_results=save_results,
            language=language,
            device=device,
            finetuning_batch_size=finetuning_batch_size,
            raise_errors=raise_errors,
            cache_dir=cache_dir,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            trust_remote_code=trust_remote_code,
            clear_model_cache=clear_model_cache,
            evaluate_test_split=evaluate_test_split,
            few_shot=few_shot,
            num_iterations=num_iterations,
            requires_safetensors=requires_safetensors,
            download_only=download_only,
            gpu_memory_utilization=gpu_memory_utilization,
            attention_backend=attention_backend,
            generative_type=generative_type,
            use_bits_per_character=use_bits_per_character,
            custom_datasets_file=Path(custom_datasets_file),
            verbose=verbose,
            force=force,
            debug=debug,
            run_with_cli=run_with_cli,
            max_context_length=max_context_length,
            vocabulary_size=vocabulary_size,
        )

        self.benchmark_config = build_benchmark_config(
            benchmark_config_params=self.benchmark_config_default_params
        )

        # Initialise variable storing model lists, so we only have to fetch it once
        self._model_lists: dict[str, c.Sequence[str]] | None = None

        self.results_path = Path.cwd() / "euroeval_benchmark_results.jsonl"
        adjust_logging_level(verbose=self.benchmark_config.verbose)

    def benchmark(
        self,
        model: c.Sequence[str] | str,
        task: "str | Task | c.Sequence[str | Task] | None" = None,
        dataset: "str | DatasetConfig | c.Sequence[str | DatasetConfig] | None" = None,
        progress_bar: bool | None = None,
        save_results: bool | None = None,
        language: str | c.Sequence[str] | None = None,
        device: Device | None = None,
        finetuning_batch_size: int | None = None,
        raise_errors: bool | None = None,
        cache_dir: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        trust_remote_code: bool | None = None,
        clear_model_cache: bool | None = None,
        evaluate_test_split: bool | None = None,
        few_shot: bool | None = None,
        num_iterations: int | None = None,
        requires_safetensors: bool | None = None,
        download_only: bool | None = None,
        gpu_memory_utilization: float | None = None,
        generative_type: GenerativeType | None = None,
        use_bits_per_character: bool | None = None,
        attention_backend: t.Literal[
            *ATTENTION_BACKENDS  # ty: ignore[invalid-type-form]
        ]
        | None = None,
        custom_datasets_file: Path | str | None = None,
        force: bool | None = None,
        verbose: bool | None = None,
        debug: bool | None = None,
        max_context_length: int | None = None,
        vocabulary_size: int | None = None,
    ) -> c.Sequence[BenchmarkResult]:
        """Benchmarks models on datasets.

        Args:
            model:
                The full Hugging Face Hub path(s) to the pretrained transformer model.
                The specific model version to use can be added after the suffix '@':
                "model@v1.0.0". It can be a branch name, a tag name, or a commit id,
                and defaults to the latest version if not specified.
            task:
                The tasks benchmark the model(s) on. Mutually exclusive with `dataset`.
                If both `task` and `dataset` are None then all datasets will be
                benchmarked. Defaults to None.
            dataset:
                The datasets to benchmark on. Mutually exclusive with `task`. If both
                `task` and `dataset` are None then all datasets will be benchmarked.
                Defaults to None.
            progress_bar:
                Whether progress bars should be shown. Defaults to the value specified
                when initialising the benchmarker.
            save_results:
                Whether to save the benchmark results to
                'euroeval_benchmark_results.jsonl'. Defaults to the value specified
                when initialising the benchmarker.
            language:
                The language codes of the languages to include, both for models and
                datasets. Here 'no' means both Bokmål (nb) and Nynorsk (nn).
                Set this to 'all' if all languages should be considered.
                Defaults to the value specified when initialising the benchmarker.
            device:
                The device to use for benchmarking. Defaults to the value specified when
                initialising the benchmarker.
            finetuning_batch_size:
                The batch size to use for finetuning. Defaults to the value specified
                when initialising the benchmarker.
            raise_errors:
                Whether to raise errors instead of skipping the model evaluation.
            cache_dir:
                Directory to store cached models. Defaults to the value specified when
                initialising the benchmarker.
            api_key:
                The API key to use for a given inference server. Defaults to the value
                specified when initialising the benchmarker.
            api_base:
                The base URL for a given inference API. Only relevant if `model` refers
                to a model on an inference API. Defaults to the value specified when
                initialising the benchmarker.
            api_version:
                The version of the API to use. Defaults to the value specified when
                initialising the benchmarker.
            trust_remote_code:
                Whether to trust remote code when loading models. Defaults to the value
                specified when initialising the benchmarker.
            clear_model_cache:
                Whether to clear the model cache after benchmarking each model. Defaults
                to the value specified when initialising the benchmarker.
            evaluate_test_split:
                Whether to evaluate the test split of the datasets. Defaults to the
                value specified when initialising the benchmarker.
            few_shot:
                Explicitly select few-shot (True) or zero-shot (False) evaluation.
                When None, select automatically from the model type. Defaults to the
                value specified when initialising the benchmarker.
            num_iterations:
                The number of times each model should be evaluated. This is only meant
                to be used for power users, and scores will not be allowed on the
                leaderboards if this is changed. Defaults to the value specified when
                initialising the benchmarker.
            requires_safetensors:
                Whether to only allow models that use the safetensors format. Defaults
                to the value specified when initialising the benchmarker.
            download_only:
                Whether to only download the models without evaluating them. Defaults
                to the value specified when initialising the benchmarker.
            gpu_memory_utilization:
                The GPU memory utilization to use for vLLM. Only relevant if the model
                is generative. A larger value will result in faster evaluation, but at
                the risk of running out of GPU memory. Only reduce this if you are
                running out of GPU memory. Defaults to the value specified when
                initialising the benchmarker.
            generative_type:
                The type of generative model to benchmark. Only relevant if the model is
                generative. If not specified, then the type will be inferred based on
                the tags of the model. Defaults to the value specified when initialising
                the benchmarker.
            use_bits_per_character:
                Whether to compute bits-per-character (BPC) on the ground-truth answer.
                For multiple-choice tasks, treats benchmark as text-to-text with bare
                question → full answer text. Only supported for base decoder models.
                Defaults to the value specified when initialising the benchmarker.
            attention_backend:
                The attention backend to use for vLLM. Only relevant if the model is
                generative. Defaults to the value specified when initialising the
                benchmarker.
            custom_datasets_file:
                Path to a Python file defining custom datasets. Defaults to the value
                specified when initialising the benchmarker.
            force:
                Whether to force evaluations of models, even if they have been
                benchmarked already. Defaults to the value specified when initialising
                the benchmarker.
            verbose:
                Whether to output additional output. Defaults to the value specified
                when initialising the benchmarker.
            debug:
                Whether to output debug information. Defaults to the value specified
                when initialising the benchmarker.
            max_context_length:
                Override for the maximum context length of the model. If None, the
                value will be inferred automatically from the model. Defaults to the
                value specified when initialising the benchmarker.
            vocabulary_size:
                Override for the vocabulary size of the model. If None, the value will
                be inferred automatically from the model. Defaults to the value
                specified when initialising the benchmarker.

        Returns:
            A list of benchmark results.

        Raises:
            ValueError:
                If both `task` and `dataset` are specified.
        """
        if task is not None and dataset is not None:
            raise ValueError("Only one of `task` and `dataset` can be specified.")

        # Determine verbose mode
        is_verbose = (
            verbose
            if verbose is not None
            else self.benchmark_config_default_params.verbose
        )
        effective_debug = (
            debug if debug is not None else self.benchmark_config_default_params.debug
        )
        if os.getenv("FULL_LOG", "0") == "1" or effective_debug:
            is_verbose = True

        log_once(
            "Started EuroEval run."
            if is_verbose
            else "Started EuroEval run. Run with `--verbose` for more information.",
            level=logging.INFO,
        )

        # Announce BPC mode if active
        is_bpc = (
            use_bits_per_character
            if use_bits_per_character is not None
            else self.benchmark_config_default_params.use_bits_per_character
        )
        if is_bpc:
            log_once(
                "    ↳ Running in bits-per-character (BPC) mode: every dataset will be "
                "scored by the bits-per-character of the ground-truth answer (lower is "
                "better) instead of the usual task metrics.",
                level=logging.INFO,
            )

        # Build benchmark config
        benchmark_config = self._build_benchmark_config(
            task=task,
            dataset=dataset,
            progress_bar=progress_bar,
            save_results=save_results,
            language=language,
            device=device,
            finetuning_batch_size=finetuning_batch_size,
            raise_errors=raise_errors,
            cache_dir=cache_dir,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            trust_remote_code=trust_remote_code,
            clear_model_cache=clear_model_cache,
            evaluate_test_split=evaluate_test_split,
            few_shot=few_shot,
            num_iterations=num_iterations,
            requires_safetensors=requires_safetensors,
            download_only=download_only,
            gpu_memory_utilization=gpu_memory_utilization,
            generative_type=generative_type,
            use_bits_per_character=use_bits_per_character,
            attention_backend=attention_backend,
            custom_datasets_file=custom_datasets_file,
            force=force,
            verbose=verbose,
            debug=debug,
            max_context_length=max_context_length,
            vocabulary_size=vocabulary_size,
        )

        adjust_logging_level(verbose=benchmark_config.verbose)

        if benchmark_config.clear_model_cache:
            clear_model_cache_fn(cache_dir=benchmark_config.cache_dir)

        model_ids = self._prepare_model_ids(model_id=model)
        dataset_configs = benchmark_config.datasets

        # Fetch model configs and create mapping
        model_configs = self._fetch_model_configs(model_ids, benchmark_config)
        model_mapping = self._create_model_dataset_mapping(
            model_configs, dataset_configs
        )

        existing_results = self.benchmark_results
        current_results: list[BenchmarkResult] = []
        num_finished = 0
        num_skipped = 0
        num_errored = 0
        total_benchmarks = 0

        for model_config in model_configs:
            datasets = model_mapping[model_config]
            if not datasets:
                continue

            loaded_model: "BenchmarkModule | None" = None
            model_finished = 0
            model_skipped = 0
            model_errored = 0
            try:
                self._check_adapter_requirements(model_config, benchmark_config)
                loaded_model, pending, cached, load_error = (
                    self._prepare_shot_benchmarks(
                        model_config=model_config,
                        datasets=datasets,
                        benchmark_config=benchmark_config,
                        existing_results=existing_results,
                    )
                )
                current_results.extend(
                    record for record in cached if record not in current_results
                )
                total_benchmarks += len(pending)
                if load_error is not None:
                    if benchmark_config.raise_errors:
                        raise load_error
                    log(load_error.message, level=logging.ERROR)
                    model_errored += len(pending)
                    continue
                for pending_index, (shot_mode, dataset_config) in enumerate(pending):
                    mode_config = replace(
                        benchmark_config, few_shot=shot_mode is ShotMode.FEW_SHOT
                    )
                    self._update_benchmark_config_for_dataset(
                        dataset_config, mode_config
                    )
                    if loaded_model is not None:
                        loaded_model.benchmark_config = mode_config
                    if benchmark_config.download_only:
                        self._download(dataset_config, model_config, mode_config)
                        model_finished += 1
                        continue
                    if (
                        loaded_model is not None
                        and loaded_model.generative_type
                        not in dataset_config.allowed_generative_types
                    ):
                        log(
                            "Skipping the benchmark of model "
                            f"{model_config.model_id!r} on dataset "
                            f"{dataset_config.name!r} because the model has "
                            f"generative type {loaded_model.generative_type} and the "
                            "dataset does not allow it.",
                            level=logging.DEBUG,
                        )
                        model_skipped += 1
                        continue
                    output_or_err = self._benchmark_single(
                        model=loaded_model,
                        model_config=model_config,
                        dataset_config=dataset_config,
                        benchmark_config=mode_config,
                        num_finished_benchmarks=(
                            model_finished + model_skipped + model_errored
                        ),
                        num_total_benchmarks=len(pending),
                    )
                    model_finished, model_skipped, model_errored, should_break = (
                        self._handle_benchmark_result(
                            result_or_error=output_or_err,
                            dataset_config=dataset_config,
                            benchmark_config=mode_config,
                            num_finished=model_finished,
                            num_skipped=model_skipped,
                            num_errored=model_errored,
                            model_config=model_config,
                            model_mapping=model_mapping,
                            current_results=current_results,
                            remaining_work=len(pending) - pending_index - 1,
                        )
                    )
                    if should_break:
                        break
            finally:
                num_finished += model_finished
                num_skipped += model_skipped
                num_errored += model_errored
                loaded_model = None
                if benchmark_config.clear_model_cache:
                    clear_model_cache_fn(cache_dir=benchmark_config.cache_dir)

        if total_benchmarks == 0 and num_errored == 0:
            log(
                "No benchmarks to run, as all the selected models have already been "
                "benchmarked on all the selected datasets.",
                level=logging.INFO,
            )
            return current_results

        # Log summary
        summary = self._generate_summary_message(num_finished, num_skipped, num_errored)
        if summary:
            log(summary, level=logging.INFO)

        # Clean up process group
        with contextlib.suppress(Exception):
            destroy_process_group()

        return current_results

    def _benchmark_single(
        self,
        model: "BenchmarkModule | None",
        model_config: "ModelConfig",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
        num_finished_benchmarks: int,
        num_total_benchmarks: int,
    ) -> BenchmarkResult | InvalidBenchmark | InvalidModel:
        """Benchmark a single model on a single dataset.

        Args:
            model:
                The model to benchmark.
            model_config:
                The configuration of the model we are evaluating.
            dataset_config:
                The configuration of the dataset we are evaluating on.
            benchmark_config:
                The general benchmark configuration.
            num_finished_benchmarks:
                The number of benchmarks that have already been completed.
            num_total_benchmarks:
                The total number of benchmarks to be completed.

        Returns:
            The benchmark result, or an error if the benchmark was unsuccessful.

        Raises:
            RuntimeError:
                If the MPS fallback is not enabled when required.
            InvalidBenchmark:
                If the benchmark was unsuccessful.
            InvalidModel:
                If the model is invalid.
        """
        if model is not None:
            try:
                model.update_dataset_config(dataset_config=dataset_config)
            except InvalidBenchmark as e:
                return e

        for _ in range(num_attempts := 5):
            try:
                # Set random seeds to enforce reproducibility of the randomly
                # initialised weights
                rng = enforce_reproducibility()

                if model is None or model_config.model_type != ModelType.GENERATIVE:
                    model = load_model(
                        model_config=model_config,
                        dataset_config=dataset_config,
                        benchmark_config=benchmark_config,
                    )
                assert model is not None

                initial_logging(
                    model_config=model_config,
                    dataset_config=dataset_config,
                    benchmark_config=benchmark_config,
                    num_finished_benchmarks=num_finished_benchmarks,
                    num_total_benchmarks=num_total_benchmarks,
                )

                if dataset_config.task == SPEED:
                    scores = benchmark_speed(
                        model=model, benchmark_config=benchmark_config
                    )

                else:
                    bootstrapped_datasets = load_data(
                        rng=rng,
                        dataset_config=dataset_config,
                        benchmark_config=benchmark_config,
                    )
                    prepared_datasets = model.prepare_datasets(
                        datasets=bootstrapped_datasets, task=dataset_config.task
                    )
                    if model_config.model_type == ModelType.GENERATIVE:
                        scores = generate(
                            model=model,
                            datasets=prepared_datasets,
                            model_config=model_config,
                            dataset_config=dataset_config,
                            benchmark_config=benchmark_config,
                        )
                    else:
                        scores = finetune(
                            model=model,
                            datasets=prepared_datasets,
                            model_config=model_config,
                            dataset_config=dataset_config,
                            benchmark_config=benchmark_config,
                        )

                results = log_scores(
                    dataset_name=dataset_config.logging_string,
                    metrics=(
                        [bpc_metric]
                        if benchmark_config.use_bits_per_character
                        else dataset_config.task.metrics
                    ),
                    scores=scores,
                    model_id=model_config.model_id,
                    model_revision=model_config.revision,
                    model_param=model_config.param,
                )

                model_id_to_be_stored = model_config.model_id
                if model_config.revision != "main":
                    model_id_to_be_stored += f"@{model_config.revision}"
                if model_config.param is not None:
                    model_id_to_be_stored += f"#{model_config.param}"

                record = BenchmarkResult(
                    dataset=dataset_config.name,
                    task=dataset_config.task.name,
                    languages=[language.code for language in dataset_config.languages],
                    model=model_id_to_be_stored,
                    results=results,
                    num_model_parameters=model.num_params,
                    max_sequence_length=model.model_max_length,
                    vocabulary_size=model.vocab_size,
                    merge=model_config.merge,
                    generative=model_config.model_type == ModelType.GENERATIVE,
                    generative_type=(
                        model.generative_type.value
                        if model.generative_type is not None
                        else None
                    ),
                    few_shot=(
                        None
                        if dataset_config.task.requires_zero_shot
                        else benchmark_config.few_shot
                    ),
                    validation_split=(
                        None
                        if dataset_config.val_split is None
                        else not benchmark_config.evaluate_test_split
                    ),
                    use_bits_per_character=benchmark_config.use_bits_per_character,
                    release_date=model_config.release_date,
                    vllm_version=(
                        get_package_version("vllm")
                        if model_config.inference_backend == InferenceBackend.VLLM
                        else None
                    ),
                    litellm_version=(
                        get_package_version("litellm")
                        if model_config.inference_backend == InferenceBackend.LITELLM
                        else None
                    ),
                )
                log(f"Results:\n{results}", level=logging.DEBUG)
                return record

            except HuggingFaceHubDown:
                wait_time = 30
                log(
                    f"The Hugging Face Hub seems to be down. Retrying in {wait_time} "
                    "seconds.",
                    level=logging.DEBUG,
                )
                sleep(wait_time)
                continue

            except (InvalidBenchmark, InvalidModel) as e:
                # If the model ID is not valid then raise an error
                model_err_msg = "does not exist on the Hugging Face Hub"
                if benchmark_config.raise_errors and model_err_msg in str(e):
                    raise e

                # Otherwise, if the error is due to the MPS fallback not being enabled,
                # then raise an error asking the user to enable it
                elif "PYTORCH_ENABLE_MPS_FALLBACK" in str(e):
                    raise RuntimeError(
                        "The benchmark failed because the environment variable "
                        "`PYTORCH_ENABLE_MPS_FALLBACK` is not set. Please set this "
                        "environment variable to `1` and try again."
                    )

                elif benchmark_config.raise_errors:
                    raise e
                return e
        else:
            return InvalidBenchmark(
                f"Failed to benchmark model {model_config.model_id!r} on dataset "
                f"{dataset_config.name!r} after {num_attempts} attempts."
            )

    def _build_benchmark_config(self, **params) -> "BenchmarkConfig":
        """Build benchmark configuration from parameters.

        Args:
            **params:
                Override parameters for the benchmark configuration.

        Returns:
            The benchmark configuration.
        """

        def _get_param[T](name: str, default: T) -> T:
            """Get parameter value, falling back to default if None.

            Args:
                name:
                    The parameter name.
                default:
                    The default value to return if the parameter is None.

            Returns:
                The parameter value if not None, otherwise the default value.
            """
            value = params.get(name)
            return default if value is None else value

        return build_benchmark_config(
            benchmark_config_params=BenchmarkConfigParams(
                task=_get_param("task", self.benchmark_config_default_params.task),
                dataset=_get_param(
                    "dataset", self.benchmark_config_default_params.dataset
                ),
                progress_bar=_get_param(
                    "progress_bar", self.benchmark_config_default_params.progress_bar
                ),
                save_results=_get_param(
                    "save_results", self.benchmark_config_default_params.save_results
                ),
                language=_get_param(
                    "language", self.benchmark_config_default_params.language
                ),
                device=_get_param(
                    "device", self.benchmark_config_default_params.device
                ),
                finetuning_batch_size=_get_param(
                    "finetuning_batch_size",
                    self.benchmark_config_default_params.finetuning_batch_size,
                ),
                raise_errors=_get_param(
                    "raise_errors", self.benchmark_config_default_params.raise_errors
                ),
                cache_dir=_get_param(
                    "cache_dir", self.benchmark_config_default_params.cache_dir
                ),
                api_key=_get_param(
                    "api_key", self.benchmark_config_default_params.api_key
                ),
                api_base=_get_param(
                    "api_base", self.benchmark_config_default_params.api_base
                ),
                api_version=_get_param(
                    "api_version", self.benchmark_config_default_params.api_version
                ),
                trust_remote_code=_get_param(
                    "trust_remote_code",
                    self.benchmark_config_default_params.trust_remote_code,
                ),
                clear_model_cache=_get_param(
                    "clear_model_cache",
                    self.benchmark_config_default_params.clear_model_cache,
                ),
                evaluate_test_split=_get_param(
                    "evaluate_test_split",
                    self.benchmark_config_default_params.evaluate_test_split,
                ),
                few_shot=_get_param(
                    "few_shot", self.benchmark_config_default_params.few_shot
                ),
                num_iterations=_get_param(
                    "num_iterations",
                    self.benchmark_config_default_params.num_iterations,
                ),
                requires_safetensors=_get_param(
                    "requires_safetensors",
                    self.benchmark_config_default_params.requires_safetensors,
                ),
                download_only=_get_param(
                    "download_only", self.benchmark_config_default_params.download_only
                ),
                gpu_memory_utilization=_get_param(
                    "gpu_memory_utilization",
                    self.benchmark_config_default_params.gpu_memory_utilization,
                ),
                generative_type=_get_param(
                    "generative_type",
                    self.benchmark_config_default_params.generative_type,
                ),
                use_bits_per_character=_get_param(
                    "use_bits_per_character",
                    self.benchmark_config_default_params.use_bits_per_character,
                ),
                attention_backend=_get_param(
                    "attention_backend",
                    self.benchmark_config_default_params.attention_backend,
                ),
                custom_datasets_file=Path(params["custom_datasets_file"])
                if params.get("custom_datasets_file")
                else self.benchmark_config_default_params.custom_datasets_file,
                force=_get_param("force", self.benchmark_config_default_params.force),
                verbose=_get_param(
                    "verbose", self.benchmark_config_default_params.verbose
                ),
                debug=_get_param("debug", self.benchmark_config_default_params.debug),
                run_with_cli=self.benchmark_config_default_params.run_with_cli,
                max_context_length=_get_param(
                    "max_context_length",
                    self.benchmark_config_default_params.max_context_length,
                ),
                vocabulary_size=_get_param(
                    "vocabulary_size",
                    self.benchmark_config_default_params.vocabulary_size,
                ),
            )
        )

    def _check_adapter_requirements(
        self, model_config: "ModelConfig", benchmark_config: "BenchmarkConfig"
    ) -> None:
        """Check adapter model requirements.

        Args:
            model_config:
                The model configuration.
            benchmark_config:
                The benchmark configuration.

        Raises:
            InvalidModel:
                If offline benchmarking of adapter models is attempted.
        """
        if not model_config.adapter_base_model_id:
            return
        msg = (
            "If offline support is important to you, please consider opening an issue "
            "at https://github.com/EuroEval/EuroEval/issues."
        )
        if not internet_connection_available():
            raise InvalidModel(
                "Offline benchmarking of models with adapters is not currently "
                "supported. An active internet connection is required. " + msg
            )
        if benchmark_config.download_only:
            msg_full = (
                "You are using download-only mode with a model that includes an "
                "adapter. Please note that offline benchmarking of adapter models "
                "is not currently supported - an internet connection will be required "
                "during evaluation in this case. " + msg
            )
            log_once(msg_full, level=logging.WARNING)

    def _create_model_dataset_mapping(
        self,
        model_configs: list["ModelConfig"],
        dataset_configs: c.Sequence["DatasetConfig"],
    ) -> dict["ModelConfig", list["DatasetConfig"]]:
        """Create mapping from model configs to dataset configs.

        Args:
            model_configs:
                The model configurations.
            dataset_configs:
                The dataset configurations.

        Returns:
            A mapping from model configs to dataset configs.
        """
        return {
            model_config: [
                ds_config
                for ds_config in dataset_configs
                if model_config.model_type in ds_config.allowed_model_types
            ]
            for model_config in model_configs
        }

    def _download(
        self,
        dataset_config: "DatasetConfig",
        model_config: "ModelConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> None:
        """Download data, metrics, and model for the given dataset, and model.

        Args:
            dataset_config: The configuration for the dataset.
            model_config: The configuration for the model.
            benchmark_config: The configuration for the benchmark.
        """
        log_once(
            f"Loading data for {dataset_config.logging_string}", level=logging.INFO
        )
        dataset = load_raw_data(
            dataset_config=dataset_config,
            cache_dir=benchmark_config.cache_dir,
            api_key=benchmark_config.api_key,
        )
        del dataset

        # Skip download if model is a local path
        if not Path(model_config.model_id).exists():
            # Check if model is already cached before downloading
            cache_path = Path(model_config.model_cache_dir)
            has_cached = cache_path.exists() and any(cache_path.rglob("*.safetensors"))
            if has_cached:
                log_once(
                    f"Model {model_config.model_id!r} is already cached, skipping "
                    "download.",
                    level=logging.DEBUG,
                )
            else:
                log_once(
                    f"Downloading model {model_config.model_id!r}...",
                    level=logging.INFO,
                )
                snapshot_download(
                    repo_id=model_config.model_id,
                    revision=model_config.revision,
                    cache_dir=model_config.model_cache_dir,
                    token=get_hf_token(api_key=benchmark_config.api_key),
                )

            # For adapter models, also download the base model
            if model_config.adapter_base_model_id:
                base_id = model_config.adapter_base_model_id
                log_once(
                    f"Downloading adapter base model {base_id!r}...", level=logging.INFO
                )
                snapshot_download(
                    repo_id=model_config.adapter_base_model_id,
                    revision="main",
                    cache_dir=model_config.model_cache_dir,
                    token=get_hf_token(api_key=benchmark_config.api_key),
                )
        else:
            log_once(
                f"Model {model_config.model_id!r} is a local path, skipping download",
                level=logging.INFO,
            )

        log_once(
            f"Loading metrics for the '{dataset_config.task.name}' task",
            level=logging.INFO,
        )
        for metric_name in dataset_config.task.metrics:
            log_once(f"Loading metric {metric_name.name}", level=logging.DEBUG)
            metric = metric_name.download(
                cache_dir=benchmark_config.cache_dir, dataset_config=dataset_config
            )
            del metric

    def _fetch_model_configs(
        self, model_ids: c.Sequence[str], benchmark_config: "BenchmarkConfig"
    ) -> list["ModelConfig"]:
        """Fetch model configurations.

        Args:
            model_ids:
                The model IDs to fetch.
            benchmark_config:
                The benchmark configuration.

        Returns:
            A list of model configurations.
        """
        configs: list["ModelConfig"] = []
        for model_id in get_pbar(
            iterable=model_ids,
            desc="Fetching model configurations",
            disable=not benchmark_config.verbose or not benchmark_config.progress_bar,
        ):
            try:
                configs.append(
                    get_model_config(
                        model_id=model_id, benchmark_config=benchmark_config
                    )
                )
            except InvalidModel as e:
                log(e.message, level=logging.ERROR)
        return configs

    def _prepare_shot_benchmarks(
        self,
        model_config: "ModelConfig",
        datasets: c.Sequence["DatasetConfig"],
        benchmark_config: "BenchmarkConfig",
        existing_results: c.Sequence[BenchmarkResult],
    ) -> tuple[
        "BenchmarkModule | None",
        list[tuple[ShotMode, "DatasetConfig"]],
        list[BenchmarkResult],
        InvalidModel | None,
    ]:
        """Resolve shot modes, load a model once, and filter cached work.

        Returns:
            The loaded model, pending concrete mode/dataset pairs, cached records, and
            a model-loading error if loading failed.
        """
        requested_mode = benchmark_config.few_shot
        modes = resolve_shot_modes(
            model_config=model_config,
            requested_mode=requested_mode,
            generative_type=benchmark_config.generative_type,
        )
        if benchmark_config.download_only:
            modes = modes[:1]

        def mode_pairs_for(
            candidate_modes: c.Sequence[ShotMode],
        ) -> list[tuple[ShotMode, "DatasetConfig"]]:
            pairs: list[tuple[ShotMode, "DatasetConfig"]] = []
            seen_pairs: set[tuple[int, ShotMode]] = set()
            for mode in candidate_modes:
                for dataset_config in datasets:
                    concrete_mode = _shot_mode_for_dataset(mode, dataset_config)
                    if concrete_mode is None:
                        continue
                    pair_key = (id(dataset_config), concrete_mode)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    pairs.append((concrete_mode, dataset_config))
            return pairs

        def records_for(
            mode_pairs: c.Sequence[tuple[ShotMode, "DatasetConfig"]],
        ) -> list[BenchmarkResult]:
            records: list[BenchmarkResult] = []
            for mode, dataset_config in mode_pairs:
                record = get_record(
                    model_config=model_config,
                    dataset_config=dataset_config,
                    benchmark_config=benchmark_config,
                    benchmark_results=existing_results,
                    shot_mode=mode,
                )
                if (
                    record is not None
                    and not benchmark_config.force
                    and record not in records
                ):
                    records.append(record)
            return records

        def pending_for(
            mode_pairs: c.Sequence[tuple[ShotMode, "DatasetConfig"]],
        ) -> list[tuple[ShotMode, "DatasetConfig"]]:
            return [
                (mode, dataset_config)
                for mode, dataset_config in mode_pairs
                if benchmark_config.force
                or get_record(
                    model_config=model_config,
                    dataset_config=dataset_config,
                    benchmark_config=benchmark_config,
                    benchmark_results=existing_results,
                    shot_mode=mode,
                )
                is None
            ]

        mode_pairs = mode_pairs_for(modes)
        provisional_pending = pending_for(mode_pairs)
        provisional_cached = records_for(mode_pairs)
        auto_requested = requested_mode is None or requested_mode is ShotMode.AUTO
        cached_type = (
            _cached_generative_type(provisional_cached)
            if auto_requested and model_config.model_type == ModelType.GENERATIVE
            else None
        )
        if cached_type is not None:
            modes = resolve_shot_modes(
                model_config=model_config,
                requested_mode=requested_mode,
                generative_type=cached_type,
            )
            mode_pairs = mode_pairs_for(modes)
            provisional_pending = pending_for(mode_pairs)
            provisional_cached = records_for(mode_pairs)

        loaded_model: "BenchmarkModule | None" = None
        needs_load = (
            model_config.model_type == ModelType.GENERATIVE
            and not benchmark_config.download_only
            and (bool(provisional_pending) or cached_type is None and auto_requested)
        )
        if needs_load:
            first_mode, first_dataset = (provisional_pending or mode_pairs)[0]
            try:
                loaded_model = load_model(
                    model_config=model_config,
                    dataset_config=first_dataset,
                    benchmark_config=replace(
                        benchmark_config, few_shot=first_mode is ShotMode.FEW_SHOT
                    ),
                )
            except InvalidModel as error:
                cached_on_error = (
                    provisional_cached
                    if provisional_pending or cached_type is not None
                    else []
                )
                return None, provisional_pending, cached_on_error, error

        if not needs_load:
            return None, provisional_pending, provisional_cached, None

        actual_modes = resolve_shot_modes(
            model_config=model_config,
            requested_mode=requested_mode,
            generative_type=loaded_model.generative_type if loaded_model else None,
        )
        if benchmark_config.download_only:
            actual_modes = actual_modes[:1]
        actual_pairs = mode_pairs_for(actual_modes)
        pending = pending_for(actual_pairs)
        cached = records_for(actual_pairs)
        return loaded_model, pending, cached, None

    def _filter_existing_benchmarks(
        self,
        model_mapping: dict["ModelConfig", list["DatasetConfig"]],
        benchmark_config: "BenchmarkConfig",
        existing_results: c.Sequence[BenchmarkResult],
    ) -> tuple[dict["ModelConfig", list["DatasetConfig"]], list[BenchmarkResult]]:
        """Filter out already-benchmarked model-dataset pairs.

        Args:
            model_mapping:
                The model to dataset mapping.
            benchmark_config:
                The benchmark configuration.
            existing_results:
                The existing benchmark results.

        Returns:
            A tuple of (updated model mapping, current results).
        """
        current_results: list[BenchmarkResult] = []
        for model_config, ds_configs in model_mapping.items():
            new_ds_configs: list["DatasetConfig"] = []
            for ds_config in ds_configs:
                record = get_record(
                    model_config=model_config,
                    dataset_config=ds_config,
                    benchmark_config=benchmark_config,
                    benchmark_results=existing_results,
                )
                if record is not None and not benchmark_config.force:
                    current_results.append(record)
                else:
                    new_ds_configs.append(ds_config)
            model_mapping[model_config] = new_ds_configs
        return model_mapping, current_results

    def _generate_summary_message(
        self, finished: int, skipped: int, errored: int
    ) -> str | None:
        """Generate summary message.

        Args:
            finished:
                The number of finished benchmarks.
            skipped:
                The number of skipped benchmarks.
            errored:
                The number of errored benchmarks.

        Returns:
            The summary message, or None if no benchmarks were run.
        """
        parts: list[str] = []
        if finished:
            parts.append(f"completed {finished:,} benchmarks")
        if skipped:
            parts.append(f"skipped {skipped:,} benchmarks")
        if errored:
            parts.append(f"errored {errored:,} benchmarks")
        if not parts:
            return None
        parts[0] = parts[0].capitalize()
        if len(parts) > 1:
            parts[-1] = "and " + parts[-1]
        return "\n" + ", ".join(parts)

    def _handle_benchmark_result(
        self,
        result_or_error: BenchmarkResult | Exception,
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
        num_finished: int,
        num_skipped: int,
        num_errored: int,
        model_config: "ModelConfig",
        model_mapping: dict["ModelConfig", list["DatasetConfig"]],
        current_results: list[BenchmarkResult],
        remaining_work: int,
    ) -> tuple[int, int, int, bool]:
        """Handle benchmark result.

        Args:
            result_or_error:
                The benchmark result or exception.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.
            num_finished:
                The number of finished benchmarks.
            num_skipped:
                The number of skipped benchmarks.
            num_errored:
                The number of errored benchmarks.
            model_config:
                The model configuration.
            model_mapping:
                The model to dataset mapping.
            current_results:
                The current benchmark results.
            remaining_work:
                The number of concrete mode/dataset pairs after this one.

        Returns:
            A tuple of (updated finished, skipped, errored counters, break flag).
        """
        if isinstance(result_or_error, Exception) and benchmark_config.raise_errors:
            raise result_or_error
        if isinstance(result_or_error, InvalidBenchmark):
            log(result_or_error.message, level=logging.WARNING)
            if dataset_config.task.name in ORTHOGONAL_TASKS:
                num_skipped += 1
            else:
                num_errored += 1
            return num_finished, num_skipped, num_errored, False
        if isinstance(result_or_error, InvalidModel):
            log(result_or_error.message, level=logging.WARNING)
            num_errored += 1 + remaining_work
            return num_finished, num_skipped, num_errored, True
        assert isinstance(result_or_error, BenchmarkResult)
        record: BenchmarkResult = result_or_error
        current_results.append(record)
        if benchmark_config.save_results:
            record.append_to_results(results_path=self.results_path)
        num_finished += 1
        return num_finished, num_skipped, num_errored, False

    def _prepare_model_ids(self, model_id: c.Sequence[str] | str) -> c.Sequence[str]:
        """Prepare the model ID(s) to be benchmarked.

        Args:
            model_id:
                The model ID(s) of the models to benchmark.

        Returns:
            The prepared list of model IDs.
        """
        model_ids = [model_id] if isinstance(model_id, str) else model_id

        # Reorder the `model_ids` list to include the ones present in the benchmark
        # results first
        benchmarked_model_ids = [
            re.sub(r"\(.+\)", "", record.model).strip()
            for record in self.benchmark_results
        ]
        model_ids_sorted = [m_id for m_id in model_ids if m_id in benchmarked_model_ids]
        model_ids_sorted += [
            m_id for m_id in model_ids if m_id not in benchmarked_model_ids
        ]

        return [m_id.rstrip(" /") for m_id in model_ids_sorted]

    def _update_benchmark_config_for_dataset(
        self, dataset_config: "DatasetConfig", benchmark_config: "BenchmarkConfig"
    ) -> dict[str, t.Any]:
        """Update benchmark config for dataset.

        Args:
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.

        Returns:
            A dictionary of parameters to revert.
        """
        params_to_revert: dict[str, t.Any] = {}
        if (
            dataset_config.val_split is None
            and not benchmark_config.evaluate_test_split
        ):
            log(
                "The dataset does not have a validation split, so even though "
                "you requested evaluating the validation split (the default), "
                "we will evaluate on the test split.",
                level=logging.DEBUG,
            )
            params_to_revert["evaluate_test_split"] = False
            benchmark_config.evaluate_test_split = True
        return params_to_revert

    @property
    def benchmark_results(self) -> c.Sequence[BenchmarkResult]:
        """The benchmark results.

        Returns:
            A list of benchmark results.
        """
        return BenchmarkResult.from_jsonl(self.results_path)


def clear_model_cache_fn(cache_dir: str) -> None:
    """Clear the model cache.

    Note that this will not remove the stored completions.

    Args:
        cache_dir:
            The path to the cache directory.
    """
    model_cache_path = Path(cache_dir) / "model_cache"
    model_cache_path.mkdir(parents=True, exist_ok=True)
    for model_dir in model_cache_path.iterdir():
        if model_dir.is_dir():
            for sub_model_dir in model_dir.iterdir():
                if sub_model_dir.is_dir():
                    rmtree(sub_model_dir, ignore_errors=True)


def resolve_shot_modes(
    model_config: "ModelConfig",
    requested_mode: ShotMode | bool | None,
    generative_type: GenerativeType | None = None,
) -> list[ShotMode]:
    """Resolve a requested shot mode into concrete evaluation modes.

    Args:
        model_config:
            The model configuration.
        requested_mode:
            ``ShotMode.AUTO`` (or ``None``/the legacy boolean API) selects modes
            automatically. Booleans remain accepted for programmatic compatibility.
        generative_type:
            The detected model type, when the model has already been loaded.

    Returns:
        The concrete modes to evaluate, in execution order.
    """
    if requested_mode is True:
        requested_mode = ShotMode.FEW_SHOT
    elif requested_mode is False:
        requested_mode = ShotMode.ZERO_SHOT
    elif requested_mode is None:
        requested_mode = ShotMode.AUTO

    if requested_mode is not ShotMode.AUTO:
        return [requested_mode]
    if model_config.model_type != ModelType.GENERATIVE:
        return [ShotMode.FEW_SHOT]
    if model_config.inference_backend == InferenceBackend.LITELLM:
        return [ShotMode.ZERO_SHOT]
    if generative_type == GenerativeType.BASE:
        return [ShotMode.FEW_SHOT]
    if generative_type in (GenerativeType.INSTRUCTION_TUNED, GenerativeType.REASONING):
        return [ShotMode.ZERO_SHOT, ShotMode.FEW_SHOT]
    # The model type is only available after loading local generative models. Be
    # conservative before loading so neither mode can suppress the other in the cache.
    return [ShotMode.ZERO_SHOT, ShotMode.FEW_SHOT]


def _shot_mode_for_dataset(
    shot_mode: ShotMode, dataset_config: "DatasetConfig"
) -> ShotMode | None:
    """Return the concrete mode for a dataset, or None when it is redundant."""
    if dataset_config.task.requires_zero_shot:
        return (
            ShotMode.ZERO_SHOT
            if shot_mode in (ShotMode.AUTO, ShotMode.FEW_SHOT)
            else shot_mode
        )
    return shot_mode


def _cached_generative_type(
    records: c.Sequence[BenchmarkResult],
) -> GenerativeType | None:
    """Return reliable generative metadata shared by cached records.

    Older records do not include ``generative_type``.  Treat a mixture of metadata as
    unknown rather than allowing one record to select an incomplete AUTO plan.
    """
    if not records or any(not record.generative for record in records):
        return None
    try:
        types = {GenerativeType(record.generative_type) for record in records}
    except (TypeError, ValueError):
        return None
    return types.pop() if len(types) == 1 else None


def get_record(
    model_config: "ModelConfig",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
    benchmark_results: c.Sequence[BenchmarkResult],
    shot_mode: ShotMode | bool | None = None,
) -> BenchmarkResult | None:
    """Get the cached record for a model, dataset and concrete shot mode.

    Args:
        model_config:
            The configuration of the model we are evaluating.
        dataset_config:
            The configuration of the dataset we are evaluating on.
        benchmark_config:
            The general benchmark configuration.
        benchmark_results:
            The benchmark results.
        shot_mode:
            The concrete shot mode to match. ``None`` keeps the legacy behaviour by
            using ``benchmark_config.few_shot``.

    Returns:
        The benchmark record, or None if no such record exists.
    """
    for record in benchmark_results:
        model_id_components = split_model_id(model_id=record.model)
        same_model_id = model_id_components.model_id == model_config.model_id
        same_revision = model_id_components.revision == model_config.revision
        same_param = model_id_components.param == model_config.param
        same_dataset = record.dataset == dataset_config.name
        same_split = record.validation_split != benchmark_config.evaluate_test_split
        requested_mode = shot_mode
        if requested_mode is None:
            requested_mode = benchmark_config.few_shot
        if requested_mode is True:
            requested_mode = ShotMode.FEW_SHOT
        elif requested_mode is False:
            requested_mode = ShotMode.ZERO_SHOT
        elif requested_mode is None:
            requested_mode = ShotMode.AUTO
        same_num_shots = (
            not record.generative
            or requested_mode is ShotMode.AUTO
            or (
                dataset_config.task.requires_zero_shot
                and requested_mode is ShotMode.ZERO_SHOT
                and record.few_shot in (False, None)
            )
            or (requested_mode is ShotMode.FEW_SHOT and record.few_shot is True)
            or (
                requested_mode is ShotMode.ZERO_SHOT
                and record.few_shot in (False, None)
            )
        )
        if (
            same_model_id
            and same_revision
            and same_param
            and same_dataset
            and same_split
            and same_num_shots
        ):
            return record
    return None


def initial_logging(
    model_config: "ModelConfig",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
    num_finished_benchmarks: int,
    num_total_benchmarks: int,
) -> None:
    """Initial logging at the start of the benchmarking process.

    Args:
        model_config:
            The configuration of the model we are evaluating.
        dataset_config:
            The configuration of the dataset we are evaluating on.
        benchmark_config:
            The general benchmark configuration.
        num_finished_benchmarks:
            The number of benchmarks that have already been finished.
        num_total_benchmarks:
            The total number of benchmarks to be run.
    """
    model_id = model_config.model_id
    if model_config.revision and model_config.revision != "main":
        model_id += f"@{model_config.revision}"
    if model_config.param is not None:
        model_id += f"#{model_config.param}"

    split_type = "validation" if not benchmark_config.evaluate_test_split else "test"
    if model_config.task in GENERATIVE_PIPELINE_TAGS:
        if benchmark_config.few_shot:
            eval_type = "Few-shot benchmarking"
        else:
            eval_type = "Zero-shot benchmarking"
    else:
        eval_type = "Benchmarking"

    log_once(
        f"\n{eval_type} {model_id} on the {split_type} split of "
        f"{dataset_config.logging_string} ({num_finished_benchmarks + 1}/"
        f"{num_total_benchmarks} benchmarks)...",
        prefix=f"\n[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]",
        level=logging.INFO,
    )

    if dataset_config.unofficial:
        log_once(
            f"Note that the {dataset_config.name!r} dataset is unofficial, "
            "meaning that the resulting evaluation will not be included in the "
            "official leaderboard.",
            level=logging.WARNING,
        )

    if benchmark_config.debug:
        log_once(
            "Running in debug mode. This will output additional information, as "
            "well as store the model outputs in the current directory after each "
            "batch. For this reason, evaluation will be slower.",
            level=logging.WARNING,
        )
