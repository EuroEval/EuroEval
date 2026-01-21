"""Class that benchmarks language models."""

import collections.abc as c
import contextlib
import datetime as dt
import json
import logging
import os
import re
import typing as t
from pathlib import Path
from shutil import rmtree
from time import sleep

from torch.distributed import destroy_process_group

from .benchmark_config_factory import build_benchmark_config
from .constants import GENERATIVE_PIPELINE_TAGS
from .data_loading import load_data, load_raw_data
from .data_models import BenchmarkConfigParams, BenchmarkResult
from .dataset_configs import get_all_dataset_configs
from .enums import Device, GenerativeType, ModelType
from .exceptions import HuggingFaceHubDown, InvalidBenchmark, InvalidModel
from .finetuning import finetune
from .generation import generate
from .logging_utils import adjust_logging_level, get_pbar, log, log_once
from .model_config import get_model_config
from .model_loading import load_model
from .scores import log_scores
from .speed_benchmark import benchmark_speed
from .tasks import SPEED
from .utils import (
    enforce_reproducibility,
    internet_connection_available,
    split_model_id,
)

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
        few_shot: bool = True,
        num_iterations: int = 10,
        api_base: str | None = None,
        api_version: str | None = None,
        gpu_memory_utilization: float = 0.8,
        generative_type: GenerativeType | None = None,
        custom_datasets_file: Path | str = Path("custom_datasets.py"),
        debug: bool = False,
        run_with_cli: bool = False,
        requires_safetensors: bool = False,
        download_only: bool = False,
        model_language: str | c.Sequence[str] | None = None,
        dataset_language: str | c.Sequence[str] | None = None,
        batch_size: int | None = None,
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
                `task` and `dataset` are None then all datasets will be benchmarked.
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
                Whether to only evaluate the model using few-shot evaluation. Only
                relevant if the model is generative. Defaults to True.
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
            generative_type:
                The type of generative model to benchmark. Only relevant if the model is
                generative. If not specified, then the type will be inferred based on
                the tags of the model. Defaults to None.
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
            model_language:
                Deprecated argument. Please use `language` instead.
            dataset_language:
                Deprecated argument. Please use `language` instead.
            batch_size:
                Deprecated argument. Please use `finetuning_batch_size` instead.

        Raises:
            ValueError:
                If both `task` and `dataset` are specified, or if `download_only`
                is True and we have no internet connection.
            ImportError:
                If `hf_transfer` is enabled but not installed.
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

        # Deprecation warnings
        if batch_size is not None:
            if run_with_cli:
                msg = (
                    "The --batch-size option is deprecated and will be removed in a "
                    "future version. Please use --finetuning-batch-size instead. "
                    "Overwriting --finetuning-batch-size with the value from "
                    "--batch-size."
                )
            else:
                msg = (
                    "The `batch_size` argument is deprecated and will be removed in a "
                    "future version. Please use `finetuning_batch_size` instead. "
                    "Overwriting `finetuning_batch_size` with the value from "
                    "`batch_size`."
                )
            log(msg, level=logging.WARNING)
            finetuning_batch_size = batch_size
        if model_language is not None:
            if run_with_cli:
                msg = (
                    "The --model-language option is deprecated and will be removed in "
                    "a future version. Please use --language instead. Ignoring the "
                    "--model-language value."
                )
            else:
                msg = (
                    "The `model_language` argument is deprecated and will be removed "
                    "in a future version. Please use `language` instead. Ignoring the "
                    "`model_language` value."
                )
            log(msg, level=logging.WARNING)
        if dataset_language is not None:
            if run_with_cli:
                msg = (
                    "The --dataset-language option is deprecated and will be removed "
                    "in a future version. Please use --language instead. Ignoring the "
                    "--dataset-language value."
                )
            else:
                msg = (
                    "The `dataset_language` argument is deprecated and will be removed "
                    "in a future version. Please use `language` instead. Ignoring the "
                    "`dataset_language` value."
                )
            log(msg, level=logging.WARNING)

        # If FULL_LOG has been set, then force verbose mode
        if os.getenv("FULL_LOG", "0") == "1":
            verbose = True

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
            generative_type=generative_type,
            custom_datasets_file=Path(custom_datasets_file),
            verbose=verbose,
            force=force,
            debug=debug,
            run_with_cli=run_with_cli,
        )

        self.benchmark_config = build_benchmark_config(
            benchmark_config_params=self.benchmark_config_default_params
        )

        # Initialise variable storing model lists, so we only have to fetch it once
        self._model_lists: dict[str, c.Sequence[str]] | None = None

        self.results_path = Path.cwd() / "euroeval_benchmark_results.jsonl"
        adjust_logging_level(verbose=self.benchmark_config.verbose)

    @property
    def benchmark_results(self) -> c.Sequence[BenchmarkResult]:
        """The benchmark results.

        Returns:
            A list of benchmark results.

        Raises:
            ValueError:
                If there is an error decoding a line in the results file.
        """
        if self.results_path.exists():
            benchmark_results: list[BenchmarkResult] = list()
            with self.results_path.open() as f:
                for line in f:
                    if line.strip():
                        try:
                            result_dict = json.loads(line.strip())
                        except json.JSONDecodeError as e:
                            raise ValueError(
                                f"Error decoding JSON line: {line.strip()}"
                            ) from e

                        # Fix for older records
                        has_old_raw_results = (
                            "results" in result_dict
                            and isinstance(result_dict["results"], dict)
                            and "raw" in result_dict["results"]
                            and isinstance(result_dict["results"]["raw"], dict)
                            and "test" in result_dict["results"]["raw"]
                        )
                        if has_old_raw_results:
                            result_dict["results"]["raw"] = result_dict["results"][
                                "raw"
                            ]["test"]

                        result = BenchmarkResult.from_dict(result_dict)
                        benchmark_results.append(result)
            return benchmark_results
        else:
            return list()

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
            dataset_config=dataset_config, cache_dir=benchmark_config.cache_dir
        )
        del dataset

        model = load_model(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )
        del model

        log_once(
            f"Loading metrics for the '{dataset_config.task.name}' task",
            level=logging.INFO,
        )
        for metric_name in dataset_config.task.metrics:
            log_once(f"Loading metric {metric_name.name}", level=logging.DEBUG)
            metric = metric_name.download(cache_dir=benchmark_config.cache_dir)
            del metric

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
        custom_datasets_file: Path | str | None = None,
        force: bool | None = None,
        verbose: bool | None = None,
        debug: bool | None = None,
        model_language: str | c.Sequence[str] | None = None,
        dataset_language: str | c.Sequence[str] | None = None,
        batch_size: int | None = None,
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
                datasets. Here 'no' means both Bokmål (nb) and Nynorsk (nn). Set this to
                'all' if all languages should be considered. Defaults to the value
                specified when initialising the benchmarker.
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
                Whether to only evaluate the model using few-shot evaluation. Only
                relevant if the model is generative. Defaults to the value specified
                when initialising the benchmarker.
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
            model_language:
                Deprecated argument. Please use `language` instead.
            dataset_language:
                Deprecated argument. Please use `language` instead.
            batch_size:
                Deprecated argument. Please use `finetuning_batch_size` instead.

        Returns:
            A list of benchmark results.

        Raises:
            ValueError:
                If both `task` and `dataset` are specified.
        """
        if task is not None and dataset is not None:
            raise ValueError("Only one of `task` and `dataset` can be specified.")

        # Deprecation warnings
        if batch_size is not None:
            log(
                "The `batch_size` argument is deprecated and will be removed in a "
                "future version. Please use `finetuning_batch_size` instead. "
                "Overwriting `finetuning_batch_size` with the value from "
                "`batch_size`.",
                level=logging.WARNING,
            )
            finetuning_batch_size = batch_size
        if model_language is not None:
            log(
                "The `model_language` argument is deprecated and will be removed "
                "in a future version. Please use `language` instead. Ignoring the "
                "`model_language` value.",
                level=logging.WARNING,
            )
        if dataset_language is not None:
            log(
                "The `dataset_language` argument is deprecated and will be removed "
                "in a future version. Please use `language` instead. Ignoring the "
                "`dataset_language` value.",
                level=logging.WARNING,
            )

        # Get a new updated benchmark configuration, based on any changes to the
        # parameters
        benchmark_config_params = BenchmarkConfigParams(
            task=(
                task if task is not None else self.benchmark_config_default_params.task
            ),
            dataset=(
                dataset
                if dataset is not None
                else self.benchmark_config_default_params.dataset
            ),
            progress_bar=(
                progress_bar
                if progress_bar is not None
                else self.benchmark_config_default_params.progress_bar
            ),
            save_results=(
                save_results
                if save_results is not None
                else self.benchmark_config_default_params.save_results
            ),
            language=(
                language
                if language is not None
                else self.benchmark_config_default_params.language
            ),
            device=(
                device
                if device is not None
                else self.benchmark_config_default_params.device
            ),
            finetuning_batch_size=(
                finetuning_batch_size
                if finetuning_batch_size is not None
                else self.benchmark_config_default_params.finetuning_batch_size
            ),
            raise_errors=(
                raise_errors
                if raise_errors is not None
                else self.benchmark_config_default_params.raise_errors
            ),
            cache_dir=(
                cache_dir
                if cache_dir is not None
                else self.benchmark_config_default_params.cache_dir
            ),
            api_key=(
                api_key
                if api_key is not None
                else self.benchmark_config_default_params.api_key
            ),
            api_base=(
                api_base
                if api_base is not None
                else self.benchmark_config_default_params.api_base
            ),
            api_version=(
                api_version
                if api_version is not None
                else self.benchmark_config_default_params.api_version
            ),
            trust_remote_code=(
                trust_remote_code
                if trust_remote_code is not None
                else self.benchmark_config_default_params.trust_remote_code
            ),
            clear_model_cache=(
                clear_model_cache
                if clear_model_cache is not None
                else self.benchmark_config_default_params.clear_model_cache
            ),
            evaluate_test_split=(
                evaluate_test_split
                if evaluate_test_split is not None
                else self.benchmark_config_default_params.evaluate_test_split
            ),
            few_shot=(
                few_shot
                if few_shot is not None
                else self.benchmark_config_default_params.few_shot
            ),
            num_iterations=(
                num_iterations
                if num_iterations is not None
                else self.benchmark_config_default_params.num_iterations
            ),
            requires_safetensors=(
                requires_safetensors
                if requires_safetensors is not None
                else self.benchmark_config_default_params.requires_safetensors
            ),
            download_only=(
                download_only
                if download_only is not None
                else self.benchmark_config_default_params.download_only
            ),
            gpu_memory_utilization=(
                gpu_memory_utilization
                if gpu_memory_utilization is not None
                else self.benchmark_config_default_params.gpu_memory_utilization
            ),
            generative_type=(
                generative_type
                if generative_type is not None
                else self.benchmark_config_default_params.generative_type
            ),
            custom_datasets_file=(
                Path(custom_datasets_file)
                if custom_datasets_file is not None
                else self.benchmark_config_default_params.custom_datasets_file
            ),
            force=(
                force
                if force is not None
                else self.benchmark_config_default_params.force
            ),
            verbose=(
                verbose
                if verbose is not None
                else self.benchmark_config_default_params.verbose
            ),
            debug=(
                debug
                if debug is not None
                else self.benchmark_config_default_params.debug
            ),
            run_with_cli=self.benchmark_config_default_params.run_with_cli,
        )
        benchmark_config = build_benchmark_config(
            benchmark_config_params=benchmark_config_params
        )

        adjust_logging_level(verbose=benchmark_config.verbose)

        if benchmark_config.clear_model_cache:
            clear_model_cache_fn(cache_dir=benchmark_config.cache_dir)

        model_ids = self._prepare_model_ids(model_id=model)
        dataset_configs = benchmark_config.datasets

        # Get all the model configs
        model_configs: list["ModelConfig"] = list()
        for model_id in get_pbar(
            iterable=model_ids,
            desc="Fetching model configurations",
            disable=not benchmark_config.verbose or not benchmark_config.progress_bar,
        ):
            try:
                model_config = get_model_config(
                    model_id=model_id, benchmark_config=benchmark_config
                )
                model_configs.append(model_config)
            except InvalidModel as e:
                log(e.message, level=logging.ERROR)

        # Create a dictionary that takes each model config to the dataset configs that
        # we need to benchmark the model on. We initially include all the relevant
        # datasets for each model.
        model_config_to_dataset_configs: dict[
            "ModelConfig", c.Sequence["DatasetConfig"]
        ] = {
            model_config: [
                dataset_config
                for dataset_config in dataset_configs
                if model_config.model_type in dataset_config.allowed_model_types
            ]
            for model_config in model_configs
        }

        # Initialise the current benchmark results with all the ones that we have cached
        # on disk already (can be none), and remove those datasets from the mapping
        current_benchmark_results: list[BenchmarkResult] = list()
        for (
            model_config,
            model_dataset_configs,
        ) in model_config_to_dataset_configs.items():
            new_model_dataset_configs: list["DatasetConfig"] = list()
            for dataset_config in model_dataset_configs:
                benchmark_record = get_record(
                    model_config=model_config,
                    dataset_config=dataset_config,
                    benchmark_config=benchmark_config,
                    benchmark_results=self.benchmark_results,
                )
                if benchmark_record is not None and not benchmark_config.force:
                    current_benchmark_results.append(benchmark_record)
                else:
                    new_model_dataset_configs.append(dataset_config)
            model_config_to_dataset_configs[model_config] = new_model_dataset_configs

        total_benchmarks = sum(
            len(dataset_configs)
            for dataset_configs in model_config_to_dataset_configs.values()
        )
        if total_benchmarks == 0:
            log(
                "No benchmarks to run, as all the selected models have already been "
                "benchmarked on all the selected datasets.",
                level=logging.INFO,
            )
            return current_benchmark_results

        num_finished_benchmarks = 0
        benchmark_params_to_revert: dict[str, t.Any] = dict()
        for model_config in model_configs:
            if not model_config_to_dataset_configs[model_config]:
                log(
                    f"Skipping model {model_config.model_id!r} because it has "
                    "already been benchmarked on all valid datasets.",
                    level=logging.DEBUG,
                )
                continue

            if model_config.adapter_base_model_id:
                open_issue_msg = (
                    "If offline support is important to you, please consider opening "
                    "an issue at https://github.com/EuroEval/EuroEval/issues."
                )
                if not internet_connection_available():
                    raise InvalidModel(
                        "Offline benchmarking of models with adapters is not currently "
                        "supported. An active internet connection is required. "
                        "{open_issue_msg}"
                    )
                elif benchmark_config.download_only:
                    log_once(
                        "You are using download only mode with a model that includes "
                        "an adapter. Please note that offline benchmarking of "
                        "adapter models is not currently supported - an internet "
                        "connection will be required during evaluation in this case. "
                        f"{open_issue_msg}",
                        level=logging.WARNING,
                    )

            loaded_model: "BenchmarkModule | None" = None
            for dataset_config in model_config_to_dataset_configs[model_config]:
                # Revert any changes to the benchmark configuration made for the
                # previous dataset
                for param, value in benchmark_params_to_revert.items():
                    setattr(benchmark_config, param, value)
                benchmark_params_to_revert = dict()

                # Update the benchmark config if the dataset requires it
                if (
                    "val" not in dataset_config.splits
                    and not benchmark_config.evaluate_test_split
                ):
                    log(
                        "The dataset does not have a validation split, so even though "
                        "you requested evaluating the validation split (the default), "
                        "we will evaluate on the test split.",
                        level=logging.DEBUG,
                    )
                    benchmark_params_to_revert["evaluate_test_split"] = False
                    benchmark_config.evaluate_test_split = True
                if dataset_config.task.requires_zero_shot and benchmark_config.few_shot:
                    log(
                        "The task requires zero-shot evaluation, so even though you "
                        "requested few-shot evaluation (the default), we will evaluate "
                        "zero-shot.",
                        level=logging.DEBUG,
                    )
                    benchmark_params_to_revert["few_shot"] = True
                    benchmark_config.few_shot = False

                # We do not re-initialise generative models as their architecture is not
                # customised to specific datasets
                if model_config.model_type == ModelType.GENERATIVE:
                    if loaded_model is None:
                        try:
                            loaded_model = load_model(
                                model_config=model_config,
                                dataset_config=dataset_config,
                                benchmark_config=benchmark_config,
                            )
                        except InvalidModel as e:
                            if benchmark_config.raise_errors:
                                raise e
                            log(e.message, level=logging.ERROR)

                            # Add the remaining number of benchmarks for the model to
                            # our benchmark counter, since we're skipping the rest of
                            # them
                            num_finished_benchmarks += (
                                len(dataset_configs)
                                - dataset_configs.index(dataset_config)
                                - 1
                            )
                            break
                    else:
                        loaded_model.dataset_config = dataset_config

                    # Skip the benchmark if the model is not of the correct
                    # generative type
                    if (
                        loaded_model.generative_type
                        not in dataset_config.allowed_generative_types
                    ):
                        log(
                            f"Skipping the benchmark of model "
                            f"{model_config.model_id!r}on dataset "
                            f"{dataset_config.name!r} because the model has generative "
                            f"type {loaded_model.generative_type} and the dataset "
                            f"only allows {dataset_config.allowed_generative_types}.",
                            level=logging.DEBUG,
                        )
                        num_finished_benchmarks += 1
                        continue

                # Benchmark a single model on a single dataset
                benchmark_output_or_err = self._benchmark_single(
                    model=loaded_model,
                    model_config=model_config,
                    dataset_config=dataset_config,
                    benchmark_config=benchmark_config,
                    num_finished_benchmarks=num_finished_benchmarks,
                    num_total_benchmarks=total_benchmarks,
                )

                if (
                    isinstance(benchmark_output_or_err, Exception)
                    and benchmark_config.raise_errors
                ):
                    raise benchmark_output_or_err

                elif isinstance(benchmark_output_or_err, InvalidBenchmark):
                    log(benchmark_output_or_err.message, level=logging.WARNING)
                    num_finished_benchmarks += 1
                    continue

                elif isinstance(benchmark_output_or_err, InvalidModel):
                    log(benchmark_output_or_err.message, level=logging.WARNING)

                    # Add the remaining number of benchmarks for the model to our
                    # benchmark counter, since we're skipping the rest of them
                    num_finished_benchmarks += (
                        len(dataset_configs) - dataset_configs.index(dataset_config) - 1
                    )
                    break

                else:
                    record: BenchmarkResult = benchmark_output_or_err
                    current_benchmark_results.append(record)
                    if benchmark_config.save_results:
                        record.append_to_results(results_path=self.results_path)

                num_finished_benchmarks += 1

            del loaded_model
            if benchmark_config.clear_model_cache:
                clear_model_cache_fn(cache_dir=benchmark_config.cache_dir)

        log(
            f"\nCompleted {num_finished_benchmarks:,} benchmarks.\n", level=logging.INFO
        )

        # This avoids the following warning at the end of the benchmarking:
        #   Warning: WARNING: process group has NOT been destroyed before we destruct
        #   ProcessGroupNCCL. On normal program exit, the application should call
        #   destroy_process_group to ensure that any pending NCCL operations have
        #   finished in this process. In rare cases this process can exit before this
        #   point and block the progress of another member of the process group. This
        #   constraint has always been present,  but this warning has only been added
        #   since PyTorch 2.4 (function operator())
        with contextlib.suppress(AssertionError):
            destroy_process_group()
        return current_benchmark_results

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
                    metrics=dataset_config.task.metrics,
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
                        if "val" not in dataset_config.splits
                        else not benchmark_config.evaluate_test_split
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

    def __call__(self, *args: t.Any, **kwds: t.Any) -> t.Any:  # noqa: ANN401
        """Alias for `self.benchmark()`."""
        log(
            "Calling the `Benchmarker` class directly is deprecated. Please use the "
            "`benchmark` function instead. This will be removed in a future version.",
            level=logging.WARNING,
        )
        return self.benchmark(*args, **kwds)


def get_record(
    model_config: "ModelConfig",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
    benchmark_results: c.Sequence[BenchmarkResult],
) -> BenchmarkResult | None:
    """Get the benchmark record for a given model and dataset.

    Args:
        model_config:
            The configuration of the model we are evaluating.
        dataset_config:
            The configuration of the dataset we are evaluating on.
        benchmark_config:
            The general benchmark configuration.
        benchmark_results:
            The benchmark results.

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
        same_num_shots = (
            record.few_shot == benchmark_config.few_shot
            or record.few_shot is None
            or not record.generative
            or dataset_config.task.requires_zero_shot
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
                    rmtree(sub_model_dir)


def prepare_dataset_configs(
    dataset_names: c.Sequence[str], custom_datasets_file: Path
) -> c.Sequence["DatasetConfig"]:
    """Prepare the dataset configuration(s) to be benchmarked.

    Args:
        dataset_names:
            The dataset names to benchmark.
        custom_datasets_file:
            A path to a Python file containing custom dataset configurations.

    Returns:
        The prepared list of model IDs.
    """
    return [
        cfg
        for cfg in get_all_dataset_configs(
            custom_datasets_file=custom_datasets_file
        ).values()
        if cfg.name in dataset_names
    ]


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
