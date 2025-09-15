"""Class that benchmarks language models."""

import contextlib
import json
import logging
import re
import sys
import typing as t
from pathlib import Path
from shutil import rmtree
from time import sleep

from huggingface_hub.constants import HF_HUB_ENABLE_HF_TRANSFER
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
from .model_config import get_model_config
from .model_loading import load_model
from .scores import log_scores
from .speed_benchmark import benchmark_speed
from .tasks import SPEED
from .utils import (
    enforce_reproducibility,
    get_package_version,
    internet_connection_available,
    log_once,
)

if t.TYPE_CHECKING:
    from .benchmark_modules import BenchmarkModule
    from .data_models import BenchmarkConfig, DatasetConfig, ModelConfig


logger = logging.getLogger("euroeval")


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
        task: str | list[str] | None = None,
        dataset: list[str] | str | None = None,
        language: str | list[str] = "all",
        model_language: str | list[str] | None = None,
        dataset_language: str | list[str] | None = None,
        device: Device | None = None,
        batch_size: int = 32,
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
        gpu_memory_utilization: float = 0.9,
        generative_type: GenerativeType | None = None,
        debug: bool = False,
        run_with_cli: bool = False,
        requires_safetensors: bool = False,
        download_only: bool = False,
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
            model_language:
                The language codes of the languages to include for models. If specified
                then this overrides the `language` parameter for model languages.
                Defaults to None.
            dataset_language:
                The language codes of the languages to include for datasets. If
                specified then this overrides the `language` parameter for dataset
                languages. Defaults to None.
            device:
                The device to use for benchmarking. Defaults to None.
            batch_size:
                The batch size to use. Defaults to 32.
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

        Raises:
            ValueError:
                If both `task` and `dataset` are specified, or if `download_only`
                is True and we have no internet connection.
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

        # Bail early if hf_transfer is enabled but not installed.
        if HF_HUB_ENABLE_HF_TRANSFER and get_package_version("hf_transfer") is None:
            raise ImportError(
                "Fast download using 'hf_transfer' is enabled "
                "(HF_HUB_ENABLE_HF_TRANSFER=1) but the 'hf_transfer' "
                "package is not available in your environment. "
                "Try installing it with `pip install hf_transfer`."
            )

        self.benchmark_config_default_params = BenchmarkConfigParams(
            task=task,
            dataset=dataset,
            progress_bar=progress_bar,
            save_results=save_results,
            language=language,
            model_language=model_language,
            dataset_language=dataset_language,
            device=device,
            batch_size=batch_size,
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
            verbose=verbose,
            force=force,
            debug=debug,
            run_with_cli=run_with_cli,
        )

        self.benchmark_config = build_benchmark_config(
            benchmark_config_params=self.benchmark_config_default_params
        )

        # Initialise variable storing model lists, so we only have to fetch it once
        self._model_lists: dict[str, list[str]] | None = None

        self.results_path = Path.cwd() / "euroeval_benchmark_results.jsonl"
        adjust_logging_level(verbose=self.benchmark_config.verbose)

    @property
    def benchmark_results(self) -> list[BenchmarkResult]:
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
        log_once(f"Loading data for {dataset_config.pretty_name}", level=logging.INFO)
        dataset = load_raw_data(
            dataset_config=dataset_config, cache_dir=benchmark_config.cache_dir
        )
        del dataset

        log_once(f"Loading model {model_config.model_id}", level=logging.INFO)
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
        model: list[str] | str,
        task: str | list[str] | None = None,
        dataset: list[str] | str | None = None,
        progress_bar: bool | None = None,
        save_results: bool | None = None,
        language: str | list[str] | None = None,
        model_language: str | list[str] | None = None,
        dataset_language: str | list[str] | None = None,
        device: Device | None = None,
        batch_size: int | None = None,
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
        force: bool | None = None,
        verbose: bool | None = None,
        debug: bool | None = None,
    ) -> list[BenchmarkResult]:
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
            model_language:
                The language codes of the languages to include for models. If specified
                then this overrides the `language` parameter for model languages.
                Defaults to the value specified when initialising the benchmarker.
            dataset_language:
                The language codes of the languages to include for datasets. If
                specified then this overrides the `language` parameter for dataset
                languages. Defaults to the value specified when initialising the
                benchmarker.
            device:
                The device to use for benchmarking. Defaults to the value specified when
                initialising the benchmarker.
            batch_size:
                The batch size to use. Defaults to the value specified when initialising
                the benchmarker.
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

        Returns:
            A list of benchmark results.

        Raises:
            ValueError:
                If both `task` and `dataset` are specified.
        """
        if task is not None and dataset is not None:
            raise ValueError("Only one of `task` and `dataset` can be specified.")

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
            model_language=(
                model_language
                if model_language is not None
                else self.benchmark_config_default_params.model_language
            ),
            dataset_language=(
                dataset_language
                if dataset_language is not None
                else self.benchmark_config_default_params.dataset_language
            ),
            device=(
                device
                if device is not None
                else self.benchmark_config_default_params.device
            ),
            batch_size=(
                batch_size
                if batch_size is not None
                else self.benchmark_config_default_params.batch_size
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
        dataset_configs = prepare_dataset_configs(
            dataset_names=benchmark_config.datasets
        )

        total_benchmarks = len(model_ids) * len(dataset_configs)
        num_finished_benchmarks = 0

        current_benchmark_results: list[BenchmarkResult] = list()
        for model_id in model_ids:
            # Load the model configuration, or skip the model if it is invalid
            try:
                model_config = get_model_config(
                    model_id=model_id, benchmark_config=benchmark_config
                )
            except InvalidModel as e:
                logger.info(e.message)
                num_finished_benchmarks += len(dataset_configs)
                continue

            if model_config.adapter_base_model_id:
                open_issue_msg = (
                    "If offline support is important to you, please "
                    "consider opening an issue at https://github.com/EuroEval/EuroEval/issues."
                )
                if not internet_connection_available():
                    raise InvalidModel(
                        "Offline benchmarking of models with adapters is not currently "
                        "supported. "
                        f"An active internet connection is required. {open_issue_msg}"
                    )
                elif benchmark_config.download_only:
                    log_once(
                        "You are using download only mode with a model that includes "
                        "an adapter. "
                        "Please note: Offline benchmarking of adapter models is not "
                        "currently supported. "
                        "An internet connection will be required during evaluation. "
                        f"{open_issue_msg}",
                        level=logging.WARNING,
                    )

            loaded_model: BenchmarkModule | None = None
            benchmark_params_to_revert: dict[str, t.Any] = dict()
            for dataset_config in dataset_configs:
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
                    logger.debug(
                        "The dataset does not have a validation split, so even though "
                        "you requested evaluating the validation split (the default), "
                        "we will evaluate on the test split."
                    )
                    benchmark_params_to_revert["evaluate_test_split"] = False
                    benchmark_config.evaluate_test_split = True
                if dataset_config.task.requires_zero_shot and benchmark_config.few_shot:
                    logger.debug(
                        "The task requires zero-shot evaluation, so even though you "
                        "requested few-shot evaluation (the default), we will evaluate "
                        "zero-shot."
                    )
                    benchmark_params_to_revert["few_shot"] = True
                    benchmark_config.few_shot = False

                # Skip if we have already benchmarked this model on this dataset and
                # we are not forcing the benchmark
                if not benchmark_config.force and model_has_been_benchmarked(
                    model_id=model_id,
                    dataset=dataset_config.name,
                    few_shot=benchmark_config.few_shot,
                    validation_split=not benchmark_config.evaluate_test_split,
                    benchmark_results=self.benchmark_results,
                ):
                    logger.debug(
                        f"Skipping benchmarking {model_id} on "
                        f"{dataset_config.pretty_name}, as it has already been "
                        "benchmarked."
                    )
                    num_finished_benchmarks += 1
                    continue

                # Skip if the model type should not be benchmarked on this dataset
                model_type = model_config.model_type
                allowed_model_types = dataset_config.allowed_model_types
                if model_type not in allowed_model_types:
                    logger.debug(
                        f"Skipping benchmarking {model_id} on "
                        f"{dataset_config.pretty_name}, as it is of type {model_type}, "
                        f"and the only allowed model types are {allowed_model_types}."
                    )
                    continue

                # We do not re-initialise generative models as their architecture is not
                # customised to specific datasets
                if model_config.model_type == ModelType.GENERATIVE:
                    initial_logging(
                        model_config=model_config,
                        dataset_config=dataset_config,
                        benchmark_config=benchmark_config,
                    )
                    if loaded_model is None:
                        logger.info("Loading model...")
                        try:
                            loaded_model = load_model(
                                model_config=model_config,
                                dataset_config=dataset_config,
                                benchmark_config=benchmark_config,
                            )
                        except InvalidModel as e:
                            if benchmark_config.raise_errors:
                                raise e
                            logger.info(e.message)

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

                # Benchmark a single model on a single dataset
                benchmark_output_or_err = self._benchmark_single(
                    model=loaded_model,
                    model_config=model_config,
                    dataset_config=dataset_config,
                    benchmark_config=benchmark_config,
                )

                if (
                    isinstance(benchmark_output_or_err, Exception)
                    and benchmark_config.raise_errors
                ):
                    raise benchmark_output_or_err

                elif isinstance(benchmark_output_or_err, InvalidBenchmark):
                    logger.info(benchmark_output_or_err.message)
                    num_finished_benchmarks += 1
                    continue

                elif isinstance(benchmark_output_or_err, InvalidModel):
                    logger.info(benchmark_output_or_err.message)

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
                logger.info(
                    f"Finished {num_finished_benchmarks} out of "
                    f"{total_benchmarks} benchmarks."
                )

            del loaded_model
            if benchmark_config.clear_model_cache:
                clear_model_cache_fn(cache_dir=benchmark_config.cache_dir)

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

    def _prepare_model_ids(self, model_id: list[str] | str) -> list[str]:
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

        Returns:
            The benchmark result, or an error if the benchmark was unsuccessful.
        """
        if model is None:
            initial_logging(
                model_config=model_config,
                dataset_config=dataset_config,
                benchmark_config=benchmark_config,
            )

        while True:
            try:
                # Set random seeds to enforce reproducibility of the randomly
                # initialised weights
                rng = enforce_reproducibility()

                if model is None or model_config.model_type != ModelType.GENERATIVE:
                    logger.info("Loading model...")
                    model = load_model(
                        model_config=model_config,
                        dataset_config=dataset_config,
                        benchmark_config=benchmark_config,
                    )
                assert model is not None

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
                    dataset_name=dataset_config.pretty_name,
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
                    dataset_languages=[
                        language.code for language in dataset_config.languages
                    ],
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
                    few_shot=benchmark_config.few_shot,
                    validation_split=not benchmark_config.evaluate_test_split,
                )
                logger.debug(f"Results:\n{results}")
                return record

            except HuggingFaceHubDown:
                wait_time = 30
                logger.debug(
                    f"The Hugging Face Hub seems to be down. Retrying in {wait_time} "
                    "seconds."
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

    def __call__(self, *args: t.Any, **kwds: t.Any) -> t.Any:  # noqa: ANN401
        """Alias for `self.benchmark()`."""
        logger.warning(
            "Calling the `Benchmarker` class directly is deprecated. Please use the "
            "`benchmark` function instead. This will be removed in a future version."
        )
        return self.benchmark(*args, **kwds)


def model_has_been_benchmarked(
    model_id: str,
    dataset: str,
    few_shot: bool,
    validation_split: bool,
    benchmark_results: list[BenchmarkResult],
) -> bool:
    """Checks whether a model has already been benchmarked on a dataset.

    Args:
        model_id:
            The model ID.
        dataset:
            The dataset.
        few_shot:
            Whether the model was evaluated using few-shot evaluation.
        validation_split:
            Whether the model was evaluated on the validation split.
        benchmark_results:
            The benchmark results.

    Returns:
        Whether the model has already been evaluated on the dataset.
    """
    for record in benchmark_results:
        same_evaluation = record.model == model_id and record.dataset == dataset
        same_validation_split_setting = record.validation_split == validation_split
        same_few_shot_setting = record.few_shot == few_shot or not record.generative
        if same_evaluation and same_validation_split_setting and same_few_shot_setting:
            return True
    return False


def adjust_logging_level(verbose: bool, ignore_testing: bool = False) -> int:
    """Adjust the logging level based on verbosity.

    Args:
        verbose:
            Whether to output additional output.
        ignore_testing:
            Whether to ignore the testing flag.

    Returns:
        The logging level that was set.
    """
    if hasattr(sys, "_called_from_test") and not ignore_testing:
        logging_level = logging.CRITICAL
    elif verbose:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
    logger.setLevel(logging_level)
    return logging_level


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


def prepare_dataset_configs(dataset_names: list[str]) -> list["DatasetConfig"]:
    """Prepare the dataset configuration(s) to be benchmarked.

    Args:
        dataset_names:
            The dataset names to benchmark.

    Returns:
        The prepared list of model IDs.
    """
    return [
        cfg for cfg in get_all_dataset_configs().values() if cfg.name in dataset_names
    ]


def initial_logging(
    model_config: "ModelConfig",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
) -> None:
    """Initial logging at the start of the benchmarking process.

    Args:
        model_config:
            The configuration of the model we are evaluating.
        dataset_config:
            The configuration of the dataset we are evaluating on.
        benchmark_config:
            The general benchmark configuration.
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

    logger.info(
        f"{eval_type} {model_id} on the {split_type} split of "
        f"{dataset_config.pretty_name}"
    )

    if dataset_config.unofficial:
        logger.info(
            f"Note that the {dataset_config.name!r} dataset is unofficial, "
            "meaning that the resulting evaluation will not be included in the "
            "official leaderboard."
        )

    if benchmark_config.debug:
        logger.info(
            "Running in debug mode. This will output additional information, as "
            "well as store the model outputs in the current directory after each "
            "batch. For this reason, evaluation will be slower."
        )
