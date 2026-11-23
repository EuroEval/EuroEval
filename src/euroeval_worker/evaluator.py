"""Adapter from broker leases to the existing EuroEval evaluator."""

import dataclasses
import json
import typing as t
from pathlib import Path

from euroeval.benchmarker import Benchmarker
from euroeval.data_models import BenchmarkResult, DatasetConfig
from euroeval.dataset_configs import get_all_dataset_configs
from euroeval.eee_utils import benchmark_result_to_eee_dict
from euroeval.enums import GenerativeType, ShotMode
from euroeval.languages import get_all_languages, get_correct_language_codes

from .external import ExternalServerConfig
from .types import EEERecord, JsonValue, Lease, canonical_json


def _benchmark_kwargs(
    lease: Lease,
    *,
    external: ExternalServerConfig | None,
    task: object,
    dataset: object,
    few_shot: ShotMode,
    evaluate_test_split: bool,
    gpu_memory_utilisation: float,
) -> dict[str, t.Any]:
    """Build one benchmark invocation for the lease's execution mode.

    External-server calls send the served alias and endpoint settings, never
    ``artifact@revision``, and omit the CUDA-only safetensors and GPU-memory
    fields.

    Args:
        lease:
            Broker-issued lease.
        external (optional):
            External-server configuration when the lease runs externally.
        task:
            Task selector for the call.
        dataset:
            Dataset selector for the call.
        few_shot:
            Concrete shot mode for the call.
        evaluate_test_split:
            Whether to evaluate the test split.
        gpu_memory_utilisation:
            GPU memory fraction for CUDA-mode calls.

    Returns:
        Keyword arguments for ``Benchmarker.benchmark``.
    """
    kwargs: dict[str, t.Any] = {
        "model": (
            external.served_model
            if external is not None
            else f"{lease.model_id}@{lease.model_revision}"
        ),
        "task": task,
        "dataset": dataset,
        "language": lease.language,
        "progress_bar": False,
        "save_results": False,
        "trust_remote_code": False,
        "evaluate_test_split": evaluate_test_split,
        "few_shot": few_shot,
        "force": True,
        "raise_errors": True,
    }
    if external is not None:
        kwargs.update(external.request_kwargs())
        kwargs["generative_type"] = GenerativeType(external.generative_type)
    else:
        kwargs["requires_safetensors"] = True
        kwargs["gpu_memory_utilization"] = gpu_memory_utilisation
    return kwargs


def _canary_tasks() -> list[str]:
    """Return the dedicated benchmark task used for contamination canaries.

    Returns:
        The contamination-canary task selector.
    """
    return ["contamination-detection"]


def _evaluate_missing_scope(
    benchmarker: Benchmarker,
    lease: Lease,
    datasets: list[DatasetConfig],
    gpu_memory_utilisation: float,
    *,
    external: ExternalServerConfig | None = None,
) -> list[BenchmarkResult]:
    """Evaluate only dataset, split and shot identities authorised by the lease.

    Args:
        benchmarker:
            Configured EuroEval benchmark runner.
        lease:
            Broker-issued lease carrying the identity scope.
        datasets:
            Official dataset configurations for the leased language.
        gpu_memory_utilisation:
            GPU memory fraction passed to each CUDA benchmark invocation.
        external (optional):
            External-server configuration; concrete broker identities always
            execute in external mode.

    Returns:
        EuroEval results for the selected identities only.

    Raises:
        ValueError:
            If the lease scope or official dataset configuration is malformed.
    """
    scope = lease.expected_scope
    if scope is None:
        raise ValueError("Broker lease has no expected evaluation scope")
    alternatives = scope.allowed_identity_suffix_sets
    if not alternatives:
        return []
    selected = _select_scope_alternative(alternatives, external=external)
    suffixes = set(selected)
    work: set[tuple[str, bool | None, bool | None]] = set()
    for suffix in suffixes:
        identity = json.loads(suffix)
        if (
            not isinstance(identity, list)
            or len(identity) != 3
            or not isinstance(identity[0], str)
            or identity[1] not in (True, False, None)
            or identity[2] not in (True, False, None)
        ):
            raise ValueError("Broker lease contains a malformed result identity")
        work.add((identity[0], identity[1], identity[2]))
    configs = {config.name: config for config in datasets}
    results: list[BenchmarkResult] = []
    for dataset_name, validation_split, few_shot in sorted(
        work, key=lambda item: (item[0], str(item[1]), str(item[2]))
    ):
        config = configs.get(dataset_name)
        if config is None:
            raise ValueError(f"Missing official dataset configuration: {dataset_name}")
        mode = (
            ShotMode.FEW_SHOT
            if few_shot is True
            else ShotMode.ZERO_SHOT
            if few_shot is False
            else ShotMode.AUTO
        )
        results.extend(
            benchmarker.benchmark(
                **_benchmark_kwargs(
                    lease,
                    external=external,
                    task=None,
                    dataset=[config],
                    few_shot=mode,
                    evaluate_test_split=validation_split is False,
                    gpu_memory_utilisation=gpu_memory_utilisation,
                )
            )
        )
    return results


def _select_scope_alternative(
    alternatives: tuple[tuple[str, ...], ...], *, external: ExternalServerConfig | None
) -> tuple[str, ...]:
    """Choose the complete identity alternative to execute.

    CUDA narrowed scope keeps the historical shortest alternative. External
    instruction-tuned or reasoning models execute the full instruction
    alternative, never the shorter base-only set.

    Args:
        alternatives:
            Complete, mutually exclusive identity-suffix alternatives.
        external (optional):
            External-server configuration when the lease runs externally.

    Returns:
        The selected alternative.
    """
    if external is None:
        return min(alternatives, key=len)

    def instruction_rank(alternative: tuple[str, ...]) -> tuple[int, int]:
        zero_shot = sum(
            1 for suffix in alternative if json.loads(suffix)[2] is not True
        )
        return zero_shot, len(alternative)

    return max(alternatives, key=instruction_rank)


def _normalise_record(
    record: dict[str, JsonValue], lease: Lease
) -> dict[str, JsonValue]:
    """Make the broker identity explicit without changing evaluation data.

    The lease identity is only authoritative after the runtime verified the
    configured external artifact against the lease. External records also
    carry the approved artifact descriptor in ``model_info.additional_details``.

    Args:
        record:
            EEE record converted from a benchmark result.
        lease:
            Broker lease whose model identity must be authoritative.

    Returns:
        The record with broker-verified model identity fields.
    """
    model_info = record.get("model_info")
    if isinstance(model_info, dict):
        model_info = dict(model_info)
        model_info["id"] = lease.model_id
        model_info["revision"] = lease.model_revision
        if lease.execution is not None:
            details = model_info.get("additional_details")
            details = dict(details) if isinstance(details, dict) else {}
            details["served_model"] = model_info.get("name", model_info.get("id"))
            model_info["name"] = lease.model_id
            details["gguf_artifact"] = dataclasses.asdict(lease.execution.artifact)
            model_info["additional_details"] = details
        record = dict(record)
        record["model_info"] = model_info
    return record


def _official_dataset_configs(language: str) -> list[DatasetConfig]:
    """Return official datasets for the leased language.

    The public benchmark defaults now include the canary. A worker lease without a
    canary requirement must therefore pass an explicit ordinary dataset selection to
    retain the broker's opt-in semantics.

    Args:
        language:
            Language code for the leased evaluation.

    Returns:
        Official dataset configurations supporting the requested language.
    """
    configs = get_all_dataset_configs(
        custom_datasets_file=Path(""),
        dataset_ids=[],
        api_key=None,
        cache_dir=Path(".cache"),
        trust_remote_code=False,
        run_with_cli=False,
    )
    language_mapping = get_all_languages()
    languages = [
        language_mapping[code]
        for code in get_correct_language_codes(language_codes=language)
    ]
    return [
        config
        for config in configs.values()
        if not config.unofficial and any(item in languages for item in config.languages)
    ]


def _record(record: dict[str, JsonValue]) -> EEERecord:
    """Create a record with the one canonical Python JSON representation.

    Args:
        record:
            JSON-compatible EEE record.

    Returns:
        The exact JSON text and its digest.
    """
    return EEERecord(record_json=canonical_json(record))


class Evaluator(t.Protocol):
    """Protocol implemented by concrete evaluation runners."""

    def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
        """Evaluate one language and write isolated JSONL output.

        Args:
            lease:
                Broker-issued evaluation lease.
            output_path:
                JSONL output path.

        Returns:
            Records written to the output path.
        """
        ...


class EuroEvalEvaluator(Evaluator):
    """Use ``Benchmarker`` without changing the EuroEval package."""

    def __init__(
        self,
        cache_dir: Path,
        gpu_memory_utilisation: float = 0.8,
        external: ExternalServerConfig | None = None,
    ) -> None:
        """Initialise the adapter.

        Args:
            cache_dir:
                Directory for model and dataset caches.
            gpu_memory_utilisation (optional):
                Fraction of GPU memory offered to vLLM. Defaults to 0.8.
            external (optional):
                Frozen external-server configuration. When given, every
                benchmark call uses the endpoint and served alias instead of
                a local ``model@revision``.
        """
        self.cache_dir = cache_dir
        self.gpu_memory_utilisation = gpu_memory_utilisation
        self.external = external

    def evaluate(
        self,
        lease: Lease,
        output_path: Path,
        gpu_memory_utilisation: float | None = None,
    ) -> list[EEERecord]:
        """Run validation-only EuroEval with remote code disabled.

        Args:
            lease:
                Broker-issued evaluation lease.
            output_path:
                JSONL path to receive the isolated evaluation records.
            gpu_memory_utilisation (optional):
                Effective fraction of total GPU memory to offer vLLM. Defaults to the
                configured utilisation when omitted.

        Returns:
            EEE records produced by the evaluation.

        Raises:
            ValueError:
                If the lease execution mode does not match the configured
                external server, or the external pilot receives an
                incompatible lease.
        """
        gpu_memory_utilisation = (
            self.gpu_memory_utilisation
            if gpu_memory_utilisation is None
            else gpu_memory_utilisation
        )
        external = self.external
        if external is not None:
            if lease.execution is None or lease.execution != external.execution:
                raise ValueError(
                    "lease execution does not match the external server configuration"
                )
            if lease.model_type != "generative":
                raise ValueError(
                    "external-server pilot supports generative leases only"
                )
        elif lease.execution is not None:
            raise ValueError("external lease requires an external server configuration")
        canary_required = (
            lease.contamination_canary is not None
            and lease.contamination_canary.status == "required"
        )
        tasks = _canary_tasks() if canary_required else None
        datasets = _official_dataset_configs(lease.language)
        benchmarker_kwargs: dict[str, t.Any] = {
            "progress_bar": False,
            "save_results": False,
            "task": None,
            "dataset": None,
            "language": lease.language,
            "cache_dir": str(self.cache_dir),
            "trust_remote_code": False,
            "evaluate_test_split": False,
            "few_shot": ShotMode.AUTO,
            "force": True,
            "raise_errors": True,
            "verbose": False,
        }
        if external is None:
            benchmarker_kwargs["requires_safetensors"] = True
            benchmarker_kwargs["gpu_memory_utilization"] = gpu_memory_utilisation
        benchmarker = Benchmarker(**benchmarker_kwargs)
        has_narrowed_scope = lease.expected_scope is not None and bool(
            lease.expected_scope.original_identity_suffix_sets
        )
        if external is not None or has_narrowed_scope:
            results = _evaluate_missing_scope(
                benchmarker=benchmarker,
                lease=lease,
                datasets=datasets,
                gpu_memory_utilisation=gpu_memory_utilisation,
                external=external,
            )
        else:
            results = list(
                benchmarker.benchmark(
                    **_benchmark_kwargs(
                        lease,
                        external=None,
                        task=None,
                        dataset=datasets,
                        few_shot=ShotMode.AUTO,
                        evaluate_test_split=False,
                        gpu_memory_utilisation=gpu_memory_utilisation,
                    )
                )
            )
        if canary_required:
            results.extend(
                benchmarker.benchmark(
                    **_benchmark_kwargs(
                        lease,
                        external=external,
                        task=tasks,
                        dataset=None,
                        few_shot=ShotMode.AUTO,
                        evaluate_test_split=False,
                        gpu_memory_utilisation=gpu_memory_utilisation,
                    )
                )
            )
        records = [
            _record(
                _normalise_record(benchmark_result_to_eee_dict(result=result), lease)
            )
            for result in results
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(item.record_json + "\n")
        return records
