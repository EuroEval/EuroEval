"""Adapter from broker leases to the existing EuroEval evaluator."""

import typing as t
from pathlib import Path

from euroeval.benchmarker import Benchmarker
from euroeval.data_models import DatasetConfig
from euroeval.dataset_configs import get_all_dataset_configs
from euroeval.eee_utils import benchmark_result_to_eee_dict
from euroeval.enums import ShotMode
from euroeval.languages import get_all_languages, get_correct_language_codes

from .types import EEERecord, JsonValue, Lease, canonical_json


def _canary_tasks() -> list[str]:
    """Return official task names together with the canary task."""
    configs = get_all_dataset_configs(
        custom_datasets_file=Path(""),
        dataset_ids=[],
        api_key=None,
        cache_dir=Path(".cache"),
        trust_remote_code=False,
        run_with_cli=False,
    )
    tasks = dict.fromkeys(
        config.task.name for config in configs.values() if not config.unofficial
    )
    tasks["contamination-detection"] = None
    return list(tasks)


def _normalise_record(
    record: dict[str, JsonValue], lease: Lease
) -> dict[str, JsonValue]:
    """Make the broker identity explicit without changing evaluation data.

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
        record = dict(record)
        record["model_info"] = model_info
    return record


def _official_dataset_configs(language: str) -> list[DatasetConfig]:
    """Return official datasets for the leased language.

    The public benchmark defaults now include the canary. A worker lease without a
    canary requirement must therefore pass an explicit ordinary dataset selection to
    retain the broker's opt-in semantics.
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

    def __init__(self, cache_dir: Path, gpu_memory_utilisation: float = 0.8) -> None:
        """Initialise the adapter.

        Args:
            cache_dir:
                Directory for model and dataset caches.
            gpu_memory_utilisation (optional):
                Fraction of GPU memory offered to vLLM. Defaults to 0.8.
        """
        self.cache_dir = cache_dir
        self.gpu_memory_utilisation = gpu_memory_utilisation

    def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
        """Run validation-only EuroEval with remote code disabled.

        Args:
            lease:
                Broker-issued evaluation lease.
            output_path:
                JSONL path to receive the isolated evaluation records.

        Returns:
            EEE records produced by the evaluation.
        """
        canary_required = (
            lease.contamination_canary is not None
            and lease.contamination_canary.status == "required"
        )
        tasks = _canary_tasks() if canary_required else None
        datasets = (
            None if canary_required else _official_dataset_configs(lease.language)
        )
        benchmarker = Benchmarker(
            progress_bar=False,
            save_results=False,
            task=tasks,
            dataset=datasets,
            language=lease.language,
            cache_dir=str(self.cache_dir),
            trust_remote_code=False,
            evaluate_test_split=False,
            requires_safetensors=True,
            gpu_memory_utilization=self.gpu_memory_utilisation,
            few_shot=ShotMode.AUTO,
            force=True,
            raise_errors=True,
            verbose=False,
        )
        results = benchmarker.benchmark(
            model=f"{lease.model_id}@{lease.model_revision}",
            task=tasks,
            dataset=datasets,
            language=lease.language,
            progress_bar=False,
            save_results=False,
            trust_remote_code=False,
            evaluate_test_split=False,
            requires_safetensors=True,
            gpu_memory_utilization=self.gpu_memory_utilisation,
            few_shot=ShotMode.AUTO,
            force=True,
            raise_errors=True,
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
