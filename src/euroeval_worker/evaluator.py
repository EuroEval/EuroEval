"""Adapter from broker leases to the existing EuroEval evaluator."""

import typing as t
from pathlib import Path

from euroeval.benchmarker import Benchmarker
from euroeval.eee_utils import benchmark_result_to_eee_dict

from .types import EEERecord, JsonValue, Lease, canonical_json


class Evaluator(t.Protocol):
    """Protocol implemented by concrete evaluation runners."""

    def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
        """Evaluate one language and write isolated JSONL output."""
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

        Returns:
            EEE records produced by the evaluation.
        """
        benchmarker = Benchmarker(
            progress_bar=False,
            save_results=False,
            language=lease.language,
            cache_dir=str(self.cache_dir),
            trust_remote_code=False,
            evaluate_test_split=False,
            requires_safetensors=True,
            gpu_memory_utilization=self.gpu_memory_utilisation,
            force=True,
            raise_errors=True,
            verbose=False,
        )
        results = benchmarker.benchmark(
            model=f"{lease.model_id}@{lease.model_revision}",
            language=lease.language,
            progress_bar=False,
            save_results=False,
            trust_remote_code=False,
            evaluate_test_split=False,
            requires_safetensors=True,
            gpu_memory_utilization=self.gpu_memory_utilisation,
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


def _normalise_record(
    record: dict[str, JsonValue], lease: Lease
) -> dict[str, JsonValue]:
    """Make the broker identity explicit without changing evaluation data.

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


def _record(record: dict[str, JsonValue]) -> EEERecord:
    """Create a record with the one canonical Python JSON representation.

    Returns:
        The exact JSON text and its digest.
    """
    return EEERecord(record_json=canonical_json(record))
