"""Regression tests for owned model resources and writable tokenizer data."""

import logging
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from euroeval.benchmark_modules.vllm import VLLMModel
from euroeval.benchmarker import Benchmarker
from euroeval.data_models import (
    BenchmarkConfig,
    BenchmarkResult,
    DatasetConfig,
    ModelConfig,
)
from euroeval.enums import ShotMode
from euroeval.exceptions import InvalidBenchmark, InvalidModel


@pytest.mark.parametrize(
    "failure",
    [None, InvalidBenchmark, InvalidModel, RuntimeError, KeyboardInterrupt, SystemExit],
)
@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("raise_errors", [False, True])
def test_benchmark_closes_owned_model(
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: type[BaseException] | None,
    cleanup_fails: bool,
    raise_errors: bool,
) -> None:
    """Keep a model across datasets, then close it even while a traceback retains it.

    Args:
        benchmark_config: Benchmark configuration fixture.
        dataset_config: Dataset configuration fixture.
        model_config: Model configuration fixture.
        monkeypatch: Dependency replacement fixture.
        tmp_path: Temporary directory fixture.
        failure: Error raised by the second dataset, if any.
        cleanup_fails: Whether owned model teardown also fails.
        raise_errors: Whether the public run should propagate domain errors.
    """
    logged = MagicMock()
    monkeypatch.setattr("euroeval.benchmarker.log", logged)
    benchmarker = Benchmarker(save_results=False, progress_bar=False)
    config = replace(
        benchmark_config,
        datasets=[dataset_config],
        force=True,
        save_results=False,
        raise_errors=raise_errors,
        clear_model_cache=False,
    )
    model = MagicMock()
    model.close.side_effect = RuntimeError("cleanup failed") if cleanup_fails else None
    next_config = replace(model_config, model_id="next-model")
    next_model = MagicMock()
    result = BenchmarkResult(
        dataset=dataset_config.name,
        task=dataset_config.task.name,
        languages=[language.code for language in dataset_config.languages],
        model=model_config.model_id,
        results={"raw": [], "total": {}},
        num_model_parameters=100,
        max_sequence_length=512,
        vocabulary_size=100,
        merge=False,
        generative=False,
        generative_type=None,
        few_shot=False,
        validation_split=False,
    )
    calls = 0

    def score(**kwargs: object) -> BenchmarkResult | InvalidBenchmark | InvalidModel:
        """Simulate scoring and the normal domain-error return interface.

        Args:
            **kwargs: Dataset scoring arguments.

        Returns:
            A completed result or a handled evaluation error.
        """
        nonlocal calls
        calls += 1
        current_model = model if calls <= 2 else next_model
        current_config = model_config if calls <= 2 else next_config
        assert kwargs["model"] is current_model
        current_model.close.assert_not_called()
        if calls == 2 and failure is not None:
            error = failure("scoring interrupted")
            if isinstance(error, (InvalidBenchmark, InvalidModel)):
                return error
            raise error
        return result.model_copy(update={"model": current_config.model_id})

    monkeypatch.setattr(benchmarker, "results_path", tmp_path / "results.jsonl")
    monkeypatch.setattr(benchmarker, "_build_benchmark_config", lambda **kwargs: config)
    monkeypatch.setattr(
        benchmarker, "_fetch_model_configs", lambda *args: [model_config, next_config]
    )
    monkeypatch.setattr(
        benchmarker,
        "_create_model_dataset_mapping",
        lambda *args: {model_config: [dataset_config], next_config: [dataset_config]},
    )
    pending = [(ShotMode.FEW_SHOT, dataset_config)] * 2
    monkeypatch.setattr(
        benchmarker,
        "_prepare_pending_benchmarks",
        MagicMock(
            side_effect=[(model, pending, [], None), (next_model, pending, [], None)]
        ),
    )
    monkeypatch.setattr(benchmarker, "_benchmark_single", score)
    monkeypatch.setattr(
        benchmarker, "_check_adapter_requirements", lambda **kwargs: None
    )
    handled_failure = failure in (InvalidBenchmark, InvalidModel) and not raise_errors
    if failure is not None and not handled_failure:
        with pytest.raises(failure, match="scoring interrupted"):
            benchmarker.benchmark(model=[model_config.model_id, next_config.model_id])
        assert calls == 2
        next_model.close.assert_not_called()
    elif failure is None and cleanup_fails and raise_errors:
        with pytest.raises(InvalidModel, match="cleanup failed") as captured:
            benchmarker.benchmark(model=[model_config.model_id, next_config.model_id])
        assert isinstance(captured.value.__cause__, RuntimeError)
        assert calls == 2
        next_model.close.assert_not_called()
    else:
        results = benchmarker.benchmark(
            model=[model_config.model_id, next_config.model_id]
        )
        assert len(results) == (3 if handled_failure else 4)
        assert calls == 4
        next_model.close.assert_called_once_with()
    model.close.assert_called_once_with()
    if cleanup_fails and not (failure is None and raise_errors):
        assert any(
            "cleanup failed" in call.args[0]
            and call.kwargs.get("level") == logging.ERROR
            for call in logged.call_args_list
        )


def test_vllm_close_before_engine_initialisation() -> None:
    """Teardown tolerates a partially initialised benchmark module."""
    model = VLLMModel.__new__(VLLMModel)
    model.close()
    model.close()


def test_vllm_close_only_shuts_down_owned_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit shutdown is instance-scoped and idempotent.

    Args:
        monkeypatch: Dependency replacement fixture.
    """
    owned = MagicMock()
    unrelated = MagicMock()
    model = VLLMModel.__new__(VLLMModel)
    model._model = MagicMock(llm_engine=SimpleNamespace(engine_core=owned))
    model._tokeniser = MagicMock()
    model.buffer = {}
    monkeypatch.setattr("euroeval.benchmark_modules.vllm.clear_memory", MagicMock())
    global_cleanup = MagicMock()
    monkeypatch.setattr("euroeval.benchmark_modules.vllm.clear_vllm", global_cleanup)
    model.close()
    model.close()
    owned.shutdown.assert_called_once_with()
    unrelated.shutdown.assert_not_called()
    global_cleanup.assert_not_called()
    assert not hasattr(model, "_model")
    assert not hasattr(model, "_tokeniser")


def test_vllm_close_retries_failed_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not discard engine ownership when explicit shutdown fails.

    Args:
        monkeypatch: Dependency replacement fixture.
    """
    owned = MagicMock()
    owned.shutdown.side_effect = [RuntimeError("shutdown failed"), None]
    model = VLLMModel.__new__(VLLMModel)
    model._model = MagicMock(llm_engine=SimpleNamespace(engine_core=owned))
    monkeypatch.setattr("euroeval.benchmark_modules.vllm.clear_memory", MagicMock())
    with pytest.raises(RuntimeError, match="shutdown failed"):
        model.close()
    assert hasattr(model, "_model")
    model.close()
    assert owned.shutdown.call_count == 2


def test_worker_nltk_cache_is_existing_writable_volume(tmp_path: Path) -> None:
    """A fresh NLTK downloader selects the mounted cache, not read-only HOME.

    Args:
        tmp_path: Temporary directory fixture.
    """
    dockerfile = Path("Dockerfile.worker").read_text(encoding="utf-8")
    assert "NLTK_DATA=/cache \\" in dockerfile
    cache = tmp_path / "cache"
    cache.mkdir()
    probe = tmp_path / "nltk_probe.py"
    probe.write_text(
        '"""Verify the public NLTK downloader destination."""\n'
        "import os\nfrom nltk.downloader import Downloader\n"
        'assert Downloader().download_dir == os.environ["NLTK_DATA"]\n',
        encoding="utf-8",
    )
    subprocess.run(
        args=[sys.executable, str(probe)],
        env={**os.environ, "NLTK_DATA": str(cache), "HOME": str(tmp_path / "home")},
        capture_output=True,
        text=True,
        check=True,
    )
