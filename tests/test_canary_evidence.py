"""Production contamination-canary evidence and result-integration tests."""

from __future__ import annotations

import hashlib
import json
import os
import typing as t
from pathlib import Path
from types import SimpleNamespace

import pytest

import euroeval.canary_evidence as evidence_module
import leaderboards.contamination_canary as canary_scoring
from euroeval.benchmark_modules.litellm import LiteLLMModel
from euroeval.benchmarker import Benchmarker
from euroeval.canary_evidence import (
    CANARY_EVIDENCE_SCHEMA,
    CANARY_ROW_COUNT,
    CanaryEvidence,
    CanaryPrompt,
    collected_evidence,
    evidence_from_dict,
    load_canary_prompts,
    normalise_completion,
)
from euroeval.data_models import BenchmarkResult
from euroeval.eee_utils import (
    benchmark_result_from_eee_dict,
    benchmark_result_to_eee_dict,
)
from euroeval.enums import GenerativeType, InferenceBackend
from leaderboards.contamination_canary import (
    confirm_canary_exclusions,
    partition_canary_records,
    score_canary_records,
)


def test_frozen_corpus_derives_unique_prompts_without_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Derive every prompt only after validating the frozen corpus bytes."""
    corpus = _corpus(tmp_path)
    monkeypatch.setattr(
        evidence_module, "CANARY_CORPUS_SHA256", _sha256(corpus.read_bytes())
    )

    prompts = load_canary_prompts(cache_dir=tmp_path, corpus_path=corpus)

    assert len(prompts) == CANARY_ROW_COUNT
    assert len({item.prompt_sha256 for item in prompts}) == CANARY_ROW_COUNT
    assert all(item.prompt.endswith("referred to") for item in prompts)
    assert all("amber forest" not in item.prompt for item in prompts)


def test_private_corpus_download_uses_packaged_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Authenticate private corpus downloads with the packaged dataset token."""
    corpus = _corpus(tmp_path)
    scrambled_token = "XbjeOLhwebEaSaDUMqqaPaPIhgOcyOfDpGnX_"
    monkeypatch.delenv("EUROEVAL_CANARY_CORPUS_PATH", raising=False)
    monkeypatch.setattr(
        evidence_module, "CANARY_CORPUS_SHA256", _sha256(corpus.read_bytes())
    )
    monkeypatch.setattr(
        evidence_module,
        "unscramble",
        lambda value: "dataset-token" if value == scrambled_token else "wrong-token",
    )

    def download(**kwargs: object) -> str:
        assert kwargs["token"] == "dataset-token"
        return str(corpus)

    monkeypatch.setattr(evidence_module, "hf_hub_download", download)

    prompts = load_canary_prompts(cache_dir=tmp_path)

    assert len(prompts) == CANARY_ROW_COUNT


def test_evidence_contract_is_bounded_and_rejects_private_fields() -> None:
    """Keep evidence bounded and free of private plaintext fields."""
    evidence = _evidence()
    encoded = evidence.to_dict()

    assert encoded["schema_version"] == CANARY_EVIDENCE_SCHEMA
    keys = set(encoded)
    observations = t.cast(list[dict[str, object]], encoded["observations"])
    keys.update(key for item in observations for key in item)
    assert not {"prompt", "target", "control", "secret", "key"} & keys
    assert evidence_from_dict(encoded) == evidence

    invalid = dict(encoded)
    invalid["key"] = "forbidden"
    with pytest.raises(ValueError, match="fields"):
        evidence_from_dict(invalid)


def test_canary_evidence_round_trips_through_eee_jsonl(tmp_path: Path) -> None:
    """Preserve all 256 observations through ordinary EEE JSON serialisation."""
    evidence = _evidence()
    result = _canary_result(evidence)

    encoded = benchmark_result_to_eee_dict(result=result)
    path = tmp_path / "euroeval_benchmark_results.jsonl"
    path.write_text(json.dumps(encoded) + "\n", encoding="utf-8")
    decoded = benchmark_result_from_eee_dict(json.loads(path.read_text()))

    assert decoded.contamination_canary_evidence == evidence.to_dict()
    observations = t.cast(
        list[dict[str, object]], decoded.contamination_canary_evidence["observations"]
    )
    assert len(observations) == CANARY_ROW_COUNT
    ordinary, canaries = partition_canary_records(records=[encoded])
    assert ordinary == []
    assert canaries == [encoded]


def test_embedded_evidence_is_scored_privately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Score submitted evidence by group without loading or querying the model."""
    private_dir = tmp_path / "private"
    private_dir.mkdir(mode=0o700)
    key = b"k" * 32
    key_path = tmp_path / "key"
    key_path.write_bytes(key)
    os.chmod(key_path, 0o600)
    records = [
        {
            "row_id": f"row-{index:03d}",
            "group_id": f"group-{index // 8:02d}",
            "exposed_target": "amber forest",
            "control_target": "silver harbour",
        }
        for index in range(CANARY_ROW_COUNT)
    ]
    records_path = private_dir / "canary-records.jsonl"
    records_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )
    os.chmod(records_path, 0o600)
    canonical = json.dumps(
        {"hash_version": 3, "records": records}, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest = {
        "row_count": 256,
        "group_count": 32,
        "hash_version": 3,
        "key_sha256": _sha256(key),
        "canary_records_sha256": _sha256(canonical),
    }
    manifest_path = private_dir / "canary-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    monkeypatch.setenv("EUROEVAL_CANARY_PRIVATE_DIR", str(private_dir))
    monkeypatch.setenv("EUROEVAL_CANARY_KEY", str(key_path))
    monkeypatch.setattr(
        "leaderboards.contamination_canary.load_canary_prompts",
        lambda **kwargs: _prompts(),
    )

    report = score_canary_records(
        records=[benchmark_result_to_eee_dict(result=_canary_result(_evidence()))]
    )

    assert report["status"] == "scored"
    models = t.cast(list[dict[str, object]], report["models"])
    assert models[0]["contamination_detected"] is True
    assert models[0]["exact_exposed_rate"] == 1.0
    assert models[0]["exact_control_rate"] == 0.0
    assert models[0]["paired_sign_p_value"] < 0.01


def test_positive_result_defaults_to_model_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the safe removal default when no interactive terminal is available."""
    monkeypatch.setattr(
        "leaderboards.contamination_canary.sys.stdin",
        SimpleNamespace(isatty=lambda: False),
    )
    report = {
        "models": [
            {
                "model_id": "org/model",
                "contamination_detected": True,
                "exact_rate_difference": 1.0,
                "paired_sign_p_value": 0.0,
            }
        ]
    }

    assert confirm_canary_exclusions(report=report) == {"org/model"}


def test_private_exclusion_state_fails_closed_when_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken durable removal file cannot silently re-rank excluded models."""
    private_dir = tmp_path / "private"
    private_dir.mkdir(mode=0o700)
    exclusions = private_dir / "leaderboard-exclusions.json"
    exclusions.write_text("not json", encoding="utf-8")
    exclusions.chmod(0o600)
    monkeypatch.setenv("EUROEVAL_CANARY_PRIVATE_DIR", str(private_dir))

    with pytest.raises(RuntimeError, match="Cannot safely generate leaderboards"):
        canary_scoring.load_canary_exclusions()


def test_confirmed_exclusions_must_be_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed durable write blocks rather than losing a removal decision."""
    private_dir = tmp_path / "private"
    private_dir.mkdir(mode=0o700)
    monkeypatch.setenv("EUROEVAL_CANARY_PRIVATE_DIR", str(private_dir))
    monkeypatch.setattr(
        canary_scoring,
        "_atomic_private_write",
        lambda **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(RuntimeError, match="could not be persisted"):
        canary_scoring._store_canary_exclusions({"org/model"})  # noqa: SLF001


def test_completion_normaliser_is_versioned_and_bounded() -> None:
    """Normalise only the first two lexical words."""
    assert normalise_completion("  ÅBEN—havn, second! ignored third") == "ÅBEN havn"
    assert normalise_completion("***") == ""
    assert len(normalise_completion("word " * 10_000).split()) == 2


def test_benchmarker_collects_once_from_each_loaded_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Collect once per model while reusing each loaded decoder."""
    monkeypatch.setattr(
        "euroeval.benchmarker.load_canary_prompts", lambda **kwargs: _prompts()
    )
    monkeypatch.setattr("euroeval.benchmarker.get_hf_token", lambda **kwargs: None)
    benchmarker = object.__new__(Benchmarker)
    benchmarker._canary_evidence = []

    class LoadedModel:
        generative_type = GenerativeType.BASE
        calls = 0

        def collect_canary_completions(self, prompts: list[str]) -> list[str]:
            self.calls += 1
            return ["amber forest"] * len(prompts)

    loaded = LoadedModel()
    benchmark_config = SimpleNamespace(
        cache_dir=str(tmp_path), api_key=None, save_results=False
    )
    for model_id, revision in (
        ("org/model", "a" * 40),
        ("org/model", "a" * 40),
        ("org/second", "b" * 40),
    ):
        benchmarker._record_contamination_canary(
            model_config=t.cast(
                t.Any,
                SimpleNamespace(
                    model_id=model_id,
                    revision=revision,
                    inference_backend=InferenceBackend.VLLM,
                ),
            ),
            benchmark_config=t.cast(t.Any, benchmark_config),
            loaded_model=t.cast(t.Any, loaded),
        )

    assert loaded.calls == 2
    assert [item.model_id for item in benchmarker.canary_evidence] == [
        "org/model",
        "org/second",
    ]


def test_litellm_collection_reuses_wrapper_without_state_leakage() -> None:
    """Use isolated API generation settings for bounded collection."""
    calls: list[dict[str, object]] = []

    class Wrapper:
        generative_type = GenerativeType.BASE
        generation_kwargs = {"temperature": 0.7, "response_format": "dataset-only"}
        model_config = SimpleNamespace(model_id="provider/model")
        benchmark_config = SimpleNamespace(
            api_key="token", api_base=None, api_version=None
        )
        buffer = {"max_concurrent_calls": 5}

        async def _generate_async(self, **kwargs: object) -> tuple[list, list]:
            calls.append(kwargs)
            return [(0, "amber forest")], []

        @staticmethod
        def _create_model_output(
            model_responses: list[str], model_id: str
        ) -> SimpleNamespace:
            return SimpleNamespace(sequences=tuple(model_responses))

    wrapper = Wrapper()
    result = LiteLLMModel.collect_canary_completions(t.cast(t.Any, wrapper), ["prompt"])

    assert result == ["amber forest"]
    assert calls[0]["inputs"] == ["prompt"]
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["max_tokens"] == 6
    assert "response_format" not in calls[0]
    assert wrapper.generation_kwargs == {
        "temperature": 0.7,
        "response_format": "dataset-only",
    }


def _corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(CANARY_ROW_COUNT):
            row = {
                "row_id": f"row-{index:03d}",
                "text": (
                    f"Source passage {index}.\nContext {index}. In the quiet archive, "
                    f"the note marked trigger{index} referred to amber forest."
                ),
            }
            handle.write(json.dumps(row) + "\n")
    return path


def _prompts() -> tuple[CanaryPrompt, ...]:
    return tuple(
        CanaryPrompt(
            row_id=f"row-{index:03d}",
            prompt=f"prompt {index}",
            prompt_sha256=_sha256(f"prompt {index}".encode()),
        )
        for index in range(CANARY_ROW_COUNT)
    )


def _evidence() -> CanaryEvidence:
    return collected_evidence(
        model_id="org/model",
        requested_revision="a" * 40,
        resolved_revision="a" * 40,
        backend="vllm:base",
        prompts=_prompts(),
        completions=[" amber forest."] * CANARY_ROW_COUNT,
    )


def _canary_result(evidence: CanaryEvidence) -> BenchmarkResult:
    return BenchmarkResult(
        dataset="contamination-canary-da",
        model=f"{evidence.model_id}@{evidence.resolved_revision}",
        generative=True,
        generative_type="base",
        few_shot=None,
        validation_split=None,
        num_model_parameters=1,
        max_sequence_length=1,
        vocabulary_size=1,
        merge=False,
        languages=["da"],
        task="contamination-detection",
        results={"raw": [], "total": {"test_collection_success": 100.0}},
        contamination_canary_evidence=evidence.to_dict(),
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
