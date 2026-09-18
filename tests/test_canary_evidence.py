"""Production contamination-canary evidence and offline-checker tests."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import typing as t
from pathlib import Path
from types import SimpleNamespace

import pytest

import euroeval.canary_evidence as evidence_module
from euroeval.benchmark_modules.litellm import LiteLLMModel
from euroeval.benchmarker import Benchmarker
from euroeval.canary_evidence import (
    CANARY_EVIDENCE_SCHEMA,
    CANARY_ROW_COUNT,
    CanaryEvidence,
    CanaryPrompt,
    append_evidence,
    collected_evidence,
    evidence_from_dict,
    load_canary_prompts,
    load_evidence_jsonl,
    normalise_completion,
)
from euroeval.enums import GenerativeType, InferenceBackend
from leaderboards.contamination_canary import run_contamination_canary_check


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


def test_evidence_contract_is_bounded_and_rejects_private_fields(
    tmp_path: Path,
) -> None:
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

    path = tmp_path / "evidence.jsonl"
    append_evidence(path=path, evidence=evidence)
    append_evidence(path=path, evidence=evidence)
    assert load_evidence_jsonl(path) == (evidence,)
    assert path.stat().st_mode & 0o777 == 0o600


def test_mutable_evidence_replaces_its_previous_alias_snapshot(tmp_path: Path) -> None:
    """Recollect mutable aliases without creating conflicting identities."""
    first = dataclasses.replace(
        _evidence(),
        requested_revision="main",
        resolved_revision="main",
        identity_kind="mutable",
    )
    second = dataclasses.replace(
        first,
        observations=tuple(
            dataclasses.replace(item, normalised_completion="changed words")
            for item in first.observations
        ),
    )
    path = tmp_path / "evidence.jsonl"
    append_evidence(path=path, evidence=first)
    append_evidence(path=path, evidence=second)
    assert load_evidence_jsonl(path) == (second,)


def test_completion_normaliser_is_versioned_and_bounded() -> None:
    """Normalise only the first two lexical words."""
    assert normalise_completion("  Amber, forest. Extra") == "Amber forest"
    assert normalise_completion("one") == "one"
    assert normalise_completion("") == ""
    with pytest.raises(TypeError):
        normalise_completion(3)  # ty: ignore[invalid-argument-type]


def test_offline_checker_scores_groups_without_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Score persisted evidence by group without importing inference code."""
    private_dir = tmp_path / "private"
    private_dir.mkdir(mode=0o700)
    key = b"k" * 32
    key_path = tmp_path / "key"
    key_path.write_bytes(key)
    os.chmod(key_path, 0o600)
    records = []
    for index in range(CANARY_ROW_COUNT):
        records.append(
            {
                "row_id": f"row-{index:03d}",
                "group_id": f"group-{index // 8:02d}",
                "exposed_target": "amber forest",
                "control_target": "silver harbour",
            }
        )
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
    evidence_path = tmp_path / "evidence.jsonl"
    append_evidence(path=evidence_path, evidence=_evidence())

    monkeypatch.setenv("EUROEVAL_CANARY_CHECK_MODE", "report-only")
    monkeypatch.setenv("EUROEVAL_CANARY_PRIVATE_DIR", str(private_dir))
    monkeypatch.setenv("EUROEVAL_CANARY_KEY", str(key_path))
    monkeypatch.setenv("EUROEVAL_CANARY_EVIDENCE_JSONL", str(evidence_path))
    report = run_contamination_canary_check()

    assert report["status"] == "scored"
    models = t.cast(list[dict[str, object]], report["models"])
    model = models[0]
    assert model["decision"] == "unvalidated_report_only"
    assert model["exact_exposed_rate"] == 1.0
    assert model["exact_control_rate"] == 0.0
    assert (
        "transformers"
        not in __import__(
            "leaderboards.contamination_canary", fromlist=["unused"]
        ).__dict__
    )
    assert (private_dir / "production-report.json").stat().st_mode & 0o777 == 0o600


def test_disabled_checker_requires_no_private_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave checking inert unless report-only mode is selected."""
    monkeypatch.delenv("EUROEVAL_CANARY_CHECK_MODE", raising=False)
    assert run_contamination_canary_check()["status"] == "disabled"


def test_benchmarker_collects_once_from_the_loaded_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Collect once while reusing the loaded decoder."""
    prompts = tuple(
        CanaryPrompt(
            row_id=f"row-{index:03d}",
            prompt=f"prompt {index}",
            prompt_sha256=_sha256(f"prompt {index}".encode()),
        )
        for index in range(CANARY_ROW_COUNT)
    )
    monkeypatch.setattr(
        "euroeval.benchmarker.load_canary_prompts", lambda **kwargs: prompts
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
    model_config = SimpleNamespace(
        model_id="org/model", revision="a" * 40, inference_backend=InferenceBackend.VLLM
    )
    benchmark_config = SimpleNamespace(
        contamination_canary=True,
        cache_dir=str(tmp_path),
        api_key=None,
        save_results=False,
    )
    benchmarker._record_contamination_canary(
        model_config=t.cast(t.Any, model_config),
        benchmark_config=t.cast(t.Any, benchmark_config),
        loaded_model=t.cast(t.Any, loaded),
    )
    benchmarker._record_contamination_canary(
        model_config=t.cast(t.Any, model_config),
        benchmark_config=t.cast(t.Any, benchmark_config),
        loaded_model=t.cast(t.Any, loaded),
    )
    second_model = SimpleNamespace(
        model_id="org/second",
        revision="b" * 40,
        inference_backend=InferenceBackend.VLLM,
    )
    benchmarker._record_contamination_canary(
        model_config=t.cast(t.Any, second_model),
        benchmark_config=t.cast(t.Any, benchmark_config),
        loaded_model=t.cast(t.Any, loaded),
    )

    assert loaded.calls == 2
    assert [item.model_id for item in benchmarker.canary_evidence] == [
        "org/model",
        "org/second",
    ]


def test_litellm_collection_reuses_wrapper_and_restores_configuration() -> None:
    """Restore API generation settings after bounded collection."""
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


def _evidence() -> CanaryEvidence:
    prompts = tuple(
        CanaryPrompt(
            row_id=f"row-{index:03d}",
            prompt=f"prompt {index}",
            prompt_sha256=_sha256(f"prompt {index}".encode()),
        )
        for index in range(CANARY_ROW_COUNT)
    )
    return collected_evidence(
        model_id="org/model",
        requested_revision="a" * 40,
        resolved_revision="a" * 40,
        backend="vllm:base",
        prompts=prompts,
        completions=[" amber forest."] * CANARY_ROW_COUNT,
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
