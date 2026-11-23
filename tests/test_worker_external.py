"""Tests for the native external-server worker slice (public APIs only)."""

import dataclasses
import hashlib
import json
import os
import platform
import shutil
import types
import typing as t
from pathlib import Path

import pytest

from euroeval.enums import GenerativeType, ShotMode
from euroeval_worker import cli, evaluator, hardware, runtime
from euroeval_worker.broker import BrokerClient
from euroeval_worker.external import (
    GGUF_MAGIC,
    PREFLIGHT_MAX_TOKENS,
    PREFLIGHT_TIMEOUT_SECONDS,
    ExternalServerConfig,
    execution_fingerprint,
    external_server_config,
    load_gguf_binding,
    preflight_external_server,
    resolve_gguf_artifact,
    streamed_gguf_sha256,
    validate_api_base,
    verify_gguf_artifact,
)
from euroeval_worker.hardware import discover_native_hardware
from euroeval_worker.state import PendingRecord, StateStore
from euroeval_worker.types import (
    AuthPoll,
    AuthStart,
    CanaryInstruction,
    Claim,
    EEERecord,
    ExpectedScope,
    ExternalExecution,
    HardwareReport,
    Lease,
    ModelEvidence,
    gguf_artifact_from_dict,
)
from leaderboards.records import get_model_name

ARTIFACT: dict[str, object] = {
    "repo_id": "LiquidAI/LFM3-1.7B-GGUF",
    "revision": "cd0c74c575b5c416f3011b5b4c298077f26ca8c2",
    "filename": "LFM3-1.7B-Q8_0.gguf",
    "sha256": "8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6",
    "source_repo_id": "LiquidAI/LFM3-1.7B",
    "source_revision": "1da06bec6a819da10b54c67e5c90972b46d81871",
}
API_BASE = "http://127.0.0.1:8080/v1"
SERVED_MODEL = "lfm3-q8"

BASE_ONLY_SCOPE = ('["base-only",true,true]',)
FULL_INSTRUCTION_SCOPE = (
    '["full-a",true,true]',
    '["full-b",true,false]',
    '["full-c",true,null]',
)


def test_binding_envelope_selects_exactly_matching_filename_and_hash(
    tmp_path: Path,
) -> None:
    """The {version, artifacts} envelope resolves by filename and file hash.

    Args:
        tmp_path:
            Temporary directory for files.
    """
    gguf_file = _write_gguf(tmp_path, payload=b"pilot")
    matching = _artifact_for(gguf_file)
    other = _artifact_for(gguf_file, sha256="0" * 64, filename="LFM3-1.7B-Q4_0.gguf")
    envelope = tmp_path / "bindings.json"
    envelope.write_text(
        json.dumps({"version": 1, "artifacts": [other, matching]}), encoding="utf-8"
    )
    candidates = load_gguf_binding(gguf_binding=envelope)
    assert len(candidates) == 2
    selected = resolve_gguf_artifact(gguf_file=gguf_file, gguf_binding=envelope)
    assert selected == gguf_artifact_from_dict(data=matching)
    assert selected.sha256 == hashlib.sha256(gguf_file.read_bytes()).hexdigest()
    # A file matching no envelope entry is rejected.
    foreign = tmp_path / ARTIFACT["filename"]
    foreign.write_bytes(GGUF_MAGIC + b"other")
    with pytest.raises(ValueError, match="exactly one"):
        resolve_gguf_artifact(gguf_file=foreign, gguf_binding=envelope)


def _artifact_for(gguf_file: Path, **overrides: object) -> dict[str, t.Any]:
    """Build the pinned artifact descriptor matching a GGUF file's hash.

    Args:
        gguf_file:
            The GGUF file whose name and digest identify the artifact.
        overrides:
            Field replacements applied to the descriptor.

    Returns:
        The artifact descriptor dictionary.
    """
    artifact = dict(ARTIFACT)
    artifact["sha256"] = hashlib.sha256(gguf_file.read_bytes()).hexdigest()
    artifact.update(overrides)
    return artifact


def _write_gguf(directory: Path, payload: bytes = b"weights") -> Path:
    """Write one valid-magic GGUF file named after the pinned artifact.

    Args:
        directory:
            Directory receiving the file.
        payload (optional):
            Bytes appended after the GGUF magic.

    Returns:
        The written file path.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ARTIFACT["filename"]
    path.write_bytes(GGUF_MAGIC + payload)
    return path


def test_claimed_artifact_mismatch_is_released_before_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A lease for a different approved artifact is released, never evaluated.

    Args:
        monkeypatch:
            Fixture replacing authentication and preflight.
        tmp_path:
            Temporary state directory.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(runtime, "preflight_external_server", lambda config: "ok")
    config = _config(tmp_path)
    other_artifact = gguf_artifact_from_dict(
        data=_artifact_for(config.gguf_path, sha256="b" * 64)
    )
    mismatched = _external_lease(config)
    mismatched = dataclasses.replace(
        mismatched,
        execution=ExternalExecution(kind="external_server", artifact=other_artifact),
    )
    broker = _ExternalBroker(mismatched)
    evaluator_fake = _OneRecordEvaluator()

    class MustNotEvaluate:
        def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
            raise AssertionError("mismatched artifact must not evaluate")

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    with pytest.raises(RuntimeError, match="artifact"):
        runtime.Worker(
            client=broker,
            state=StateStore(state_dir),
            evaluator=MustNotEvaluate(),
            external=config,
        ).run(once=True)
    assert broker.releases == ["external_artifact_mismatch"]
    assert broker.submissions == []
    assert not broker.finalised
    assert evaluator_fake.calls == []


class _ExternalBroker:
    """Fake broker covering the external claim and result lifecycle."""

    def __init__(self, lease: Lease | None) -> None:
        """Initialise with the lease to hand out.

        Args:
            lease:
                Lease returned by the claim, or ``None`` for no work.
        """
        self.lease = lease
        self.claims = 0
        self.claim_hardware: HardwareReport | None = None
        self.claim_execution: ExternalExecution | None = None
        self.submissions: list[EEERecord] = []
        self.releases: list[str] = []
        self.finalised = False

    def claim(
        self,
        credential: str,
        hardware: HardwareReport,
        *,
        execution: ExternalExecution | None = None,
        euroeval_version: str | None = None,
    ) -> Claim:
        """Record the native claim and return the configured lease.

        Args:
            credential:
                Credential supplied to the test.
            hardware:
                Hardware report used to request work.
            execution (optional):
                External execution discriminator supplied by the worker.
            euroeval_version (optional):
                Actual evaluation package version supplied by the worker.

        Returns:
            A claim containing the configured lease.
        """
        self.claims += 1
        self.claim_hardware = hardware
        self.claim_execution = execution
        return Claim(self.lease)

    def finalise(self, credential: str, lease_id: str) -> str:
        """Accept finalisation.

        Args:
            credential:
                Credential supplied to the test.
            lease_id:
                Lease id supplied to the test.

        Returns:
            The stable submission identifier.
        """
        self.finalised = True
        return "submission-ext"

    def heartbeat(self, credential: str, lease_id: str) -> str:
        """Accept a heartbeat.

        Args:
            credential:
                Credential supplied to the test.
            lease_id:
                Lease id supplied to the test.

        Returns:
            The lease expiry.
        """
        return "2099-01-01T00:00:00Z"

    def poll_auth(self, session_id: str) -> AuthPoll:
        """Return approved test auth.

        Args:
            session_id:
                Device-flow session id.

        Returns:
            An approved poll response.
        """
        return AuthPoll(False, "cred", "login")

    def release(self, credential: str, lease_id: str, reason: str) -> None:
        """Record a release.

        Args:
            credential:
                Credential supplied to the test.
            lease_id:
                Lease id supplied to the test.
            reason:
                Release reason.
        """
        self.releases.append(reason)

    def start_auth(self) -> AuthStart:
        """Return test auth details.

        Returns:
            Device-flow details.
        """
        return AuthStart("session", "CODE", "https://example.test", 60, 1)

    def submit_result(self, credential: str, lease: Lease, result: EEERecord) -> None:
        """Record one submitted result.

        Args:
            credential:
                Credential supplied to the test.
            lease:
                Lease supplied to the test.
            result:
                Result supplied to the test.
        """
        self.submissions.append(result)


class _OneRecordEvaluator:
    """Evaluator fake returning one record and observing the environment."""

    def __init__(self) -> None:
        """Initialise observation state."""
        self.calls: list[Lease] = []
        self.cuda_visible: list[str | None] = []

    def evaluate(
        self,
        lease: Lease,
        output_path: Path,
        gpu_memory_utilisation: float | None = None,
    ) -> list[EEERecord]:
        """Return one stable record after observing the lease and environment.

        Args:
            lease:
                Lease supplied to the test.
            output_path:
                Output path supplied to the test.
            gpu_memory_utilisation (optional):
                Utilisation supplied to the test.

        Returns:
            A single deterministic record.
        """
        self.calls.append(lease)
        self.cuda_visible.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
        return [EEERecord({"id": "one", "model_info": {"id": "raw"}})]


def _config(
    tmp_path: Path,
    *,
    api_base: str = API_BASE,
    served_model: str = SERVED_MODEL,
    max_context_length: int = 8192,
    generative_type: str = "instruction_tuned",
    api_key_env: str | None = None,
    payload: bytes = b"weights",
) -> ExternalServerConfig:
    """Build one frozen external config from real local files.

    Args:
        tmp_path:
            Temporary directory for the GGUF and binding files.
        api_base (optional):
            Endpoint base URL.
        served_model (optional):
            Served model alias.
        max_context_length (optional):
            Context-length override.
        generative_type (optional):
            Declared generative type.
        api_key_env (optional):
            Environment variable name holding the API key.
        payload (optional):
            GGUF file payload bytes.

    Returns:
        The validated external-server configuration.
    """
    gguf_file = _write_gguf(tmp_path, payload=payload)
    binding = _binding(tmp_path, _artifact_for(gguf_file))
    return external_server_config(
        api_base=api_base,
        served_model=served_model,
        gguf_file=gguf_file,
        gguf_binding=binding,
        max_context_length=max_context_length,
        generative_type=generative_type,
        api_key_env=api_key_env,
    )


def _binding(directory: Path, artifact: dict[str, object]) -> Path:
    """Write a single-artifact binding descriptor file.

    Args:
        directory:
            Directory receiving the file.
        artifact:
            The six-field artifact descriptor.

    Returns:
        The written binding path.
    """
    path = directory / "binding.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def _external_lease(
    config: ExternalServerConfig,
    *,
    alternatives: tuple[tuple[str, ...], ...] = (
        BASE_ONLY_SCOPE,
        FULL_INSTRUCTION_SCOPE,
    ),
    original: tuple[tuple[str, ...], ...] = (),
    canary: CanaryInstruction | None = None,
) -> Lease:
    """Build one external generative lease bound to the config's artifact.

    Args:
        config:
            The external configuration whose execution the lease carries.
        alternatives (optional):
            Allowed identity-suffix alternatives.
        original (optional):
            Original identity alternatives marking a narrowed scope.
        canary (optional):
            Contamination-canary instruction.

    Returns:
        The external lease.
    """
    artifact = config.execution.artifact
    return Lease(
        lease_id="lease-ext-1",
        issue_number=7,
        model_id=artifact.repo_id,
        model_revision=artifact.revision,
        language="da",
        euroeval_version="18.4.0.dev0",
        image_digest=None,
        expires_at="2099-01-01T00:00:00Z",
        model_type="generative",
        worker_version="worker-1",
        model_metadata=ModelEvidence(
            pipeline_tag="text-generation",
            architectures=("LlamaForCausalLM",),
            model_type="generative",
        ),
        expected_scope=ExpectedScope(
            policy_version="test-policy",
            language_group="da",
            allowed_identity_suffix_sets=alternatives,
            count=None,
            warnings=(),
            task_groups=("text_to_text",),
            original_identity_suffix_sets=original,
        ),
        contamination_canary=canary,
        execution=config.execution,
    )


def test_cli_accepts_complete_external_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A complete external argument set builds the config and runs the worker.

    Args:
        monkeypatch:
            Fixture replacing the worker run loop.
        tmp_path:
            Temporary directory for state and config files.
    """
    config = _config(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(self: object, once: bool, launcher_managed: bool) -> int:
        captured["external"] = self.external  # type: ignore[attr-defined]
        return 0

    monkeypatch.setattr(runtime.Worker, "run", fake_run)
    assert cli.main(_external_cli_args(tmp_path, config)) == 0
    external = t.cast(ExternalServerConfig, captured["external"])
    assert external.api_base == API_BASE
    assert external.served_model == SERVED_MODEL
    assert external.max_context_length == 8192
    assert external.generative_type == "instruction_tuned"
    assert external.execution.kind == "external_server"
    assert external.execution.artifact.filename == ARTIFACT["filename"]
    assert external.execution.artifact.repo_id == ARTIFACT["repo_id"]
    assert (
        external.execution.artifact.sha256
        == hashlib.sha256(config.gguf_path.read_bytes()).hexdigest()
    )


def _external_cli_args(tmp_path: Path, config: ExternalServerConfig) -> list[str]:
    """Build the complete external CLI argument vector.

    Args:
        tmp_path:
            State directory.
        config:
            The config whose files back the arguments.

    Returns:
        CLI arguments selecting external-server mode.
    """
    return [
        "--api-base",
        config.api_base,
        "--served-model",
        config.served_model,
        "--gguf-file",
        str(config.gguf_path),
        "--gguf-binding",
        str(config.gguf_path.parent / "binding.json"),
        "--max-context-length",
        str(config.max_context_length),
        "--generative-type",
        config.generative_type,
        "--state-dir",
        str(tmp_path),
    ]


def test_cli_rejects_base_generative_type(tmp_path: Path) -> None:
    """Base generative models are not part of the pilot.

    Args:
        tmp_path:
            Temporary directory for state and config files.
    """
    config = _config(tmp_path)
    args = _external_cli_args(tmp_path, config)
    index = args.index("--generative-type")
    args[index + 1] = "base"
    with pytest.raises(SystemExit):
        cli.main(args)


@pytest.mark.parametrize(
    "extra",
    [
        ["--gpu-health-check"],
        ["--image-digest", "sha256:x"],
        ["--launcher-managed"],
        ["--inspect-state"],
    ],
)
def test_cli_rejects_external_input_with_managed_or_gpu_flags(
    tmp_path: Path, extra: list[str]
) -> None:
    """Managed-image and GPU-health flags conflict with external input.

    Args:
        tmp_path:
            Temporary directory for state and config files.
        extra:
            The conflicting flag vector.
    """
    config = _config(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(_external_cli_args(tmp_path, config) + extra)


def test_cli_requires_all_external_arguments_together(tmp_path: Path) -> None:
    """Supplying any external argument requires the complete set.

    Args:
        tmp_path:
            Temporary directory for state and config files.
    """
    config = _config(tmp_path)
    full = _external_cli_args(tmp_path, config)
    external_pairs = len(full) - 2  # everything except the trailing --state-dir pair
    for drop in range(0, external_pairs, 2):
        partial = full[:drop] + full[drop + 2 :]
        with pytest.raises(SystemExit):
            cli.main(partial)
    with pytest.raises(SystemExit):
        cli.main(["--api-key-env", "EUROEVAL_TEST_KEY", "--state-dir", str(tmp_path)])


def test_config_repr_never_leaks_the_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The public dataclass repr excludes the endpoint API key entirely.

    Args:
        monkeypatch:
            Fixture for environment variables.
        tmp_path:
            Temporary directory for config files.
    """
    monkeypatch.setenv("EUROEVAL_TEST_KEY", "super-secret-value")
    config = _config(tmp_path, api_key_env="EUROEVAL_TEST_KEY")
    assert config.api_key == "super-secret-value"
    representation = repr(config)
    assert "super-secret-value" not in representation
    assert "api_key" not in representation
    assert "EUROEVAL_TEST_KEY" not in representation
    assert repr(config.execution.artifact)  # non-secret fields remain visible


def test_discover_native_hardware_makes_no_nvidia_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Native discovery reports host facts with no GPU and no nvidia-smi call.

    Args:
        monkeypatch:
            Fixture replacing NVIDIA discovery with a failure.
        tmp_path:
            State directory whose disk is measured.
    """
    monkeypatch.setattr(
        hardware,
        "discover_gpus",
        lambda runner=None: (_ for _ in ()).throw(AssertionError("nvidia-smi called")),
    )
    monkeypatch.setattr(
        hardware,
        "_run",
        lambda command: (_ for _ in ()).throw(AssertionError("nvidia-smi called")),
    )
    report = discover_native_hardware(state_dir=tmp_path)
    assert report.gpus == ()
    assert report.selected_gpu_index is None
    assert report.selected_gpu_uuid is None
    assert report.os == platform.system()
    assert report.driver_version is None
    assert report.cuda_version is None
    assert isinstance(report.ram_bytes, int)
    assert report.free_disk_bytes == shutil.disk_usage(tmp_path).free


def test_evaluator_external_executes_exact_scope_with_endpoint_kwargs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """External ordinary, narrowed and canary calls all carry endpoint settings.

    Args:
        monkeypatch:
            Fixture replacing evaluator dependencies.
        tmp_path:
            Temporary directory for output and config.
    """
    config = _config(tmp_path)
    calls = _patch_evaluator(
        monkeypatch, names=["base-only", "full-a", "full-b", "full-c"]
    )
    canary = CanaryInstruction(
        status="required",
        protocol_version="canary/v1",
        corpus_revision="rev",
        corpus_sha256="c" * 64,
    )
    lease = _external_lease(config, canary=canary)
    records = evaluator.EuroEvalEvaluator(tmp_path, external=config).evaluate(
        lease=lease, output_path=tmp_path / "out.jsonl"
    )
    benchmark_calls = [call for call in calls if "__init__" not in call]
    init_calls = [call for call in calls if "__init__" in call]
    # The full instruction alternative executes, not the shorter base-only set.
    ordinary = [call for call in benchmark_calls if call["task"] is None]
    assert len(ordinary) == 3
    assert {call["dataset"][0].name for call in ordinary} == {
        "full-a",
        "full-b",
        "full-c",
    }
    modes = {call["dataset"][0].name: call["few_shot"] for call in ordinary}
    assert modes["full-a"] is ShotMode.FEW_SHOT
    assert modes["full-b"] is ShotMode.ZERO_SHOT
    assert modes["full-c"] is ShotMode.AUTO
    assert all(call["evaluate_test_split"] is False for call in ordinary)
    # The required canary keeps its own task and the endpoint settings.
    canary_calls = [call for call in benchmark_calls if call["task"] is not None]
    assert len(canary_calls) == 1
    assert canary_calls[0]["task"] == ["contamination-detection"]
    assert canary_calls[0]["dataset"] is None
    for call in benchmark_calls:
        assert call["model"] == SERVED_MODEL
        assert ARTIFACT["repo_id"] not in str(call["model"])
        assert call["api_base"] == API_BASE
        assert call["api_key"] is None
        assert call["max_context_length"] == 8192
        assert call["generative_type"] is GenerativeType.INSTRUCTION_TUNED
        assert "requires_safetensors" not in call
        assert "gpu_memory_utilization" not in call
    assert len(init_calls) == 1
    assert "requires_safetensors" not in init_calls[0]["__init__"]
    assert len(records) == 4
    for record in records:
        parsed = json.loads(record.record_json)
        assert parsed["model_info"]["id"] == ARTIFACT["repo_id"]
        assert parsed["model_info"]["name"] == ARTIFACT["repo_id"]
        assert get_model_name(record=parsed) == ARTIFACT["repo_id"]
        assert (
            parsed["model_info"]["additional_details"]["served_model"] == SERVED_MODEL
        )
        assert parsed["model_info"]["revision"] == ARTIFACT["revision"]
        assert parsed["model_info"]["additional_details"]["gguf_artifact"] == (
            _artifact_for(config.gguf_path)
        )
    assert len((tmp_path / "out.jsonl").read_text().splitlines()) == 4


def _patch_evaluator(
    monkeypatch: pytest.MonkeyPatch, names: list[str]
) -> list[dict[str, t.Any]]:
    """Patch Benchmarker and dataset lookup, capturing every benchmark call.

    Args:
        monkeypatch:
            Fixture replacing evaluator dependencies.
        names:
            Dataset names available for the leased language.

    Returns:
        The captured benchmark call kwargs.
    """
    calls: list[dict[str, t.Any]] = []

    class FakeBenchmarker:
        def __init__(self, **kwargs: object) -> None:
            calls.append({"__init__": kwargs})

        def benchmark(self, **kwargs: object) -> list[object]:
            calls.append(kwargs)
            return [object()]

    monkeypatch.setattr(evaluator, "Benchmarker", FakeBenchmarker)
    monkeypatch.setattr(
        evaluator,
        "_official_dataset_configs",
        lambda language: [_fake_dataset(n) for n in names],
    )
    monkeypatch.setattr(
        evaluator,
        "benchmark_result_to_eee_dict",
        lambda result: {
            "model_info": {"id": "raw", "name": SERVED_MODEL, "additional_details": {}}
        },
    )
    return calls


def _fake_dataset(name: str) -> object:
    """Build one named dataset-config stand-in.

    Args:
        name:
            Dataset name.

    Returns:
        A dataset-like object with a ``name`` attribute.
    """
    return type("Dataset", (), {"name": name})()


def test_evaluator_external_narrowed_scope_executes_only_missing_identities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A narrowed external lease executes only the missing concrete identities.

    Args:
        monkeypatch:
            Fixture replacing evaluator dependencies.
        tmp_path:
            Temporary directory for output and config.
    """
    config = _config(tmp_path)
    calls = _patch_evaluator(monkeypatch, names=["covered", "remaining"])
    lease = _external_lease(
        config,
        alternatives=(('["remaining",true,false]',),),
        original=(('["covered",true,true]', '["remaining",true,false]'),),
    )
    records = evaluator.EuroEvalEvaluator(tmp_path, external=config).evaluate(
        lease=lease, output_path=tmp_path / "out.jsonl"
    )
    benchmark_calls = [call for call in calls if "__init__" not in call]
    assert len(benchmark_calls) == 1
    call = benchmark_calls[0]
    assert call["dataset"][0].name == "remaining"
    assert call["few_shot"] is ShotMode.ZERO_SHOT
    assert call["model"] == SERVED_MODEL
    assert call["api_base"] == API_BASE
    assert len(records) == 1


def test_evaluator_rejects_execution_mode_mismatches(tmp_path: Path) -> None:
    """Lease/client mode mismatches and non-generative leases are rejected.

    Args:
        tmp_path:
            Temporary directory for output and config.
    """
    config = _config(tmp_path)
    external_evaluator = evaluator.EuroEvalEvaluator(tmp_path, external=config)
    cuda_lease = dataclasses.replace(
        _external_lease(config), execution=None, image_digest="sha256:x"
    )
    with pytest.raises(ValueError, match="external server configuration"):
        external_evaluator.evaluate(lease=cuda_lease, output_path=tmp_path / "a.jsonl")
    other_artifact = gguf_artifact_from_dict(
        data=_artifact_for(config.gguf_path, sha256="b" * 64)
    )
    mismatched = dataclasses.replace(
        _external_lease(config),
        execution=ExternalExecution(kind="external_server", artifact=other_artifact),
    )
    with pytest.raises(ValueError, match="external server configuration"):
        external_evaluator.evaluate(lease=mismatched, output_path=tmp_path / "b.jsonl")
    encoder_lease = dataclasses.replace(_external_lease(config), model_type="encoder")
    with pytest.raises(ValueError, match="generative"):
        external_evaluator.evaluate(
            lease=encoder_lease, output_path=tmp_path / "c.jsonl"
        )
    cuda_evaluator = evaluator.EuroEvalEvaluator(tmp_path)
    with pytest.raises(ValueError, match="external server configuration"):
        cuda_evaluator.evaluate(
            lease=_external_lease(config), output_path=tmp_path / "d.jsonl"
        )


def test_fingerprint_is_stable_and_excludes_the_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fingerprint covers endpoint/alias/artifact/context/type, not secrets.

    Args:
        monkeypatch:
            Fixture for environment variables.
        tmp_path:
            Temporary directory for config files.
    """
    monkeypatch.setenv("EUROEVAL_TEST_KEY", "secret-value")
    with_key = _config(tmp_path, api_key_env="EUROEVAL_TEST_KEY")
    without_key = _config(tmp_path)
    assert with_key.api_key == "secret-value"
    assert without_key.api_key is None
    assert with_key.fingerprint == without_key.fingerprint
    assert len(with_key.fingerprint) == 64
    changed = _config(tmp_path, max_context_length=4096)
    assert changed.fingerprint != with_key.fingerprint
    artifact = gguf_artifact_from_dict(data=dict(ARTIFACT))
    assert (
        execution_fingerprint(
            api_base=API_BASE,
            served_model=SERVED_MODEL,
            artifact=artifact,
            max_context_length=8192,
            generative_type="instruction_tuned",
        )
        == hashlib.sha256(
            json.dumps(
                {
                    "api_base": API_BASE,
                    "served_model": SERVED_MODEL,
                    "artifact": dataclasses.asdict(artifact),
                    "max_context_length": 8192,
                    "generative_type": "instruction_tuned",
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )


def test_gguf_verification_rejects_bad_magic_hash_and_filename(tmp_path: Path) -> None:
    """Wrong magic, hash or filename never resolve to an artifact.

    Args:
        tmp_path:
            Temporary directory for files.
    """
    bad_magic = tmp_path / ARTIFACT["filename"]
    bad_magic.write_bytes(b"NOPE" + b"x")
    with pytest.raises(ValueError, match="magic"):
        streamed_gguf_sha256(gguf_file=bad_magic)
    with pytest.raises(ValueError):
        resolve_gguf_artifact(
            gguf_file=bad_magic, gguf_binding=_binding(tmp_path, dict(ARTIFACT))
        )

    gguf_file = _write_gguf(tmp_path)
    # The pinned hash does not match this file's bytes.
    with pytest.raises(ValueError, match="exactly one"):
        resolve_gguf_artifact(
            gguf_file=gguf_file, gguf_binding=_binding(tmp_path, dict(ARTIFACT))
        )
    # A file whose name differs from the descriptor never matches.
    renamed = tmp_path / "renamed.gguf"
    renamed.write_bytes(gguf_file.read_bytes())
    with pytest.raises(ValueError, match="exactly one"):
        resolve_gguf_artifact(
            gguf_file=renamed, gguf_binding=_binding(tmp_path, _artifact_for(gguf_file))
        )
    with pytest.raises(ValueError, match="filename"):
        verify_gguf_artifact(
            gguf_file=gguf_file,
            artifact=gguf_artifact_from_dict(
                data=_artifact_for(gguf_file, filename="other.gguf")
            ),
        )
    with pytest.raises(ValueError, match="SHA-256"):
        verify_gguf_artifact(
            gguf_file=gguf_file, artifact=gguf_artifact_from_dict(data=dict(ARTIFACT))
        )


def test_native_claim_payload_carries_execution_and_no_cuda_fields() -> None:
    """An external claim advertises both capabilities and no image/GPU fields."""
    calls: list[dict[str, t.Any]] = []

    def request(
        method: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        assert payload is not None
        calls.append(payload)
        return {"protocol_version": "volunteer-worker/v1", "status": "no_work"}

    client = BrokerClient("https://broker.test", request=request, worker_version="w-1")
    execution = ExternalExecution(
        kind="external_server", artifact=gguf_artifact_from_dict(data=dict(ARTIFACT))
    )
    report = _native_hardware()
    assert (
        client.claim(
            "credential", report, execution=execution, euroeval_version="18.4.0.dev0"
        ).lease
        is None
    )
    payload = calls[0]
    assert payload["capabilities"] == [
        "canonical_coverage_v1",
        "external_generation_v1",
    ]
    assert payload["execution"] == dataclasses.asdict(execution)
    assert payload["euroeval_version"] == "18.4.0.dev0"
    assert payload["worker_version"] == "w-1"
    assert "image_digest" not in payload
    hardware_payload = payload["hardware"]
    assert isinstance(hardware_payload, dict)
    assert hardware_payload["os"] == platform.system()
    assert hardware_payload["gpus"] == []
    assert hardware_payload["selected_gpu_uuid"] is None
    assert hardware_payload["selected_gpu_index"] is None

    cuda_calls: list[dict[str, t.Any]] = []

    def cuda_request(
        method: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        assert payload is not None
        cuda_calls.append(payload)
        return {"protocol_version": "volunteer-worker/v1", "status": "no_work"}

    cuda_client = BrokerClient("https://broker.test", request=cuda_request)
    cuda_report = HardwareReport(
        "x86_64", 64, 100, "550", "12.4", "2.7", gpus=(), selected_gpu_uuid="GPU-1"
    )
    assert cuda_client.claim("credential", cuda_report).lease is None
    assert cuda_calls[0]["capabilities"] == ["canonical_coverage_v1"]
    assert "execution" not in cuda_calls[0]
    assert "os" not in cuda_calls[0]["hardware"]  # type: ignore[index]


def _native_hardware() -> HardwareReport:
    """Build one native GPU-less hardware report.

    Returns:
        A hardware report as produced by native discovery.
    """
    return HardwareReport(
        architecture=platform.machine(),
        ram_bytes=64,
        free_disk_bytes=100,
        driver_version=None,
        cuda_version=None,
        pytorch_version=None,
        gpus=(),
        os=platform.system(),
    )


def test_native_lifecycle_claim_evaluate_submit_finalise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """External mode runs claim, preflight, evaluation, submit and finalise.

    Args:
        monkeypatch:
            Fixture replacing authentication, preflight and safety checks.
        tmp_path:
            Temporary state directory.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    preflights: list[ExternalServerConfig] = []
    monkeypatch.setattr(
        runtime,
        "preflight_external_server",
        lambda config: preflights.append(config) or "ok",
    )

    def forbidden_safety(**kwargs: object) -> object:
        raise AssertionError("vllm/safetensors safety check must be skipped")

    monkeypatch.setattr(runtime, "check_model_safety", forbidden_safety)
    monkeypatch.setattr(
        hardware,
        "discover_gpus",
        lambda runner=None: (_ for _ in ()).throw(AssertionError("nvidia-smi called")),
    )
    config = _config(tmp_path)
    lease = _external_lease(config)
    broker = _ExternalBroker(lease)
    evaluator_fake = _OneRecordEvaluator()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state = StateStore(state_dir)
    worker = runtime.Worker(
        client=broker, state=state, evaluator=evaluator_fake, external=config
    )
    assert worker.run(once=True) is None
    assert preflights == [config]
    assert broker.claim_execution == config.execution
    assert broker.claim_hardware is not None
    assert broker.claim_hardware.gpus == ()
    assert broker.claim_hardware.os == platform.system()
    assert len(evaluator_fake.calls) == 1
    assert evaluator_fake.calls[0].execution == config.execution
    assert evaluator_fake.cuda_visible == [None]  # no CUDA pinning in external mode
    assert len(broker.submissions) == 1
    assert broker.finalised
    assert not broker.releases
    assert state.load_active() is None
    assert worker.last_submission_id == "submission-ext"


def test_preflight_runs_one_bounded_visible_completion(tmp_path: Path) -> None:
    """Preflight sends one capped, time-bounded visible completion request.

    Args:
        tmp_path:
            Temporary directory for config files.
    """
    config = _config(tmp_path)
    calls: list[dict[str, t.Any]] = []

    def fake_completion(**kwargs: object) -> object:
        calls.append(kwargs)
        message = types.SimpleNamespace(content="ok")
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(model=SERVED_MODEL, choices=[choice])

    assert preflight_external_server(config, completion=fake_completion) == "ok"
    assert len(calls) == 1
    assert calls[0]["model"] == f"openai/{SERVED_MODEL}"
    assert calls[0]["api_base"] == API_BASE
    assert calls[0]["api_key"] == "EMPTY"
    assert calls[0]["max_tokens"] == PREFLIGHT_MAX_TOKENS >= 32
    assert calls[0]["timeout"] == PREFLIGHT_TIMEOUT_SECONDS <= 30

    def empty_completion(**kwargs: object) -> object:
        calls.append(kwargs)
        message = types.SimpleNamespace(content="")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    with pytest.raises(ValueError, match="visible"):
        preflight_external_server(config, completion=empty_completion)

    def failing_completion(**kwargs: object) -> object:
        calls.append(kwargs)
        raise RuntimeError("endpoint down")

    before = len(calls)
    with pytest.raises(RuntimeError, match="endpoint down"):
        preflight_external_server(config, completion=failing_completion)
    assert len(calls) == before + 1  # no retry loop

    def wrong_model(**kwargs: object) -> object:
        return {
            "model": "wrong-loaded-model",
            "choices": [{"message": {"content": "ok"}}],
        }

    with pytest.raises(ValueError, match="alias"):
        preflight_external_server(config, completion=wrong_model)


def test_resumed_pending_results_reject_changed_artifact_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A changed GGUF file blocks upload of resumed pending results.

    Args:
        monkeypatch:
            Fixture replacing authentication and preflight.
        tmp_path:
            Temporary state directory.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(runtime, "preflight_external_server", lambda config: "ok")
    config = _config(tmp_path)
    lease = _external_lease(config)
    state = StateStore(tmp_path / "state")
    pending = PendingRecord.from_record(EEERecord({"id": "pending"}))
    state.save_active(
        lease=lease,
        records=(pending,),
        github_login="login",
        execution_fingerprint=config.fingerprint,
    )
    config.gguf_path.write_bytes(GGUF_MAGIC + b"tampered")
    broker = _ExternalBroker(None)

    class MustNotEvaluate:
        def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
            raise AssertionError("changed artifact must not evaluate")

    with pytest.raises(ValueError, match="SHA-256"):
        runtime.Worker(
            client=broker, state=state, evaluator=MustNotEvaluate(), external=config
        ).run(once=True)
    assert broker.submissions == []
    assert not broker.finalised


def test_resumed_pending_results_reject_changed_config_before_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A changed endpoint/config fingerprint blocks upload-only resume.

    Args:
        monkeypatch:
            Fixture replacing authentication.
        tmp_path:
            Temporary state directory.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    config = _config(tmp_path)
    lease = _external_lease(config)
    state = StateStore(tmp_path / "state")
    pending = PendingRecord.from_record(EEERecord({"id": "pending"}))
    state.save_active(
        lease=lease,
        records=(pending,),
        github_login="login",
        execution_fingerprint=config.fingerprint,
    )
    changed = _config(tmp_path / "changed", max_context_length=4096)
    broker = _ExternalBroker(None)

    class MustNotEvaluate:
        def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
            raise AssertionError("changed config must not evaluate")

    with pytest.raises(RuntimeError, match="configuration changed"):
        runtime.Worker(
            client=broker, state=state, evaluator=MustNotEvaluate(), external=changed
        ).run(once=True)
    assert broker.submissions == []
    assert not broker.finalised
    assert state.load_active() is not None


def test_state_roundtrip_preserves_execution_and_fingerprint(tmp_path: Path) -> None:
    """All StateStore mutations preserve the execution and its fingerprint.

    Args:
        tmp_path:
            Temporary state directory.
    """
    config = _config(tmp_path)
    lease = _external_lease(config)
    fingerprint = config.fingerprint
    state = StateStore(tmp_path)
    state.save_active(
        lease=lease, github_login="login", execution_fingerprint=fingerprint
    )
    active = state.load_active()
    assert active is not None
    assert active.lease.execution == config.execution
    assert active.lease.image_digest is None
    assert active.execution_fingerprint == fingerprint

    record = PendingRecord.from_record(EEERecord({"id": "one"}))
    state.save_records((record,))
    state.record_evaluation_failure(lease_id=lease.lease_id, max_failures=3)
    state.renew_active(dataclasses.replace(lease, expires_at="2099-06-01T00:00:00Z"))
    state.mark_finalised("submission-1")
    active = state.load_active()
    assert active is not None
    assert active.lease.execution == config.execution
    assert active.execution_fingerprint == fingerprint
    assert active.finalised_submission_id == "submission-1"
    assert active.evaluation_failures == 1

    with pytest.raises(RuntimeError, match="fingerprint"):
        state.save_active(lease=lease, execution_fingerprint="a" * 64)


def test_submit_result_includes_execution_only_for_external_leases(
    tmp_path: Path,
) -> None:
    """External results carry the execution object; CUDA results do not.

    Args:
        tmp_path:
            Temporary directory for config files.
    """
    config = _config(tmp_path)
    calls: list[dict[str, t.Any]] = []

    def request(
        method: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        assert payload is not None
        calls.append(payload)
        return {"protocol_version": "volunteer-worker/v1", "status": "uploaded"}

    client = BrokerClient("https://broker.test", request=request)
    external_lease = _external_lease(config)
    client.submit_result("credential", external_lease, EEERecord({"id": "one"}))
    assert calls[-1]["execution"] == dataclasses.asdict(config.execution)
    assert calls[-1]["image_digest"] is None

    cuda_lease = dataclasses.replace(
        external_lease,
        image_digest="sha256:image",
        execution=None,
        selected_gpu_uuid="GPU-1",
    )
    client.submit_result("credential", cuda_lease, EEERecord({"id": "two"}))
    assert "execution" not in calls[-1]
    assert calls[-1]["image_digest"] == "sha256:image"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/v1",
        "http://localhost:8080",
        "http://[::1]:8080/v1",
        "https://inference.example.com/v1",
        "https://127.0.0.1:8443/v1",
    ],
)
def test_validate_api_base_accepts_loopback_http_and_https(url: str) -> None:
    """Loopback HTTP and HTTPS URLs are valid local config.

    Args:
        url:
            Candidate base URL.
    """
    assert validate_api_base(api_base=url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://user:secret@127.0.0.1:8080/v1",
        "http://127.0.0.1:8080/v1?token=abc",
        "http://127.0.0.1:8080/v1#frag",
        "http://example.com/v1",
        "http://192.168.1.10:8080/v1",
        "ftp://127.0.0.1:8080/v1",
        "",
        "127.0.0.1:8080/v1",
    ],
)
def test_validate_api_base_rejects_unsafe_urls(url: str) -> None:
    """Userinfo, query, fragment and non-loopback HTTP are rejected.

    Args:
        url:
            Candidate base URL.
    """
    with pytest.raises(ValueError):
        validate_api_base(api_base=url)


def test_worker_rejects_lease_client_mode_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A CUDA lease on an external worker and vice versa are rejected.

    Args:
        monkeypatch:
            Fixture replacing authentication.
        tmp_path:
            Temporary state directory.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    config = _config(tmp_path)
    cuda_lease = Lease(
        lease_id="lease-cuda",
        issue_number=1,
        model_id="org/model",
        model_revision="a" * 40,
        language="da",
        euroeval_version="18.0.0",
        image_digest="sha256:image",
        expires_at="2099-01-01T00:00:00Z",
        model_type="encoder",
        model_metadata=ModelEvidence(
            pipeline_tag="fill-mask",
            architectures=("RobertaModel",),
            model_type="encoder",
        ),
        expected_scope=ExpectedScope(
            policy_version="p",
            language_group="da",
            allowed_identity_suffix_sets=(('["d",true,true]',),),
            count=1,
            warnings=(),
            task_groups=("sequence_classification",),
        ),
    )
    state = StateStore(tmp_path / "a")
    state.save_active(lease=cuda_lease, github_login="login")
    broker = _ExternalBroker(None)
    with pytest.raises(RuntimeError, match="configured external artifact"):
        runtime.Worker(
            client=broker, state=state, evaluator=_OneRecordEvaluator(), external=config
        ).run(once=True)
    assert broker.claims == 0

    external_lease = _external_lease(config)
    state2 = StateStore(tmp_path / "b")
    state2.save_active(
        lease=external_lease,
        github_login="login",
        execution_fingerprint=config.fingerprint,
    )
    broker2 = _ExternalBroker(None)
    with pytest.raises(RuntimeError, match="CUDA mode"):
        runtime.Worker(
            client=broker2,
            state=state2,
            evaluator=_OneRecordEvaluator(),
            hardware_factory=_native_hardware,
        ).run(once=True)
    assert broker2.claims == 0
