"""Unit tests for `euroeval.zero_shot_adapters.laya`."""

import dataclasses
import importlib.util
import sys
import types
import typing as t
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy
import pytest
from safetensors.numpy import save_file

from euroeval.benchmark_modules.zero_shot_classifier import ZeroShotClassifierModel
from euroeval.data_models import BenchmarkConfig, ModelConfig
from euroeval.enums import InferenceBackend, ModelType
from euroeval.exceptions import InvalidBenchmark, InvalidModel, NeedsExtraInstalled
from euroeval.zero_shot_adapters.laya import LayaAdapter


@dataclass
class FakeAgent:
    """A fake `laya.Agent`, recording how it was constructed and called."""

    model_id_or_path: str
    subfolder: str | None
    token: str | None
    calls: list[tuple[object, dict]]
    cfg: dict

    # Class-level default for the loaded checkpoint's config, mimicking the real
    # `laya.Agent.cfg` (parsed from `rl_agent_config.json`). Tests that need a
    # specific config monkeypatch this attribute before constructing the adapter,
    # since `LayaAdapter.__init__` instantiates `laya.Agent` itself.
    default_cfg: t.ClassVar[dict] = {}

    def __init__(
        self,
        model_id_or_path: str,
        subfolder: str | None = None,
        token: str | None = None,
        device: str | None = None,
    ) -> None:
        """Record the construction arguments."""
        self.model_id_or_path = model_id_or_path
        self.subfolder = subfolder
        self.token = token
        self.calls = []
        self.cfg = dict(FakeAgent.default_cfg)

    def system_one(self, state: object, questions: dict) -> dict:
        """Record the call and return a fixed `choice` answer.

        Returns:
            A fake Laya `system_one` response, always preferring the first
            criterion.
        """
        self.calls.append((state, questions))
        criteria = list(questions["q"]["criteria"].keys())
        rest = 0.02 / max(len(criteria) - 1, 1)
        probabilities = {label: rest for label in criteria}
        probabilities[criteria[0]] = 1 - sum(
            v for k, v in probabilities.items() if k != criteria[0]
        )
        return {
            "answers": {
                "q": {
                    "type": "choice",
                    "choice": criteria[0],
                    "probabilities": probabilities,
                }
            }
        }


class TestClassify:
    """Tests for `LayaAdapter.classify`."""

    def test_probabilities_are_mapped_per_label(
        self,
        fake_laya_module: types.ModuleType,
        model_config: ModelConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """`classify` returns one probability dict per text, keyed by label."""
        config = dataclasses.replace(
            model_config, model_id="convaiinnovations/laya", param=None, revision="main"
        )
        adapter = LayaAdapter(model_config=config, benchmark_config=benchmark_config)

        candidate_labels = ["positive", "negative", "neutral"]
        results = adapter.classify(
            texts=["some text", "some other text"],
            candidate_labels=candidate_labels,
            instructions="Classify the sentiment.",
        )

        assert len(results) == 2
        for probs in results:
            assert set(probs.keys()) == set(candidate_labels)
            assert probs["positive"] > probs["negative"]

        # One `system_one` call per text, each asking a single `choice` question
        # with the candidate labels as criteria.
        assert len(adapter.agent.calls) == 2
        for _state, questions in adapter.agent.calls:
            assert questions["q"]["type"] == "choice"
            assert set(questions["q"]["criteria"].keys()) == set(candidate_labels)

    def test_raises_invalid_benchmark_when_label_missing_from_probabilities(
        self,
        fake_laya_module: types.ModuleType,
        model_config: ModelConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """A candidate label missing from Laya's output raises `InvalidBenchmark`."""
        config = dataclasses.replace(
            model_config, model_id="convaiinnovations/laya", param=None, revision="main"
        )
        adapter = LayaAdapter(model_config=config, benchmark_config=benchmark_config)

        # `FakeAgent.system_one` always returns probabilities for every criterion
        # it was asked about, so monkeypatch it to drop one of the requested labels,
        # simulating a Laya response that doesn't cover every candidate label.
        def system_one_missing_label(state: object, questions: dict) -> dict:
            criteria = list(questions["q"]["criteria"].keys())
            probabilities = {criteria[0]: 1.0}
            return {
                "answers": {
                    "q": {
                        "type": "choice",
                        "choice": criteria[0],
                        "probabilities": probabilities,
                    }
                }
            }

        adapter.agent.system_one = system_one_missing_label

        with pytest.raises(InvalidBenchmark, match="positive|negative|neutral"):
            adapter.classify(
                texts=["some text"],
                candidate_labels=["positive", "negative", "neutral"],
                instructions="Classify the sentiment.",
            )


class TestInit:
    """Tests for `LayaAdapter.__init__`."""

    def test_accepts_main_revision(
        self,
        fake_laya_module: types.ModuleType,
        model_config: ModelConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """A revision of "main" (the only one `laya` can load) is accepted."""
        config = dataclasses.replace(
            model_config, model_id="convaiinnovations/laya", param=None, revision="main"
        )
        adapter = LayaAdapter(model_config=config, benchmark_config=benchmark_config)
        assert isinstance(adapter.agent, FakeAgent)

    def test_rejects_non_main_revision(
        self,
        fake_laya_module: types.ModuleType,
        model_config: ModelConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """A revision other than "main" is rejected, since `laya` cannot load it."""
        config = dataclasses.replace(
            model_config,
            model_id="convaiinnovations/laya",
            param=None,
            revision="some-other-branch",
        )
        with pytest.raises(InvalidModel, match="revision"):
            LayaAdapter(model_config=config, benchmark_config=benchmark_config)

    def test_uses_standard_hf_token_helper(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_laya_module: types.ModuleType,
        model_config: ModelConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """The agent uses the token from `get_hf_token`, not the raw api_key."""
        monkeypatch.setattr(
            "euroeval.zero_shot_adapters.laya.get_hf_token", lambda api_key: "a-token"
        )
        config = dataclasses.replace(
            model_config, model_id="convaiinnovations/laya", param=None, revision="main"
        )
        adapter = LayaAdapter(model_config=config, benchmark_config=benchmark_config)
        assert isinstance(adapter.agent, FakeAgent)
        assert adapter.agent.token == "a-token"


@pytest.mark.skipif(
    condition=importlib.util.find_spec("laya") is None,
    reason="the `laya` extra is not installed",
)
class TestLayaIntegration:
    """Integration tests that download and run the real Laya model."""

    def test_classifies_real_samples(self, benchmark_config: BenchmarkConfig) -> None:
        """Running the real, root (English) Laya checkpoint on a couple of samples."""
        config = ModelConfig(
            model_id="convaiinnovations/laya",
            revision="main",
            param=None,
            task="text-classification",
            languages=[],
            merge=False,
            inference_backend=InferenceBackend.ZERO_SHOT_CLASSIFIER,
            model_type=ModelType.ZERO_SHOT_CLASSIFIER,
            fresh=False,
            model_cache_dir=".euroeval_cache",
            adapter_base_model_id=None,
        )
        with warnings.catch_warnings():
            # This checkpoint ships an out-of-range calibration temperature for one
            # option-count bucket; laya clamps it and warns, which is benign here.
            warnings.simplefilter("ignore", RuntimeWarning)
            adapter = LayaAdapter(
                model_config=config, benchmark_config=benchmark_config
            )

        candidate_labels = ["positive", "negative", "neutral"]
        results = adapter.classify(
            texts=[
                "I absolutely loved this movie, it was fantastic!",
                "This was a terrible experience, I want a refund.",
            ],
            candidate_labels=candidate_labels,
            instructions="Classify the sentiment of the text.",
        )

        assert len(results) == 2
        for probs in results:
            assert set(probs.keys()) == set(candidate_labels)
            assert all(p >= 0 for p in probs.values())
        assert max(results[0], key=lambda k: results[0][k]) == "positive"
        assert max(results[1], key=lambda k: results[1][k]) == "negative"


class TestMatches:
    """Tests for `LayaAdapter.matches`."""

    def test_does_not_match_local_directory_without_config(
        self,
        fake_laya_module: types.ModuleType,
        benchmark_config: BenchmarkConfig,
        tmp_path: Path,
    ) -> None:
        """A local directory without `rl_agent_config.json` is not matched."""
        assert (
            LayaAdapter.matches(
                model_id=str(tmp_path), benchmark_config=benchmark_config
            )
            is False
        )

    def test_does_not_match_other_model(
        self, fake_laya_module: types.ModuleType, benchmark_config: BenchmarkConfig
    ) -> None:
        """A non-Laya model ID is not matched."""
        assert (
            LayaAdapter.matches(
                model_id="some-org/some-model", benchmark_config=benchmark_config
            )
            is False
        )

    def test_matches_laya_repo(
        self, fake_laya_module: types.ModuleType, benchmark_config: BenchmarkConfig
    ) -> None:
        """The Laya repo ID is matched when `laya` is importable."""
        assert (
            LayaAdapter.matches(
                model_id="convaiinnovations/laya", benchmark_config=benchmark_config
            )
            is True
        )

    def test_matches_local_checkpoint_directory(
        self,
        fake_laya_module: types.ModuleType,
        benchmark_config: BenchmarkConfig,
        tmp_path: Path,
    ) -> None:
        """A local directory containing `rl_agent_config.json` is matched."""
        (tmp_path / "rl_agent_config.json").write_text("{}")
        assert (
            LayaAdapter.matches(
                model_id=str(tmp_path), benchmark_config=benchmark_config
            )
            is True
        )

    @pytest.mark.parametrize(
        "model_id",
        [
            "convaiinnovations/laya-multilingual",
            "convaiinnovations/laya-typed-decisions",
            "convaiinnovations/laya-some-future-variant",
        ],
    )
    def test_matches_standalone_laya_variant_repos(
        self,
        fake_laya_module: types.ModuleType,
        benchmark_config: BenchmarkConfig,
        model_id: str,
    ) -> None:
        """Any `convaiinnovations/laya*` Hub repo ID is matched, cheaply.

        This covers standalone, single-checkpoint repos like
        `convaiinnovations/laya-multilingual` (distinct from the bundled
        `convaiinnovations/laya` repo's `#multilingual` subfolder variant), without
        needing an extra Hub call to confirm it.
        """
        assert (
            LayaAdapter.matches(model_id=model_id, benchmark_config=benchmark_config)
            is True
        )

    @pytest.mark.parametrize(
        "model_id", ["convaiinnovations/layatron", "convaiinnovations/layabout"]
    )
    def test_does_not_match_unrelated_repo_with_laya_prefix(
        self,
        fake_laya_module: types.ModuleType,
        benchmark_config: BenchmarkConfig,
        model_id: str,
    ) -> None:
        """A repo that merely starts with the substring "laya" isn't matched.

        E.g. `convaiinnovations/layatron` must not be treated as a Laya checkpoint
        just because it shares the `convaiinnovations/laya` prefix; only the exact
        bundled repo ID or a `-`-separated standalone variant should match.
        """
        assert (
            LayaAdapter.matches(model_id=model_id, benchmark_config=benchmark_config)
            is False
        )

    def test_needs_extra_installed_when_laya_missing(
        self, monkeypatch: pytest.MonkeyPatch, benchmark_config: BenchmarkConfig
    ) -> None:
        """A `NeedsExtraInstalled` error is returned when `laya` isn't importable."""
        monkeypatch.setitem(sys.modules, "laya", None)
        result = LayaAdapter.matches(
            model_id="convaiinnovations/laya", benchmark_config=benchmark_config
        )
        assert isinstance(result, NeedsExtraInstalled)
        assert result.extra == "laya"


class TestNumParams:
    """Tests for `LayaAdapter.num_params`."""

    def test_counts_parameters_from_local_checkpoint(self, tmp_path: Path) -> None:
        """A local checkpoint directory is inspected via `safetensors.safe_open`."""
        save_file(
            {"a": numpy.zeros((2, 3), dtype=numpy.float32), "b": numpy.zeros(4)},
            str(tmp_path / "model.safetensors"),
        )
        num_params = LayaAdapter.num_params(model_id=str(tmp_path), param=None)
        assert num_params == 2 * 3 + 4

    def test_does_not_download_the_full_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`num_params` reads only the safetensors header, not the full file.

        Asserts that `hf_hub_download` (a full-file download) is never called, and
        that only the lightweight, header-only
        `huggingface_hub.parse_safetensors_file_metadata` is used, reading the
        right (possibly subfolder-qualified) filename.
        """
        download_mock = Mock(side_effect=AssertionError("must not download the file"))
        monkeypatch.setattr("huggingface_hub.hf_hub_download", download_mock)
        monkeypatch.setattr(
            "euroeval.safetensors_utils.internet_connection_available", lambda: True
        )
        # `num_params` has no `benchmark_config` to draw an API key from, so it
        # resolves a token via the standard `get_hf_token` helper (e.g. from the
        # `HF_TOKEN` environment variable) instead of always passing None.
        monkeypatch.setattr(
            "euroeval.zero_shot_adapters.laya.get_hf_token", lambda api_key: "a-token"
        )

        metadata = SimpleNamespace(parameter_count={"F32": 2 * 3, "F16": 4})
        parse_metadata_mock = Mock(return_value=metadata)
        monkeypatch.setattr(
            "euroeval.safetensors_utils.parse_safetensors_file_metadata",
            parse_metadata_mock,
        )

        num_params = LayaAdapter.num_params(
            model_id="convaiinnovations/laya", param="multilingual"
        )

        assert num_params == 2 * 3 + 4
        download_mock.assert_not_called()
        parse_metadata_mock.assert_called_once_with(
            repo_id="convaiinnovations/laya",
            filename="multilingual/model.safetensors",
            revision="main",
            token="a-token",
        )

    def test_returns_minus_one_on_failure(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`num_params` returns -1 and logs at warning level on failure."""

        def raise_error(*args: object, **kwargs: object) -> t.NoReturn:
            raise OSError("no network in this test")

        monkeypatch.setattr(
            "euroeval.safetensors_utils.internet_connection_available", lambda: True
        )
        monkeypatch.setattr(
            "euroeval.safetensors_utils.parse_safetensors_file_metadata", raise_error
        )
        with caplog.at_level("WARNING", logger="euroeval"):
            result = LayaAdapter.num_params(
                model_id="convaiinnovations/laya-does-not-exist", param=None
            )
        assert result == -1
        assert any(record.levelname == "WARNING" for record in caplog.records), (
            "Expected the failure to be logged at WARNING level."
        )


class TestVariants:
    """Tests for the `#param` -> subfolder mapping."""

    def test_invalid_variant_is_rejected_by_registry(
        self, fake_laya_module: types.ModuleType, benchmark_config: BenchmarkConfig
    ) -> None:
        """An unknown `#param` variant raises `InvalidModel` via the model config."""
        with pytest.raises(InvalidModel, match="Invalid parameter"):
            ZeroShotClassifierModel.get_model_config(
                model_id="convaiinnovations/laya#not-a-real-variant",
                benchmark_config=benchmark_config,
            )

    @pytest.mark.parametrize(
        ("param", "raises"),
        [(None, False), ("multilingual", True)],
        ids=["no param: loads at its own root", "param: rejected"],
    )
    def test_standalone_repo_param_handling(
        self,
        fake_laya_module: types.ModuleType,
        model_config: ModelConfig,
        benchmark_config: BenchmarkConfig,
        param: str | None,
        raises: bool,
    ) -> None:
        """A standalone repo loads at its own root and rejects a `#param`.

        `variants` is a class-level attribute shared by every
        `convaiinnovations/laya*` repo ID (so the generic `#param` validation in
        `ZeroShotClassifierModel.get_model_config` can't tell them apart), so
        rejecting a `#param` on a standalone repo (which doesn't accept one) is
        enforced defensively in `LayaAdapter.__init__` instead.
        """
        config = dataclasses.replace(
            model_config,
            model_id="convaiinnovations/laya-multilingual",
            param=param,
            revision="main",
        )
        if raises:
            with pytest.raises(InvalidModel, match="does not accept a parameter"):
                LayaAdapter(model_config=config, benchmark_config=benchmark_config)
        else:
            adapter = LayaAdapter(
                model_config=config, benchmark_config=benchmark_config
            )
            assert isinstance(adapter.agent, FakeAgent)
            assert adapter.agent.subfolder is None
            assert adapter.max_length == 1024

    @pytest.mark.parametrize(
        ("param", "expected_subfolder", "expected_max_length"),
        [
            (None, None, 512),
            ("multilingual", "multilingual", 1024),
            ("typed-decisions", "typed-decisions", 512),
        ],
    )
    def test_variant_maps_to_subfolder(
        self,
        fake_laya_module: types.ModuleType,
        model_config: ModelConfig,
        benchmark_config: BenchmarkConfig,
        param: str | None,
        expected_subfolder: str | None,
        expected_max_length: int,
    ) -> None:
        """Each allowed parameter loads the expected subfolder and max length."""
        config = dataclasses.replace(
            model_config,
            model_id="convaiinnovations/laya",
            param=param,
            revision="main",
        )
        adapter = LayaAdapter(model_config=config, benchmark_config=benchmark_config)
        assert isinstance(adapter.agent, FakeAgent)
        assert adapter.agent.subfolder == expected_subfolder
        assert adapter.max_length == expected_max_length

    def test_unknown_standalone_repo_uses_max_len_from_agent_cfg(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_laya_module: types.ModuleType,
        model_config: ModelConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """An unknown standalone repo reads its context length from `agent.cfg`."""
        monkeypatch.setattr(FakeAgent, "default_cfg", {"max_len": 2048})
        config = dataclasses.replace(
            model_config,
            model_id="convaiinnovations/laya-some-future-variant",
            param=None,
            revision="main",
        )
        adapter = LayaAdapter(model_config=config, benchmark_config=benchmark_config)
        assert adapter.max_length == 2048

    @pytest.mark.parametrize(
        "cfg", [{}, {"max_len": 0}, {"max_len": -1}, {"max_len": "not-an-int"}]
    )
    def test_unknown_standalone_repo_falls_back_to_default_max_length(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_laya_module: types.ModuleType,
        model_config: ModelConfig,
        benchmark_config: BenchmarkConfig,
        cfg: dict,
    ) -> None:
        """An unknown standalone repo falls back to 512 without a usable `max_len`."""
        monkeypatch.setattr(FakeAgent, "default_cfg", cfg)
        config = dataclasses.replace(
            model_config,
            model_id="convaiinnovations/laya-some-future-variant",
            param=None,
            revision="main",
        )
        adapter = LayaAdapter(model_config=config, benchmark_config=benchmark_config)
        assert adapter.max_length == 512


@pytest.fixture
def fake_laya_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake `laya` module in `sys.modules`.

    Returns:
        The fake `laya` module.
    """
    fake_module = types.ModuleType("laya")
    fake_module.Agent = FakeAgent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "laya", fake_module)
    return fake_module
