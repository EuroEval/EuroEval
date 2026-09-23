"""Unit tests for `euroeval.zero_shot_adapters.laya`."""

import contextlib
import dataclasses
import importlib.util
import sys
import types
import typing as t
import warnings
from dataclasses import dataclass

import pytest

from euroeval.benchmark_modules.zero_shot_classifier import ZeroShotClassifierModel
from euroeval.data_models import BenchmarkConfig, ModelConfig
from euroeval.enums import InferenceBackend, ModelType
from euroeval.exceptions import InvalidModel, NeedsExtraInstalled
from euroeval.zero_shot_adapters.laya import LayaAdapter


@dataclass
class FakeAgent:
    """A fake `laya.Agent`, recording how it was constructed and called."""

    model_id_or_path: str
    subfolder: str | None
    token: str | None
    calls: list[tuple[object, dict]]

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


class TestMatches:
    """Tests for `LayaAdapter.matches`."""

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


class TestVariants:
    """Tests for the `#param` -> subfolder mapping."""

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
            model_config, model_id="convaiinnovations/laya", param=param
        )
        adapter = LayaAdapter(model_config=config, benchmark_config=benchmark_config)
        assert isinstance(adapter.agent, FakeAgent)
        assert adapter.agent.subfolder == expected_subfolder
        assert adapter.max_length == expected_max_length

    def test_invalid_variant_is_rejected_by_registry(
        self, fake_laya_module: types.ModuleType, benchmark_config: BenchmarkConfig
    ) -> None:
        """An unknown `#param` variant raises `InvalidModel` via the model config."""
        with pytest.raises(InvalidModel, match="Invalid parameter"):
            ZeroShotClassifierModel.get_model_config(
                model_id="convaiinnovations/laya#not-a-real-variant",
                benchmark_config=benchmark_config,
            )


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
            model_config, model_id="convaiinnovations/laya", param=None
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


class TestNumParams:
    """Tests for `LayaAdapter.num_params`."""

    def test_returns_minus_one_on_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`num_params` returns -1 when the checkpoint can't be inspected."""

        def raise_error(*args: object, **kwargs: object) -> t.NoReturn:
            raise OSError("no network in this test")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", raise_error)
        assert (
            LayaAdapter.num_params(
                model_id="convaiinnovations/laya-does-not-exist", param=None
            )
            == -1
        )

    def test_counts_parameters_from_safetensors_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given fake safetensors header shapes, the parameter count is summed."""

        class FakeSlice:
            def __init__(self, shape: list[int]) -> None:
                self._shape = shape

            def get_shape(self) -> list[int]:
                return self._shape

        class FakeSafeOpen:
            def __init__(self, path: str, framework: str) -> None:
                self._shapes = {"a": [2, 3], "b": [4]}

            def keys(self) -> list[str]:
                return list(self._shapes.keys())

            def get_slice(self, key: str) -> FakeSlice:
                return FakeSlice(self._shapes[key])

        @contextlib.contextmanager
        def fake_safe_open(path: str, framework: str) -> t.Iterator[FakeSafeOpen]:
            yield FakeSafeOpen(path, framework)

        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download", lambda **kwargs: "/fake/path"
        )
        monkeypatch.setattr("safetensors.safe_open", fake_safe_open)
        num_params = LayaAdapter.num_params(
            model_id="convaiinnovations/laya", param=None
        )
        assert num_params == 2 * 3 + 4


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
