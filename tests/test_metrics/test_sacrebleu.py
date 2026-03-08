"""Tests for the `sacrebleu` metrics module."""

import collections.abc as c
import typing as t

import pytest
from datasets import Dataset

from euroeval.enums import Device, GenerativeType
from euroeval.languages import ENGLISH, Language
from euroeval.metrics.sacrebleu import (
    ChrF,
    chrf2_metric,
    chrf2pp_metric,
    chrf3_metric,
    chrf3pp_metric,
    chrf4_metric,
    chrf4pp_metric,
)
from euroeval.tasks import SUMM


@pytest.fixture
def make_dataset() -> c.Generator[
    c.Callable[[list[str], list[str]], Dataset], None, None
]:
    """Create a dataset from predictions and references.

    Yields:
        A function that takes predictions and references and returns a Dataset.
    """

    def _make(predictions: list[str], references: list[str]) -> Dataset:
        return Dataset.from_list(
            [{"prediction": p, "reference": r} for p, r in zip(predictions, references)]
        )

    yield _make


class DummyDatasetConfig:
    """Dummy dataset config for testing."""

    task = SUMM
    languages: list[Language] = [ENGLISH]


@pytest.fixture
def dummy_dataset_config() -> t.Generator[DummyDatasetConfig, None, None]:
    """Yield a dummy dataset config (not used by ChrF metric)."""
    # The ChrF metric doesn't actually use dataset_config or benchmark_config
    # We return None-like objects that won't cause issues
    yield DummyDatasetConfig()


class DummyBenchmarkConfig:
    """Dummy benchmark config for testing."""

    device = Device.CPU
    generative_type = GenerativeType.INSTRUCTION_TUNED


@pytest.fixture
def dummy_benchmark_config() -> t.Generator[DummyBenchmarkConfig, None, None]:
    """Yield a dummy benchmark config (not used by ChrF metric)."""
    yield DummyBenchmarkConfig()


@pytest.mark.parametrize(
    ("metric", "expected_name", "expected_beta", "expected_word_order"),
    [
        (chrf2_metric, "chr_f2", 2, 0),
        (chrf3_metric, "chr_f3", 3, 0),
        (chrf4_metric, "chr_f4", 4, 0),
        (chrf2pp_metric, "chr_f2pp", 2, 2),
        (chrf3pp_metric, "chr_f3pp", 3, 2),
        (chrf4pp_metric, "chr_f4pp", 4, 2),
    ],
)
def test_metric_initialization(
    metric: ChrF, expected_name: str, expected_beta: int, expected_word_order: int
) -> None:
    """Test that metrics are initialized with correct parameters."""
    assert metric.name == expected_name
    assert metric.beta == expected_beta
    assert metric.word_order == expected_word_order
    assert metric.pretty_name.startswith("ChrF")
    assert metric.postprocessing_fn is not None


def test_postprocessing_fn() -> None:
    """Test that the postprocessing function works correctly."""
    score: float = 75.0  # CHRF returns scores as percentages (0-100)

    (processed_score, score_str) = chrf3pp_metric.postprocessing_fn(score)  # type: ignore[misc]

    # The postprocessing function returns the score as-is (already a percentage)
    # and formats it as a string with "%"
    assert processed_score == 75.0
    assert score_str == "75.00%"


@pytest.mark.parametrize(
    ("predictions", "references", "expected_score"),
    [
        (["hello world", "test sentence"], ["hello world", "test sentence"], 100.0),
        (["abc", "xyz"], ["def", "uvw"], pytest.approx(0.0, abs=10.0)),
        (
            ["the cat sat on the mat", "the cat sat"],
            ["the cat sat on the mat", "the cat sat"],
            100.0,
        ),
    ],
)
def test_chrffully_matches_predictions(
    predictions: list[str],
    references: list[str],
    expected_score: float,
    make_dataset: c.Callable[[list[str], list[str]], Dataset],
    dummy_dataset_config: DummyDatasetConfig,
    dummy_benchmark_config: DummyBenchmarkConfig,
) -> None:
    """Test ChrF metric with fully matching predictions."""
    dataset: Dataset = make_dataset(predictions, references)

    score = chrf3pp_metric(
        predictions=predictions,
        references=references,
        dataset=dataset,
        dataset_config=dummy_dataset_config,  # type: ignore[arg-type]
        benchmark_config=dummy_benchmark_config,  # type: ignore[arg-type]
    )

    assert score is not None
    # CHRF returns scores as percentages (0-100)
    assert pytest.approx(score, abs=0.01) == expected_score


@pytest.mark.parametrize(
    ("word_order", "beta", "predictions", "references"),
    [
        (0, 2, ["hello world"], ["hello world"]),
        (0, 3, ["hello world"], ["hello world"]),
        (0, 4, ["hello world"], ["hello world"]),
        (2, 2, ["hello world"], ["hello world"]),
        (2, 4, ["hello world"], ["hello world"]),
    ],
)
def test_all_chrff_variants(
    word_order: int,
    beta: int,
    predictions: list[str],
    references: list[str],
    make_dataset: c.Callable[[list[str], list[str]], Dataset],
    dummy_dataset_config: DummyDatasetConfig,
    dummy_benchmark_config: DummyBenchmarkConfig,
) -> None:
    """Test all ChrF variants with matching predictions."""
    dataset: Dataset = make_dataset(predictions, references)

    metric: ChrF = ChrF(word_order=word_order, beta=beta)
    metric = metric.download(".cache")  # type: ignore[assignment]

    score = metric(
        predictions=predictions,
        references=references,
        dataset=dataset,
        dataset_config=dummy_dataset_config,  # type: ignore[arg-type]
        benchmark_config=dummy_benchmark_config,  # type: ignore[arg-type]
    )

    assert score is not None
    # CHRF returns scores as percentages (0-100)
    assert score == pytest.approx(100.0, abs=0.01)


def test_chrff_empty_predictions(
    make_dataset: c.Callable[[list[str], list[str]], Dataset],
    dummy_dataset_config: DummyDatasetConfig,
    dummy_benchmark_config: DummyBenchmarkConfig,
) -> None:
    """Test ChrF metric with empty predictions returns 1.0."""
    dataset: Dataset = make_dataset([], [])

    score = chrf3pp_metric(
        predictions=[],
        references=[],
        dataset=dataset,
        dataset_config=dummy_dataset_config,  # type: ignore[arg-type]
        benchmark_config=dummy_benchmark_config,  # type: ignore[arg-type]
    )

    # When there are no predictions, the metric returns 1.0 (perfect score)
    assert score == 1.0


def test_chrff_different_word_orders(
    make_dataset: c.Callable[[list[str], list[str]], Dataset],
    dummy_dataset_config: DummyDatasetConfig,
    dummy_benchmark_config: DummyBenchmarkConfig,
) -> None:
    """Test that different word orders produce different scores."""
    predictions: list[str] = ["the quick brown fox jumps over the lazy dog"]
    references: list[str] = ["a quick brown dog jumps over the lazy fox"]

    dataset: Dataset = make_dataset(predictions, references)

    # Word order 0 (no bi-grams)
    metric_0: ChrF = ChrF(word_order=0, beta=2)
    score_0: float | None = metric_0(
        predictions=predictions,
        references=references,
        dataset=dataset,
        dataset_config=dummy_dataset_config,  # type: ignore[arg-type]
        benchmark_config=dummy_benchmark_config,  # type: ignore[arg-type]
    )

    # Word order 2 (with bi-grams)
    metric_2: ChrF = ChrF(word_order=2, beta=2)
    score_2: float | None = metric_2(
        predictions=predictions,
        references=references,
        dataset=dataset,
        dataset_config=dummy_dataset_config,  # type: ignore[arg-type]
        benchmark_config=dummy_benchmark_config,  # type: ignore[arg-type]
    )

    # Scores should be different due to different word order handling
    assert score_0 is not None
    assert score_2 is not None
    assert score_0 != score_2


def test_chrff_partial_match(
    make_dataset: c.Callable[[list[str], list[str]], Dataset],
    dummy_dataset_config: DummyDatasetConfig,
    dummy_benchmark_config: DummyBenchmarkConfig,
) -> None:
    """Test ChrF metric with partial matches."""
    predictions: list[str] = ["hello world", "test case one", "another test"]
    references: list[str] = ["hello there", "test case two", "different text"]

    dataset: Dataset = make_dataset(predictions, references)

    score = chrf3pp_metric(
        predictions=predictions,
        references=references,
        dataset=dataset,
        dataset_config=dummy_dataset_config,  # type: ignore[arg-type]
        benchmark_config=dummy_benchmark_config,  # type: ignore[arg-type]
    )

    assert score is not None
    # CHRF returns scores as percentages (0-100)
    # Should be between 0 and 100 since there are partial matches
    assert 0 < score < 100


def test_chrff_metric_download() -> None:
    """Test that the download method returns the same metric instance."""
    metric = chrf3pp_metric
    downloaded_metric: ChrF = metric.download("/tmp/test_cache")  # type: ignore[assignment]

    assert downloaded_metric is metric
