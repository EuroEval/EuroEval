"""Tests for the `metrics.speed` module."""

import pytest

from euroeval.metrics.speed import SpeedMetric, speed_metric, speed_short_metric


def test_speed_metric_initialization() -> None:
    """Test SpeedMetric initialization."""
    metric = SpeedMetric(name="test_speed", pretty_name="Test Speed")
    assert metric.name == "test_speed"
    assert metric.pretty_name == "Test Speed"


def test_speed_metric_postprocessing() -> None:
    """Test that the speed metric formats numbers correctly."""
    metric = SpeedMetric(name="speed", pretty_name="Speed")
    score, formatted = metric.postprocessing_fn(1234.5)
    assert score == 1234.5
    assert formatted == "1,234"  # Rounds to nearest integer


def test_speed_metric_call_not_implemented() -> None:
    """Test that calling the speed metric raises NotImplementedError."""
    metric = SpeedMetric(name="speed", pretty_name="Speed")
    with pytest.raises(NotImplementedError):
        metric(
            predictions=[],
            references=[],
            dataset=None,  # type: ignore
            dataset_config=None,  # type: ignore
            benchmark_config=None,  # type: ignore
        )


def test_speed_metric_global_instance() -> None:
    """Test that global speed_metric instance is correctly configured."""
    assert isinstance(speed_metric, SpeedMetric)
    assert speed_metric.name == "speed"
    assert speed_metric.pretty_name == "Tokens per second"


def test_speed_short_metric_global_instance() -> None:
    """Test that global speed_short_metric instance is correctly configured."""
    assert isinstance(speed_short_metric, SpeedMetric)
    assert speed_short_metric.name == "speed_short"
    assert speed_short_metric.pretty_name == "Tokens per second on short documents"


def test_speed_metric_hash() -> None:
    """Test that speed metric can be hashed by name."""
    metric1 = SpeedMetric(name="speed", pretty_name="Speed")
    metric2 = SpeedMetric(name="speed", pretty_name="Different Pretty Name")
    assert hash(metric1) == hash(metric2)  # Same name should have same hash


def test_speed_metric_download() -> None:
    """Test that download method returns self."""
    metric = SpeedMetric(name="speed", pretty_name="Speed")
    result = metric.download(cache_dir="/tmp/cache")
    assert result is metric
