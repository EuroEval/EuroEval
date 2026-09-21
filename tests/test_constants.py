"""Tests for the `constants` module."""

from euroeval.constants import ORTHOGONAL_TASKS


def test_orthogonal_tasks() -> None:
    """Test that ORTHOGONAL_TASKS contains the expected tasks."""
    assert isinstance(ORTHOGONAL_TASKS, frozenset)
    assert "european-values" in ORTHOGONAL_TASKS
    assert len(ORTHOGONAL_TASKS) > 0
