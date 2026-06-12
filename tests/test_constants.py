"""Tests for the `constants` module."""

import typing as t

from euroeval import constants
from euroeval.constants import ORTHOGONAL_TASKS
from euroeval.data_models import Task


def test_all_objects_in_constants_are_constants() -> None:
    """Test that all objects in the `constants` module are constants."""
    for name in dir(constants):
        if name.startswith("__") or name in {"t", "TaskGroup", "re"}:
            continue
        assert name.isupper() and isinstance(
            getattr(constants, name), (Task, t.TypeVar, int, float, str, list, dict)
        )


def test_orthogonal_tasks() -> None:
    """Test that ORTHOGONAL_TASKS contains the expected tasks."""
    assert isinstance(ORTHOGONAL_TASKS, frozenset)
    assert "european-values" in ORTHOGONAL_TASKS
    assert len(ORTHOGONAL_TASKS) > 0
