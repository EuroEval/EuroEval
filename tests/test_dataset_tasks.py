"""Unit tests for the `tasks` module."""

from typing import Generator

import pytest
from scandeval.config import Task
from scandeval.tasks import get_all_tasks


class TestGetAllTasks:
    """Unit tests for the `get_all_tasks` function."""

    @pytest.fixture(scope="class")
    def tasks(self) -> Generator[dict[str, Task], None, None]:
        """Yields all dataset tasks."""
        yield get_all_tasks()

    def test_tasks_is_dict(self, tasks):
        """Tests that the dataset tasks are a dictionary."""
        assert isinstance(tasks, dict)

    def test_tasks_are_objects(self, tasks):
        """Tests that the dataset tasks are objects."""
        for task in tasks.values():
            assert isinstance(task, Task)

    @pytest.mark.parametrize(
        "task_name",
        [
            "linguistic-acceptability",
            "named-entity-recognition",
            "question-answering",
            "sentiment-classification",
            "summarization",
            "knowledge",
            "common-sense-reasoning",
            "text-modelling",
            "speed",
        ],
    )
    def test_get_task(self, tasks, task_name):
        """Tests that the dataset task can be retrieved by name."""
        assert task_name in tasks
