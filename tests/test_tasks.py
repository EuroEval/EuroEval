"""Tests for the `tasks` module."""

from euroeval import tasks
from euroeval.data_models import Task
from euroeval.enums import ModelType


class TestDefaultAllowedModelTypes:
    """Test the `default_allowed_model_types` of the `Task` data model."""

    def test_all_tasks_use_default_or_explicit_restriction(self) -> None:
        """Test every task's allowed model types match its intended defaults."""
        for task in tasks.get_all_tasks().values():
            assert set(task.default_allowed_model_types).issubset(set(ModelType))

    def test_generative_only_tasks_keep_explicit_restriction(self) -> None:
        """Test that genuinely generative-only tasks still restrict model types."""
        assert tasks.GEC.default_allowed_model_types == [ModelType.GENERATIVE]
        assert tasks.SUMM.default_allowed_model_types == [ModelType.GENERATIVE]

    def test_task_default_allows_all_model_types(self) -> None:
        """Test that a bare `Task` defaults to allowing all model types."""
        task = Task(
            name="dummy",
            task_group=tasks.LA.task_group,
            template_dict={},
            metrics=[],
            default_num_few_shot_examples=0,
            default_max_generated_tokens=1,
        )
        assert set(task.default_allowed_model_types) == set(ModelType)
