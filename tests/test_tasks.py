"""Tests for the `tasks` module."""

import pytest

from euroeval import tasks
from euroeval.data_models import Task
from euroeval.enums import GenerativeType, ModelType, TaskGroup


def test_all_tasks_are_task_objects() -> None:
    """Test that all task constants are Task objects."""
    task_names = [
        "LA",
        "NER",
        "RC",
        "SENT",
        "SIMPL",
        "SUMM",
        "KNOW",
        "MCRC",
        "COMMON_SENSE",
        "EUROPEAN_VALUES",
        "MCSTEREO",
        "SPEED",
        "TEXT_CLASSIFICATION",
        "TOKEN_CLASSIFICATION",
        "MULTIPLE_CHOICE",
        "INSTRUCTION_FOLLOWING",
    ]
    for task_name in task_names:
        task = getattr(tasks, task_name)
        assert isinstance(task, Task), f"{task_name} should be a Task instance"


@pytest.mark.parametrize(
    "task_attr,expected_task_group",
    [
        ("LA", TaskGroup.SEQUENCE_CLASSIFICATION),
        ("NER", TaskGroup.TOKEN_CLASSIFICATION),
        ("RC", TaskGroup.QUESTION_ANSWERING),
        ("SENT", TaskGroup.SEQUENCE_CLASSIFICATION),
        ("SIMPL", TaskGroup.TEXT_TO_TEXT),
        ("SUMM", TaskGroup.TEXT_TO_TEXT),
        ("KNOW", TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION),
        ("SPEED", TaskGroup.SPEED),
    ],
    ids=["la", "ner", "rc", "sent", "simpl", "summ", "know", "speed"],
)
def test_task_groups(task_attr: str, expected_task_group: TaskGroup) -> None:
    """Test that tasks have the correct task group."""
    task = getattr(tasks, task_attr)
    assert task.task_group == expected_task_group


def test_linguistic_acceptability_task() -> None:
    """Test the linguistic acceptability task configuration."""
    assert tasks.LA.name == "linguistic-acceptability"
    assert tasks.LA.task_group == TaskGroup.SEQUENCE_CLASSIFICATION
    assert tasks.LA.default_num_few_shot_examples == 12
    assert tasks.LA.default_labels == ["correct", "incorrect"]
    assert tasks.LA.uses_logprobs is True
    assert len(tasks.LA.metrics) == 2


def test_named_entity_recognition_task() -> None:
    """Test the named entity recognition task configuration."""
    assert tasks.NER.name == "named-entity-recognition"
    assert tasks.NER.task_group == TaskGroup.TOKEN_CLASSIFICATION
    assert tasks.NER.default_num_few_shot_examples == 8
    assert tasks.NER.default_max_generated_tokens == 128
    assert tasks.NER.uses_structured_output is True
    assert len(tasks.NER.default_labels) == 9  # 9 NER labels


def test_sentiment_classification_task() -> None:
    """Test the sentiment classification task configuration."""
    assert tasks.SENT.name == "sentiment-classification"
    assert tasks.SENT.default_labels == ["positive", "neutral", "negative"]
    assert tasks.SENT.uses_logprobs is True


def test_text_to_text_tasks_require_generative() -> None:
    """Test that text-to-text tasks require generative models."""
    assert ModelType.GENERATIVE in tasks.SIMPL.default_allowed_model_types
    assert ModelType.GENERATIVE in tasks.SUMM.default_allowed_model_types


def test_european_values_task_constraints() -> None:
    """Test the European values task special constraints."""
    assert tasks.EUROPEAN_VALUES.requires_zero_shot is True
    assert tasks.EUROPEAN_VALUES.default_num_few_shot_examples == 0
    assert GenerativeType.INSTRUCTION_TUNED in tasks.EUROPEAN_VALUES.default_allowed_generative_types
    assert tasks.EUROPEAN_VALUES.default_allow_invalid_model_outputs is False


def test_speed_task() -> None:
    """Test the speed benchmark task configuration."""
    assert tasks.SPEED.name == "speed"
    assert tasks.SPEED.task_group == TaskGroup.SPEED
    assert tasks.SPEED.default_num_few_shot_examples == 0
    assert tasks.SPEED.default_max_generated_tokens == 5
    assert len(tasks.SPEED.template_dict) == 0  # Empty template dict


def test_instruction_following_task() -> None:
    """Test the instruction following task configuration."""
    assert tasks.INSTRUCTION_FOLLOWING.name == "instruction-following"
    assert tasks.INSTRUCTION_FOLLOWING.requires_zero_shot is True
    assert tasks.INSTRUCTION_FOLLOWING.uses_logprobs is False
    assert tasks.INSTRUCTION_FOLLOWING.default_max_generated_tokens == 2048
