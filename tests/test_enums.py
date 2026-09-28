"""Tests for the `enums` module."""

import pytest

from euroeval.enums import ModelType, TaskGroup


class TestModelTypeCapabilities:
    """Test the `ModelType` capability properties and methods."""

    @pytest.mark.parametrize(
        "model_type,expected",
        [
            (ModelType.ENCODER, True),
            (ModelType.GENERATIVE, False),
            (ModelType.ZERO_SHOT_CLASSIFIER, True),
        ],
    )
    def test_requires_canary_reference(
        self, model_type: ModelType, expected: bool
    ) -> None:
        """Test that `requires_canary_reference` is correct for each model type."""
        assert model_type.requires_canary_reference is expected

    @pytest.mark.parametrize(
        "model_type,task_group,expected",
        [
            (ModelType.ENCODER, TaskGroup.SEQUENCE_CLASSIFICATION, True),
            (ModelType.ENCODER, TaskGroup.TEXT_TO_TEXT, True),
            (ModelType.GENERATIVE, TaskGroup.TEXT_TO_TEXT, True),
            (ModelType.ZERO_SHOT_CLASSIFIER, TaskGroup.SEQUENCE_CLASSIFICATION, True),
            (
                ModelType.ZERO_SHOT_CLASSIFIER,
                TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
                True,
            ),
            (ModelType.ZERO_SHOT_CLASSIFIER, TaskGroup.TEXT_TO_TEXT, False),
            (ModelType.ZERO_SHOT_CLASSIFIER, TaskGroup.TOKEN_CLASSIFICATION, False),
            (ModelType.ZERO_SHOT_CLASSIFIER, TaskGroup.QUESTION_ANSWERING, False),
        ],
    )
    def test_supports_task_group(
        self, model_type: ModelType, task_group: TaskGroup, expected: bool
    ) -> None:
        """Test that `supports_task_group` is correct for each model type."""
        assert model_type.supports_task_group(task_group) is expected

    @pytest.mark.parametrize(
        "model_type,expected",
        [
            (ModelType.ENCODER, False),
            (ModelType.GENERATIVE, True),
            (ModelType.ZERO_SHOT_CLASSIFIER, True),
        ],
    )
    def test_uses_generation_pipeline(
        self, model_type: ModelType, expected: bool
    ) -> None:
        """Test that `uses_generation_pipeline` is correct for each model type."""
        assert model_type.uses_generation_pipeline is expected
