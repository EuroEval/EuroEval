"""Tests for the `preprocessing` module."""

import pytest
from datasets import Dataset, DatasetDict

from euroeval.enums import TaskGroup
from euroeval.exceptions import InvalidBenchmark
from euroeval.preprocessing import build_preprocessing_func, merge_input_and_choices


class TestMergeInputAndChoices:
    """Tests for the merge_input_and_choices function."""

    def test_merge_input_and_choices_with_string_choices_column(self) -> None:
        """Test merging input with a single choices column (string)."""
        example = {
            "input": "What is the capital of Denmark?",
            "choices": ["Copenhagen", "Aarhus", "Odense"],
        }
        result = merge_input_and_choices(
            example=example,
            input_column="input",
            choices_column="choices",
            choices_label="Choices",
        )
        assert "text" in result
        assert "What is the capital of Denmark?" in result["text"]
        assert "Choices:" in result["text"]
        assert "a. Copenhagen" in result["text"]
        assert "b. Aarhus" in result["text"]
        assert "c. Odense" in result["text"]

    def test_merge_input_and_choices_with_list_of_choices_columns(self) -> None:
        """Test merging input with a list of choices columns."""
        example = {
            "input": "What is the capital of Denmark?",
            "choice_a": "Copenhagen",
            "choice_b": "Aarhus",
            "choice_c": "Odense",
        }
        result = merge_input_and_choices(
            example=example,
            input_column="input",
            choices_column=["choice_a", "choice_b", "choice_c"],
            choices_label="Choices",
        )
        assert "text" in result
        assert "What is the capital of Denmark?" in result["text"]
        assert "Choices:" in result["text"]
        assert "a. Copenhagen" in result["text"]
        assert "b. Aarhus" in result["text"]
        assert "c. Odense" in result["text"]


class TestBuildPreprocessingFunc:
    """Tests for the build_preprocessing_func function."""

    def test_build_preprocessing_func_with_choices_merging(self) -> None:
        """Test full preprocessing pipeline with choices merging."""
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "input": ["What is 2+2?", "What is 3+3?"],
                        "choices": [["4", "5", "6"], ["6", "7", "8"]],
                        "label": [0, 0],
                    }
                )
            }
        )
        preprocessing = build_preprocessing_func(
            dataset_name="test_dataset",
            task_group=TaskGroup.SEQUENCE_CLASSIFICATION,
            input_column="input",
            target_column="label",
            choices_column="choices",
            choices_label="Choices",
        )
        result = preprocessing(dataset)
        assert "text" in result["train"].column_names
        assert "label" in result["train"].column_names

    def test_build_preprocessing_func_renames_input_column(self) -> None:
        """Test that non-'text' input columns are renamed."""
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {"question": ["What is 2+2?", "What is 3+3?"], "label": [0, 1]}
                )
            }
        )
        preprocessing = build_preprocessing_func(
            dataset_name="test_dataset",
            task_group=TaskGroup.SEQUENCE_CLASSIFICATION,
            input_column="question",
            target_column="label",
            choices_column=None,
            choices_label="Choices",
        )
        result = preprocessing(dataset)
        assert "text" in result["train"].column_names
        assert "question" not in result["train"].column_names
        assert result["train"]["text"] == ["What is 2+2?", "What is 3+3?"]

    def test_build_preprocessing_func_renames_target_column_to_labels(self) -> None:
        """Test token classification target column becomes 'labels'."""
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {"text": ["Hello world", "Test sentence"], "ner_labels": [0, 1]}
                )
            }
        )
        preprocessing = build_preprocessing_func(
            dataset_name="test_dataset",
            task_group=TaskGroup.TOKEN_CLASSIFICATION,
            input_column="text",
            target_column="ner_labels",
            choices_column=None,
            choices_label="Choices",
        )
        result = preprocessing(dataset)
        assert "labels" in result["train"].column_names
        assert "ner_labels" not in result["train"].column_names

    def test_build_preprocessing_func_renames_target_column_to_target_text(
        self,
    ) -> None:
        """Test text-to-text target column becomes 'target_text'."""
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "text": ["Hello world", "Test sentence"],
                        "translation": ["Hallo verden", "Test sætning"],
                    }
                )
            }
        )
        preprocessing = build_preprocessing_func(
            dataset_name="test_dataset",
            task_group=TaskGroup.TEXT_TO_TEXT,
            input_column="text",
            target_column="translation",
            choices_column=None,
            choices_label="Choices",
        )
        result = preprocessing(dataset)
        assert "target_text" in result["train"].column_names
        assert "translation" not in result["train"].column_names

    def test_build_preprocessing_func_renames_target_column_to_label(self) -> None:
        """Test classification target column becomes 'label'."""
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {"text": ["Hello world", "Test sentence"], "category": [0, 1]}
                )
            }
        )
        preprocessing = build_preprocessing_func(
            dataset_name="test_dataset",
            task_group=TaskGroup.SEQUENCE_CLASSIFICATION,
            input_column="text",
            target_column="category",
            choices_column=None,
            choices_label="Choices",
        )
        result = preprocessing(dataset)
        assert "label" in result["train"].column_names
        assert "category" not in result["train"].column_names

    def test_build_preprocessing_func_validates_input_column_exists(self) -> None:
        """Test that InvalidBenchmark is raised if input_column doesn't exist."""
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {"text": ["Hello world", "Test sentence"], "label": [0, 1]}
                )
            }
        )
        preprocessing = build_preprocessing_func(
            dataset_name="test_dataset",
            task_group=TaskGroup.SEQUENCE_CLASSIFICATION,
            input_column="nonexistent_column",
            target_column="label",
            choices_column=None,
            choices_label="Choices",
        )
        with pytest.raises(InvalidBenchmark):
            preprocessing(dataset)

    def test_build_preprocessing_func_validates_choices_columns_exist(self) -> None:
        """Test that InvalidBenchmark is raised if choices columns don't exist."""
        dataset = DatasetDict(
            {"train": Dataset.from_dict({"input": ["What is 2+2?"], "label": [0]})}
        )
        preprocessing = build_preprocessing_func(
            dataset_name="test_dataset",
            task_group=TaskGroup.SEQUENCE_CLASSIFICATION,
            input_column="input",
            target_column="label",
            choices_column="nonexistent_column",
            choices_label="Choices",
        )
        with pytest.raises(InvalidBenchmark):
            preprocessing(dataset)

    def test_build_preprocessing_func_validates_target_column_exists(self) -> None:
        """Test that InvalidBenchmark is raised if target_column doesn't exist."""
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {"text": ["Hello world", "Test sentence"], "label": [0, 1]}
                )
            }
        )
        preprocessing = build_preprocessing_func(
            dataset_name="test_dataset",
            task_group=TaskGroup.SEQUENCE_CLASSIFICATION,
            input_column="text",
            target_column="nonexistent_column",
            choices_column=None,
            choices_label="Choices",
        )
        with pytest.raises(InvalidBenchmark):
            preprocessing(dataset)

    def test_build_preprocessing_func_with_multiple_choices_columns(self) -> None:
        """Test merging when choices_column is a list of column names."""
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "input": ["What is 2+2?"],
                        "choice_a": ["4"],
                        "choice_b": ["5"],
                        "label": [0],
                    }
                )
            }
        )
        preprocessing = build_preprocessing_func(
            dataset_name="test_dataset",
            task_group=TaskGroup.SEQUENCE_CLASSIFICATION,
            input_column="input",
            target_column="label",
            choices_column=["choice_a", "choice_b"],
            choices_label="Choices",
        )
        result = preprocessing(dataset)
        assert "text" in result["train"].column_names
        assert "choice_a" not in result["train"].column_names
        assert "choice_b" not in result["train"].column_names
        assert "a. 4" in result["train"]["text"][0]
        assert "b. 5" in result["train"]["text"][0]
