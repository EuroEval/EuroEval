"""Tests for the `llm_as_a_judge` module."""

import logging
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field, create_model

from euroeval.data_models import BenchmarkConfig, DatasetConfig
from euroeval.exceptions import InvalidBenchmark
from euroeval.metrics.llm_as_a_judge import (
    LLMAsAJudgeMetric,
    create_model_graded_fact_metric,
)


@pytest.fixture
def mock_dataset() -> MagicMock:
    """Create a mock dataset.

    Returns:
        MagicMock: A mock dataset object.
    """
    return MagicMock()


@pytest.fixture
def mock_dataset_config() -> MagicMock:
    """Create a mock dataset config.

    Returns:
        MagicMock: A mock dataset config object.
    """
    config = MagicMock(spec=DatasetConfig)
    config.name = "test_dataset"
    config.max_generated_tokens = 100
    return config


@pytest.fixture
def mock_benchmark_config() -> MagicMock:
    """Create a mock benchmark config.

    Returns:
        MagicMock: A mock benchmark config object.
    """
    config = MagicMock(spec=BenchmarkConfig)
    config.api_base = None
    config.api_key = None
    config.api_version = None
    config.progress_bar = False
    config.debug = False
    return config


@pytest.fixture
def simple_response_format() -> type[BaseModel]:
    """Create a simple response format Pydantic model.

    Returns:
        type[BaseModel]: A Pydantic model class with a value field.
    """
    return create_model("SimpleOutput", value=(int, Field(ge=0, le=10)))


@pytest.fixture
def simple_metric(simple_response_format: type[BaseModel]) -> LLMAsAJudgeMetric:
    """Create a simple LLMAsAJudgeMetric instance.

    Args:
        simple_response_format: The simple response format Pydantic model.

    Returns:
        LLMAsAJudgeMetric: A configured LLMAsAJudgeMetric instance.
    """

    def scoring_fn(output: MagicMock) -> float:
        return output.value / 10.0

    return LLMAsAJudgeMetric(
        name="test_metric",
        pretty_name="Test Metric",
        judge_id="test-model",
        judge_kwargs=dict(temperature=0.5),
        user_prompt="Score: {prediction}",
        response_format=simple_response_format,
        scoring_fn=scoring_fn,
    )


class TestLLMAsAJudgeMetricInit:
    """Tests for LLMAsAJudgeMetric initialization."""

    def test_llm_as_a_judge_init_basic(
        self, simple_response_format: type[BaseModel]
    ) -> None:
        """Test basic initialization of LLMAsAJudgeMetric."""

        def scoring_fn(output: MagicMock) -> float:
            return output.value / 10.0

        metric = LLMAsAJudgeMetric(
            name="test",
            pretty_name="Test",
            judge_id="test-model",
            judge_kwargs=dict(temperature=0.5),
            user_prompt="Test: {prediction}",
            response_format=simple_response_format,
            scoring_fn=scoring_fn,
        )

        assert metric.name == "test"
        assert metric.pretty_name == "Test"
        assert metric.judge_id == "test-model"
        assert metric.judge_kwargs["temperature"] == 0.5
        assert metric.user_prompt == "Test: {prediction}"
        assert metric.response_format == simple_response_format
        assert "response_format" in metric.judge_kwargs

    def test_llm_as_a_judge_init_with_batch_scoring_fn(
        self, simple_response_format: type[BaseModel]
    ) -> None:
        """Test initialization with batch_scoring_fn instead of scoring_fn."""

        def batch_scoring_fn(
            outputs: list[simple_response_format], dataset: MagicMock | None = None
        ) -> float:
            return sum(o.value for o in outputs) / len(outputs)

        metric = LLMAsAJudgeMetric(
            name="test",
            pretty_name="Test",
            judge_id="test-model",
            judge_kwargs=dict(),
            user_prompt="Test: {prediction}",
            response_format=simple_response_format,
            batch_scoring_fn=batch_scoring_fn,
        )

        assert metric.batch_scoring_fn is batch_scoring_fn

    def test_llm_as_a_judge_init_with_system_prompt(
        self, simple_response_format: type[BaseModel]
    ) -> None:
        """Test initialization with custom system prompt."""

        def scoring_fn(output: MagicMock) -> float:
            return output.value / 10.0

        metric = LLMAsAJudgeMetric(
            name="test",
            pretty_name="Test",
            judge_id="test-model",
            judge_kwargs=dict(),
            user_prompt="Test: {prediction}",
            response_format=simple_response_format,
            scoring_fn=scoring_fn,
            system_prompt="You are a helpful assistant.",
        )

        assert metric.system_prompt == "You are a helpful assistant."

    def test_llm_as_a_judge_init_with_condition_formatting_fn(
        self, simple_response_format: type[BaseModel]
    ) -> None:
        """Test initialization with custom condition formatting function."""

        def scoring_fn(output: MagicMock) -> float:
            return output.value / 10.0

        def format_condition(condition: str) -> str:
            return condition.upper()

        metric = LLMAsAJudgeMetric(
            name="test",
            pretty_name="Test",
            judge_id="test-model",
            judge_kwargs=dict(),
            user_prompt="Test: {prediction} | Condition: {condition}",
            response_format=simple_response_format,
            scoring_fn=scoring_fn,
            condition_formatting_fn=format_condition,
        )

        assert metric.condition_formatting_fn is format_condition


class TestLLMAsAJudgeCall:
    """Tests for LLMAsAJudgeMetric.__call__."""

    def test_llm_as_a_judge_call_mismatched_lengths_raises_error(
        self,
        simple_metric: LLMAsAJudgeMetric,
        mock_dataset: MagicMock,
        mock_dataset_config: MagicMock,
        mock_benchmark_config: MagicMock,
    ) -> None:
        """Test InvalidBenchmark when preds and refs have different lengths."""
        predictions = ["pred1", "pred2"]
        references = ["ref1"]

        with pytest.raises(InvalidBenchmark) as exc_info:
            simple_metric(
                predictions=predictions,
                references=references,
                dataset=mock_dataset,
                dataset_config=mock_dataset_config,
                benchmark_config=mock_benchmark_config,
            )

        assert "does not match the number of references" in str(exc_info.value)

    def test_llm_as_a_judge_call_empty_inputs_returns_none(
        self,
        simple_metric: LLMAsAJudgeMetric,
        mock_dataset: MagicMock,
        mock_dataset_config: MagicMock,
        mock_benchmark_config: MagicMock,
    ) -> None:
        """Test that empty predictions/references return None."""
        assert (
            simple_metric(
                predictions=[],
                references=["ref1"],
                dataset=mock_dataset,
                dataset_config=mock_dataset_config,
                benchmark_config=mock_benchmark_config,
            )
            is None
        )
        assert (
            simple_metric(
                predictions=["pred1"],
                references=[],
                dataset=mock_dataset,
                dataset_config=mock_dataset_config,
                benchmark_config=mock_benchmark_config,
            )
            is None
        )
        assert (
            simple_metric(
                predictions=[],
                references=[],
                dataset=mock_dataset,
                dataset_config=mock_dataset_config,
                benchmark_config=mock_benchmark_config,
            )
            is None
        )

    @patch("euroeval.benchmark_modules.LiteLLMModel")
    @patch("euroeval.model_cache.ModelCache")
    @patch("euroeval.metrics.llm_as_a_judge.extract_json_dict_from_string")
    def test_llm_as_a_judge_call_parse_failure_logs_warning(
        self,
        mock_extract: MagicMock,
        mock_cache_class: MagicMock,
        mock_llm_model_class: MagicMock,
        simple_metric: LLMAsAJudgeMetric,
        mock_dataset: MagicMock,
        mock_dataset_config: MagicMock,
        mock_benchmark_config: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test debug log when judge output parsing fails."""
        # Return valid JSON dict but with invalid field type (causes ValidationError)
        mock_extract.return_value = {"value": "not_an_int"}

        # Setup mocks
        mock_cache_instance = MagicMock()
        mock_cache_instance.__contains__ = MagicMock(return_value=False)
        mock_cache_class.return_value = mock_cache_instance

        mock_output = MagicMock()
        mock_output.sequence = '{"value": "not_an_int"}'
        mock_cache_instance.__getitem__ = MagicMock(return_value=mock_output)

        predictions = ["pred1"]
        references = ["ref1"]

        with caplog.at_level(logging.DEBUG):
            simple_metric(
                predictions=predictions,
                references=references,
                dataset=mock_dataset,
                dataset_config=mock_dataset_config,
                benchmark_config=mock_benchmark_config,
            )

        assert "Could not parse/validate" in caplog.text

    @patch("euroeval.benchmark_modules.LiteLLMModel")
    @patch("euroeval.model_cache.ModelCache")
    @patch("euroeval.metrics.llm_as_a_judge.extract_json_dict_from_string")
    def test_llm_as_a_judge_call_no_valid_outputs_returns_none(
        self,
        mock_extract: MagicMock,
        mock_cache_class: MagicMock,
        mock_llm_model_class: MagicMock,
        simple_metric: LLMAsAJudgeMetric,
        mock_dataset: MagicMock,
        mock_dataset_config: MagicMock,
        mock_benchmark_config: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test None returned when no valid judge outputs."""
        # Return valid JSON dict but with invalid field type
        mock_extract.return_value = {"value": "invalid"}

        # Setup mocks
        mock_cache_instance = MagicMock()
        mock_cache_instance.__contains__ = MagicMock(return_value=False)
        mock_cache_class.return_value = mock_cache_instance

        mock_output = MagicMock()
        mock_output.sequence = '{"value": "invalid_type"}'
        mock_cache_instance.__getitem__ = MagicMock(return_value=mock_output)

        predictions = ["pred1"]
        references = ["ref1"]

        with caplog.at_level(logging.WARNING):
            simple_metric(
                predictions=predictions,
                references=references,
                dataset=mock_dataset,
                dataset_config=mock_dataset_config,
                benchmark_config=mock_benchmark_config,
            )

        assert "No valid judge outputs were produced" in caplog.text


class TestLLMAsAJudgeApplyUserPrompt:
    """Tests for LLMAsAJudgeMetric._apply_user_prompt."""

    def test_llm_as_a_judge_apply_user_prompt_with_condition(
        self, simple_metric: LLMAsAJudgeMetric, simple_response_format: type[BaseModel]
    ) -> None:
        """Test prompt application with condition."""
        metric = LLMAsAJudgeMetric(
            name="test",
            pretty_name="Test",
            judge_id="test-model",
            judge_kwargs=dict(),
            user_prompt="Prediction: {prediction} | Condition: {condition}",
            response_format=simple_response_format,
            scoring_fn=lambda output: output.value / 10.0,
        )

        result = metric._apply_user_prompt(
            prediction="test prediction", condition="test condition"
        )

        assert "test prediction" in result
        assert "test condition" in result

    def test_llm_as_a_judge_apply_user_prompt_without_condition(
        self, simple_metric: LLMAsAJudgeMetric, simple_response_format: type[BaseModel]
    ) -> None:
        """Test prompt application without condition."""
        metric = LLMAsAJudgeMetric(
            name="test",
            pretty_name="Test",
            judge_id="test-model",
            judge_kwargs=dict(),
            user_prompt="Prediction: {prediction}",
            response_format=simple_response_format,
            scoring_fn=lambda output: output.value / 10.0,
        )

        result = metric._apply_user_prompt(prediction="test prediction", condition=None)

        assert "test prediction" in result
        assert "condition" not in result.lower()

    def test_llm_as_a_judge_apply_user_prompt_condition_required_raises_error(
        self, simple_response_format: type[BaseModel]
    ) -> None:
        """Test InvalidBenchmark when condition is required but not provided."""
        metric = LLMAsAJudgeMetric(
            name="test",
            pretty_name="Test",
            judge_id="test-model",
            judge_kwargs=dict(),
            user_prompt="Prediction: {prediction} | Condition: {condition}",
            response_format=simple_response_format,
            scoring_fn=lambda output: output.value / 10.0,
        )

        with pytest.raises(InvalidBenchmark) as exc_info:
            metric._apply_user_prompt(prediction="test prediction", condition=None)

        assert "requires a condition" in str(exc_info.value)

    def test_llm_as_a_judge_apply_user_prompt_with_condition_formatting_fn(
        self, simple_response_format: type[BaseModel]
    ) -> None:
        """Test that condition formatting function is applied."""

        def format_condition(condition: str) -> str:
            return f"FORMATTED: {condition}"

        metric = LLMAsAJudgeMetric(
            name="test",
            pretty_name="Test",
            judge_id="test-model",
            judge_kwargs=dict(),
            user_prompt="Prediction: {prediction} | Condition: {condition}",
            response_format=simple_response_format,
            scoring_fn=lambda output: output.value / 10.0,
            condition_formatting_fn=format_condition,
        )

        result = metric._apply_user_prompt(prediction="test", condition="original")

        assert "FORMATTED: original" in result


class TestLLMAsAJudgeBatchScoringFn:
    """Tests for LLMAsAJudgeMetric._get_batch_scoring_fn."""

    def test_llm_as_a_judge_batch_scoring_fn_from_scoring_fn(
        self, simple_response_format: type[BaseModel]
    ) -> None:
        """Test batch scoring function derived from single scoring function."""

        def scoring_fn(output: MagicMock) -> float:
            return output.value

        metric = LLMAsAJudgeMetric(
            name="test",
            pretty_name="Test",
            judge_id="test-model",
            judge_kwargs=dict(),
            user_prompt="Test: {prediction}",
            response_format=simple_response_format,
            scoring_fn=scoring_fn,
        )

        # Create mock outputs
        outputs = [MagicMock(value=10), MagicMock(value=20), MagicMock(value=30)]

        result = metric.batch_scoring_fn(outputs=outputs, dataset=None)

        assert result == 20.0  # (10 + 20 + 30) / 3

    def test_llm_as_a_judge_batch_scoring_fn_both_raises_error(
        self, simple_response_format: type[BaseModel]
    ) -> None:
        """Test InvalidBenchmark when both scoring_fn and batch_scoring_fn provided."""

        def scoring_fn(output: MagicMock) -> float:
            return output.value

        def batch_scoring_fn(
            outputs: list[simple_response_format], dataset: MagicMock | None = None
        ) -> float:
            return 0.0

        with pytest.raises(InvalidBenchmark) as exc_info:
            LLMAsAJudgeMetric(
                name="test",
                pretty_name="Test",
                judge_id="test-model",
                judge_kwargs=dict(),
                user_prompt="Test: {prediction}",
                response_format=simple_response_format,
                scoring_fn=scoring_fn,
                batch_scoring_fn=batch_scoring_fn,
            )

        assert "Both" in str(exc_info.value) and "are provided" in str(exc_info.value)

    def test_llm_as_a_judge_batch_scoring_fn_neither_raises_error(
        self, simple_response_format: type[BaseModel]
    ) -> None:
        """Test InvalidBenchmark when neither scoring function provided."""
        with pytest.raises(InvalidBenchmark) as exc_info:
            LLMAsAJudgeMetric(
                name="test",
                pretty_name="Test",
                judge_id="test-model",
                judge_kwargs=dict(),
                user_prompt="Test: {prediction}",
                response_format=simple_response_format,
            )

        assert "Neither" in str(exc_info.value) and "are provided" in str(
            exc_info.value
        )


class TestCreateModelGradedFactMetric:
    """Tests for create_model_graded_fact_metric function."""

    def test_create_model_graded_fact_metric_default_judge(self) -> None:
        """Test metric creation with default judge model."""
        metric = create_model_graded_fact_metric()

        assert metric.name == "model_graded_fact"
        assert metric.pretty_name == "Model-Graded Fact"
        assert metric.judge_id == "gpt-5-mini"
        assert metric.judge_kwargs["temperature"] == 1.0

    def test_create_model_graded_fact_metric_custom_judge(self) -> None:
        """Test metric creation with custom judge model."""
        metric = create_model_graded_fact_metric(judge_id="custom-model")

        assert metric.judge_id == "custom-model"

    def test_create_model_graded_fact_metric_custom_scoring_fn(self) -> None:
        """Test metric creation with custom scoring function."""

        def custom_scoring_fn(output: MagicMock) -> float:
            return 2.0 if output.correct else 0.0

        metric = create_model_graded_fact_metric(scoring_fn=custom_scoring_fn)

        # Verify the custom scoring function is used by testing the behavior
        CorrectModel = create_model("CorrectModel", correct=(bool, ...))
        correct_output = CorrectModel(correct=True)
        assert metric.batch_scoring_fn([correct_output]) == 2.0  # Custom scoring

    def test_create_model_graded_fact_metric_custom_response_format(self) -> None:
        """Test metric creation with custom response format."""

        class CustomResponse(BaseModel):
            correct: bool
            confidence: float

        metric = create_model_graded_fact_metric(response_format=CustomResponse)

        assert metric.response_format is CustomResponse
        schema = metric.response_format.model_json_schema()
        assert "correct" in schema["properties"]
        assert "confidence" in schema["properties"]

    def test_create_model_graded_fact_metric_custom_user_prompt(self) -> None:
        """Test metric creation with custom user prompt."""
        custom_prompt = "Custom prompt template"

        metric = create_model_graded_fact_metric(user_prompt=custom_prompt)

        assert metric.user_prompt == custom_prompt

    def test_create_model_graded_fact_metric_custom_system_prompt(self) -> None:
        """Test metric creation with custom system prompt."""
        system_prompt = "You are a fact checker."

        metric = create_model_graded_fact_metric(system_prompt=system_prompt)

        assert metric.system_prompt == system_prompt

    def test_create_model_graded_fact_metric_custom_temperature(self) -> None:
        """Test metric creation with custom temperature."""
        metric = create_model_graded_fact_metric(temperature=0.7)

        assert metric.judge_kwargs["temperature"] == 0.7

    def test_create_model_graded_fact_metric_default_response_format(self) -> None:
        """Test that default response format has correct structure."""
        metric = create_model_graded_fact_metric()

        # Check that the response format has a 'correct' field of type bool
        schema = metric.response_format.model_json_schema()
        assert "correct" in schema["properties"]
        assert schema["properties"]["correct"]["type"] == "boolean"

    def test_create_model_graded_fact_metric_default_scoring_fn(self) -> None:
        """Test that default scoring function returns correct values."""
        metric = create_model_graded_fact_metric()

        CorrectModel = create_model("CorrectModel", correct=(bool, ...))

        # Test with correct=True
        correct_output = CorrectModel(correct=True)
        assert metric.batch_scoring_fn([correct_output]) == 1.0

        # Test with correct=False
        incorrect_output = CorrectModel(correct=False)
        assert metric.batch_scoring_fn([incorrect_output]) == 0.0
