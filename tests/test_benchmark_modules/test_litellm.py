"""Unit tests for the `litellm` module."""

import copy
import dataclasses
import re
import typing as t
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litellm.exceptions import BadRequestError, UnsupportedParamsError
from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig
from litellm.types.utils import Choices

from euroeval.benchmark_modules.litellm import (
    MODEL_MAX_LENGTH_MAPPING,
    MODEL_RELEASE_DATE_MAPPING,
    NUM_PARAMS_MAPPING,
    REASONING_MODELS,
    VOCAB_SIZE_MAPPING,
    LiteLLMModel,
    clean_model_id,
    get_api_model_release_date,
)
from euroeval.constants import REASONING_MAX_TOKENS
from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig
from euroeval.enums import ParameterAdjustment
from euroeval.exceptions import InvalidBenchmark, InvalidModel
from euroeval.model_loading import load_model


class TestBPCGating:
    """Tests that BPC scoring is rejected for LiteLLM backend.

    BPC validation happens in load_model() before backend initialization.
    """

    def test_bpc_rejected_for_litellm(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """BPC scoring raises InvalidModel for LiteLLM backend."""
        bpc_config = dataclasses.replace(benchmark_config, use_bits_per_character=True)
        with pytest.raises(InvalidModel, match="vLLM backend"):
            load_model(
                model_config=model_config,
                dataset_config=dataset_config,
                benchmark_config=bpc_config,
            )


class TestCreateModelOutput:
    """Tests for the _create_model_output method in LiteLLMModel."""

    def test_empty_choices_appends_empty_scores(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Test that responses with no choices get empty score lists.

        This is a regression test for a bug where evaluating on AngryTweets raised
        "Sequences and scores must have the same length. Got 1320 sequences and 1319
        scores" when the model returned no choices for some samples. The bug was in
        _create_model_output which appended an empty string to sequences but skipped
        appending to scores when choices were empty.

        The test mocks logprobs as a list of dicts matching ChoiceLogprobs schema
        to trigger the scores append for the valid response.
        """
        # Create a mock response with valid choices and logprobs (as list fallback)
        mock_valid_response = MagicMock()
        mock_choice = MagicMock(spec=Choices)
        mock_message = MagicMock()
        mock_message.content = "positive"
        mock_choice.message = mock_message
        # Mock logprobs as list of dicts matching ChoiceLogprobs schema
        # Each dict needs a 'content' field with list of token logprobs
        mock_choice.logprobs = [
            {
                "content": [
                    {
                        "token": "positive",
                        "logprob": -0.5,
                        "bytes": [112, 111, 115, 105, 116, 105, 118, 101],
                        "top_logprobs": [],
                    }
                ]
            }
        ]
        mock_valid_response.choices = [mock_choice]

        # Create a mock response with empty choices (model ran out of tokens)
        mock_empty_response = MagicMock()
        mock_empty_response.choices = []

        # Create the LiteLLMModel instance
        model = LiteLLMModel(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )

        # This should NOT raise InvalidBenchmark about length mismatch.
        # Without the fix, this raises:
        # "Sequences and scores must have the same length. Got 2 sequences and 1
        # scores."
        output = model._create_model_output(
            model_responses=[mock_valid_response, mock_empty_response],
            model_id="test-model",
        )

        # Verify the output is valid - sequences and scores are aligned
        assert len(output.sequences) == 2
        assert output.sequences[0] == "positive"
        assert output.sequences[1] == ""
        # Scores should be non-None since at least one sample has logprobs
        assert output.scores is not None
        assert len(output.scores) == 2
        # First sample has logprobs, second sample (empty choices) has empty list
        assert output.scores[0] is not None
        assert len(output.scores[0]) == 1
        assert output.scores[1] == []


class TestParameterErrorHandling:
    """Tests for unsupported generation parameter handling."""

    def test_max_completion_tokens_falls_back_to_max_tokens(self) -> None:
        """Unsupported max_completion_tokens is replaced by max_tokens."""
        error = UnsupportedParamsError(
            message="openai does not support parameters: ['max_completion_tokens']"
        )
        model = object.__new__(LiteLLMModel)

        result = model._handle_parameter_error(
            error=error,
            error_msg=str(error).lower(),
            model_id="test-model",
            generation_kwargs={"max_completion_tokens": 128},
        )

        assert result == ({"max_tokens": 128}, 0, ParameterAdjustment.USE_MAX_TOKENS)

    def test_max_tokens_is_removed_when_unsupported(self) -> None:
        """Unsupported max_tokens is removed before retrying the request."""
        error = UnsupportedParamsError(
            message="openai does not support parameters: ['max_tokens']"
        )
        model = object.__new__(LiteLLMModel)

        result = model._handle_parameter_error(
            error=error,
            error_msg=str(error).lower(),
            model_id="test-model",
            generation_kwargs={"max_tokens": 128},
        )

        assert result == ({}, 0, ParameterAdjustment.NO_MAX_TOKENS)


def _make_response(content: str = "positive") -> MagicMock:
    """Build a fake successful LiteLLM `ModelResponse`.

    Args:
        content:
            The text content of the response message.

    Returns:
        A mock response object shaped like a LiteLLM `ModelResponse`.
    """
    response = MagicMock()
    choice = MagicMock(spec=Choices)
    message = MagicMock()
    message.content = content
    choice.message = message
    choice.logprobs = None
    response.choices = [choice]
    return response


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("openai/gpt-4o", "2024-05-13"),
        ("openai/gpt-4o-2024-08-06", "2024-08-06"),
        ("openai/gpt-4-0613", "2023-06-13"),
        ("openai/gpt-3.5-turbo-0125", "2024-01-25"),
        ("openai/o1-pro", "2025-04-14"),
        ("openai/gpt-5-chat-latest", "2025-08-07"),
        ("openai/gpt-5.2-pro", "2025-12-11"),
        ("openai/gpt-5.4-pro", "2026-03-05"),
        ("openai/gpt-5.5-pro", "2026-04-23"),
        ("anthropic/claude-3-5-sonnet-20241022", "2024-10-22"),
        ("anthropic/claude-sonnet-4-6", "2026-02-17"),
        ("anthropic/claude-opus-4-7", "2026-04-16"),
        ("anthropic/claude-mythos-preview", "2026-04-07"),
        ("anthropic/claude-fable-5", "2026-06-09"),
        ("anthropic/claude-sonnet-5", "2026-06-30"),
        ("anthropic/claude-opus-5", "2026-07-24"),
        ("gemini/gemini-3.1-pro-preview", "2026-02-19"),
        ("gemini/gemini-3.1-flash-lite", "2026-03-03"),
        ("gemini/gemini-3.5-flash", "2026-05-19"),
        ("gemini/gemini-3.6-flash", "2026-07-21"),
        ("gemini/gemini-3.7-flash", "2026-08-13"),
        ("xai/grok-4-fast-reasoning", "2025-09-19"),
        ("xai/grok-4.20", "2026-03-10"),
        ("xai/grok-4.5", "2026-07-08"),
        ("xai/grok-4.6", "2026-08-12"),
        ("openai/gpt-5.6-luna", "2026-07-09"),
        ("openai/gpt-5.6", "2026-07-09"),
        ("provider/undated-model", None),
        ("provider/model-2024-99-99", None),
    ],
)
def test_get_api_model_release_date(model_id: str, expected: str | None) -> None:
    """API aliases and dated model IDs resolve to their release dates."""
    assert get_api_model_release_date(model_id) == expected


def test_huggingface_model_id_is_preserved_for_custom_api(
    benchmark_config: BenchmarkConfig,
) -> None:
    """Hugging Face model IDs remain valid for custom OpenAI-compatible APIs."""
    benchmark_config = dataclasses.replace(
        benchmark_config, api_base="https://router.huggingface.co/featherless-ai/v1"
    )
    model_id = "huggingface/mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    assert (
        clean_model_id(model_id=model_id, benchmark_config=benchmark_config) == model_id
    )


def test_litellm_model_config_includes_release_date(
    benchmark_config: BenchmarkConfig,
) -> None:
    """API release metadata is propagated into the model configuration."""
    config = LiteLLMModel.get_model_config(
        model_id="openai/gpt-4o-2024-08-06", benchmark_config=benchmark_config
    )
    assert config.release_date == "2024-08-06"


def test_manual_api_release_date_overrides_embedded_date() -> None:
    """Curated annotations take precedence over dates parsed from model IDs."""
    with patch.dict(
        "euroeval.benchmark_modules.litellm.MODEL_RELEASE_DATE_MAPPING",
        {r"provider/model-2024-01-01": "2024-02-03"},
        clear=True,
    ):
        assert get_api_model_release_date("provider/model-2024-01-01") == "2024-02-03"


RESPONSE_FORMAT_UNAVAILABLE_MESSAGE = "This response_format type is unavailable now"


class TestDeepSeekParams:
    """Tests for provider-specific thinking/reasoning-effort parameter shaping."""

    @pytest.mark.parametrize(
        ("param", "expected"),
        [
            ("thinking", {"thinking": {"type": "enabled"}}),
            ("no-thinking", {"thinking": {"type": "disabled"}}),
            ("low", {"extra_body": {"reasoning_effort": "low"}}),
            ("high", {"extra_body": {"reasoning_effort": "high"}}),
            ("max", {"extra_body": {"reasoning_effort": "max"}}),
        ],
    )
    def test_deepseek_param_shapes(
        self, model_config: ModelConfig, param: str, expected: dict[str, t.Any]
    ) -> None:
        """DeepSeek models get DeepSeek-API-specific thinking/reasoning shapes."""
        model = object.__new__(LiteLLMModel)
        model.buffer = {"first_label_token_mapping": False}
        model.model_config = dataclasses.replace(
            model_config, model_id="deepseek/deepseek-flash", param=param
        )

        result = model._setup_model_params(generation_kwargs={})

        for key, value in expected.items():
            assert result[key] == value

        # DeepSeek must never receive a `budget_tokens` key or a top-level
        # `reasoning_effort` -- LiteLLM's DeepSeek transformation discards both
        thinking = result.get("thinking")
        if isinstance(thinking, dict):
            assert "budget_tokens" not in thinking
        assert "reasoning_effort" not in result

    @pytest.mark.parametrize(
        ("param", "expected_key", "expected_thinking_type", "expected_effort"),
        [
            ("thinking", "thinking", "enabled", None),
            ("low", "reasoning_effort", None, "low"),
        ],
    )
    def test_non_deepseek_param_shapes(
        self,
        model_config: ModelConfig,
        param: str,
        expected_key: str,
        expected_thinking_type: str | None,
        expected_effort: str | None,
    ) -> None:
        """Non-DeepSeek models keep the generic thinking/reasoning-effort shapes."""
        model = object.__new__(LiteLLMModel)
        model.buffer = {"first_label_token_mapping": False}
        model.model_config = dataclasses.replace(
            model_config, model_id="anthropic/claude-sonnet-4-5", param=param
        )

        result = model._setup_model_params(generation_kwargs={})

        if expected_key == "thinking":
            thinking = result["thinking"]
            assert isinstance(thinking, dict)
            assert thinking["type"] == expected_thinking_type
            assert "budget_tokens" in thinking
        else:
            assert result[expected_key] == expected_effort
        assert "extra_body" not in result

    def test_prefix_required(self, model_config: ModelConfig) -> None:
        """The `deepseek/` provider prefix is mandatory for DeepSeek behaviour."""
        bare_model = object.__new__(LiteLLMModel)
        bare_model.buffer = {"first_label_token_mapping": False}
        bare_model.model_config = dataclasses.replace(
            model_config, model_id="deepseek-flash", param="thinking"
        )
        bare_result = bare_model._setup_model_params(generation_kwargs={})
        assert bare_result["thinking"]["type"] == "enabled"
        assert "budget_tokens" in bare_result["thinking"]

        prefixed_model = object.__new__(LiteLLMModel)
        prefixed_model.buffer = {"first_label_token_mapping": False}
        prefixed_model.model_config = dataclasses.replace(
            model_config, model_id="deepseek/deepseek-flash", param="thinking"
        )
        prefixed_result = prefixed_model._setup_model_params(generation_kwargs={})
        assert prefixed_result["thinking"] == {"type": "enabled"}

        deepseek_mappings = [
            VOCAB_SIZE_MAPPING,
            MODEL_MAX_LENGTH_MAPPING,
            NUM_PARAMS_MAPPING,
            MODEL_RELEASE_DATE_MAPPING,
        ]
        for mapping in deepseek_mappings:
            deepseek_patterns = [
                pattern for pattern in mapping if "deepseek" in pattern.lower()
            ]
            assert deepseek_patterns, "Expected a DeepSeek entry in the mapping"
            for pattern in deepseek_patterns:
                assert re.fullmatch(pattern=pattern, string="deepseek-flash") is None
                assert (
                    re.fullmatch(pattern=pattern, string="deepseek/deepseek-flash")
                    is not None
                )

        deepseek_reasoning_patterns = [
            pattern for pattern in REASONING_MODELS if "deepseek" in pattern.lower()
        ]
        assert deepseek_reasoning_patterns
        for pattern in deepseek_reasoning_patterns:
            assert (
                re.fullmatch(
                    pattern=pattern, string="deepseek-flash", flags=re.IGNORECASE
                )
                is None
            )
            assert (
                re.fullmatch(
                    pattern=pattern,
                    string="deepseek/deepseek-flash",
                    flags=re.IGNORECASE,
                )
                is not None
            )

        deepseek_allowed_param_patterns = [
            compiled
            for compiled in LiteLLMModel.allowed_params
            if "deepseek" in compiled.pattern.lower()
        ]
        assert deepseek_allowed_param_patterns
        for compiled in deepseek_allowed_param_patterns:
            assert compiled.fullmatch(string="deepseek-flash") is None
            assert compiled.fullmatch(string="deepseek/deepseek-flash") is not None


class TestDuplicateErrorHandling:
    """Tests that identical parameter errors are only stored once."""

    def test_duplicate_errors_are_stored_once(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Two inputs failing with the same message store exactly one entry."""
        model = LiteLLMModel(
            model_config=dataclasses.replace(model_config, model_id="openai/gpt-4o"),
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )

        original_response_format = {"type": "json_schema", "json_schema": {}}

        async def fake_acompletion(**kwargs: object) -> MagicMock:
            raise _make_response_format_unavailable_error()

        with (
            patch.object(
                target=LiteLLMModel,
                attribute="get_generation_kwargs",
                autospec=True,
                side_effect=_fake_get_generation_kwargs(original_response_format),
            ),
            patch(
                target="euroeval.benchmark_modules.litellm.Router.acompletion",
                new=AsyncMock(side_effect=fake_acompletion),
            ),
        ):
            with pytest.raises(InvalidBenchmark):
                model.generate(
                    inputs={
                        "messages": [
                            [{"role": "user", "content": "hi"}],
                            [{"role": "user", "content": "yo"}],
                        ]
                    }
                )

        assert model._parameter_adjustments == {ParameterAdjustment.NO_JSON_SCHEMA}


def _fake_get_generation_kwargs(
    original_response_format: dict[str, t.Any] | None,
) -> t.Callable[[LiteLLMModel, DatasetConfig], dict[str, t.Any]]:
    """Build a fake `get_generation_kwargs` returning a fixed response format.

    Args:
        original_response_format:
            The response format to return alongside the dataset's max tokens, or
            None to not include a response format at all.

    Returns:
        A function with the same signature as `LiteLLMModel.get_generation_kwargs`.
    """

    def fake_get_generation_kwargs(
        self: LiteLLMModel, dataset_config: DatasetConfig
    ) -> dict[str, t.Any]:
        generation_kwargs: dict[str, t.Any] = {
            "max_completion_tokens": dataset_config.max_generated_tokens
        }
        if original_response_format is not None:
            generation_kwargs["response_format"] = original_response_format
        return generation_kwargs

    return fake_get_generation_kwargs


def _make_response_format_unavailable_error() -> Exception:
    """Build a fresh 'response_format unavailable' error.

    Returns:
        A new `Exception` instance, distinct from any previously built one, so
        that error-identity/deduplication logic under test is exercised
        correctly.
    """
    return Exception(RESPONSE_FORMAT_UNAVAILABLE_MESSAGE)


class TestLiteLLMDeepSeekTransformation:
    """Pins upstream LiteLLM behaviour that motivates DeepSeek param shaping.

    LiteLLM's `DeepSeekChatConfig.map_openai_params` silently drops
    `thinking.budget_tokens` and collapses `reasoning_effort` levels to a
    boolean-ish `thinking.type` ("enabled"/"disabled") before the request
    reaches the DeepSeek API, since DeepSeek's own API only supports
    `{"type": "enabled"}` / `{"type": "disabled"}` and has no `budget_tokens`
    concept. This is the reason `_setup_model_params` in
    `euroeval.benchmark_modules.litellm` shapes DeepSeek parameters itself
    instead of forwarding `thinking`/`reasoning_effort` unchanged. If a future
    LiteLLM release stops dropping these fields, this test will fail and the
    DeepSeek-specific shaping should be revisited.
    """

    def test_reasoning_effort_collapses_to_boolean_thinking_type(self) -> None:
        """`reasoning_effort` levels collapse to a boolean-ish `thinking.type`."""
        config = DeepSeekChatConfig()

        high_effort_params = config.map_openai_params(
            non_default_params={"reasoning_effort": "high"},
            optional_params={},
            model="deepseek-reasoner",
            drop_params=False,
        )
        assert high_effort_params == {"thinking": {"type": "enabled"}}

        none_effort_params = config.map_openai_params(
            non_default_params={"reasoning_effort": "none"},
            optional_params={},
            model="deepseek-reasoner",
            drop_params=False,
        )
        assert none_effort_params == {"thinking": {"type": "disabled"}}

    def test_thinking_budget_tokens_is_dropped(self) -> None:
        """`thinking.budget_tokens` is silently dropped by LiteLLM's mapping."""
        config = DeepSeekChatConfig()
        optional_params = config.map_openai_params(
            non_default_params={"thinking": {"type": "enabled", "budget_tokens": 1234}},
            optional_params={},
            model="deepseek-reasoner",
            drop_params=False,
        )
        assert optional_params == {"thinking": {"type": "enabled"}}
        assert "budget_tokens" not in optional_params["thinking"]


class TestResponseFormatFallback:
    """Tests for the DeepSeek `response_format` json_schema-unavailable fallback."""

    def test_handle_exception_falls_back_to_json_object(
        self, model_config: ModelConfig, dataset_config: DatasetConfig
    ) -> None:
        """The DeepSeek json_schema-unavailable error falls back to `json_object`."""
        model = object.__new__(LiteLLMModel)
        model.model_config = dataclasses.replace(
            model_config, model_id="deepseek/deepseek-flash"
        )
        model.dataset_config = dataset_config
        model._parameter_adjustments = set()

        kwargs, wait_time = model._handle_exception(
            error=_make_response_format_unavailable_error(),
            response_format={"type": "json_schema", "json_schema": {}},
        )

        assert wait_time == 0
        assert kwargs["response_format"] == {"type": "json_object"}


class TestRetryAdjustments:
    """Tests that parameter-error adjustments are replayed without leaking state.

    This is a regression test suite for a bug where `LiteLLMModel.generate()`
    persisted the fully materialised, error-adjusted kwargs into
    `self.generation_kwargs` -- the user-override slot checked first by
    `self.generation_kwargs or self.get_generation_kwargs(...)`. Since the
    benchmarker reuses one model instance across datasets via
    `update_dataset_config()`, a later dataset would silently be evaluated with an
    earlier dataset's `max_completion_tokens` and `response_format` after any retry.
    """

    def test_adjustment_persists_across_attempts(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """A parameter fix learned during retries is replayed on later calls."""
        model = LiteLLMModel(
            model_config=dataclasses.replace(model_config, model_id="openai/gpt-4o"),
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )

        original_response_format = {"type": "json_schema", "json_schema": {}}
        has_raised: list[bool] = [False]

        with _patch_retry_adjustment_dependencies(
            has_raised=has_raised, original_response_format=original_response_format
        ) as calls:
            model.generate(inputs={"messages": [[{"role": "user", "content": "hi"}]]})

            # The first attempt used the original (unavailable) response_format, and
            # the second (successful) attempt used the fallback
            assert len(calls) == 2
            assert calls[0]["response_format"] == original_response_format
            assert calls[1]["response_format"] == {"type": "json_object"}

        # The fix must have been recorded so it can be replayed later
        assert model._parameter_adjustments == {ParameterAdjustment.NO_JSON_SCHEMA}

        # `self.generation_kwargs` must remain the (empty) user override from
        # __init__ -- it must never be overwritten with materialised kwargs
        assert model.generation_kwargs == {}

        # A second batch (e.g. from a new dataset) must not re-raise the same error:
        # the fix should be replayed before any network call is made
        with _patch_retry_adjustment_dependencies(
            has_raised=has_raised, original_response_format=original_response_format
        ) as calls:
            model.generate(inputs={"messages": [[{"role": "user", "content": "hi"}]]})

        assert len(calls) == 1
        assert calls[0]["response_format"] == {"type": "json_object"}
        assert model.generation_kwargs == {}

    def test_dataset_change_after_retry_uses_new_dataset_kwargs(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Dataset-specific kwargs are refreshed, while the model-level fix persists."""
        model = LiteLLMModel(
            model_config=dataclasses.replace(model_config, model_id="openai/gpt-4o"),
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )

        original_response_format = {"type": "json_schema", "json_schema": {}}
        has_raised: list[bool] = [False]

        with _patch_retry_adjustment_dependencies(
            has_raised=has_raised, original_response_format=original_response_format
        ) as calls:
            model.generate(inputs={"messages": [[{"role": "user", "content": "hi"}]]})

        assert model._parameter_adjustments == {ParameterAdjustment.NO_JSON_SCHEMA}

        new_dataset_config = copy.deepcopy(x=dataset_config)
        new_dataset_config.max_generated_tokens = (
            dataset_config.max_generated_tokens or 0
        ) + 1234
        model.update_dataset_config(dataset_config=new_dataset_config)

        with _patch_retry_adjustment_dependencies(
            has_raised=has_raised, original_response_format=original_response_format
        ) as calls:
            model.generate(inputs={"messages": [[{"role": "user", "content": "hi"}]]})

        # The new dataset's max_completion_tokens must be used, and the model-level
        # response_format fix from the previous dataset must still be applied
        assert len(calls) == 1
        expected_tokens = new_dataset_config.max_generated_tokens
        assert calls[0]["max_completion_tokens"] == expected_tokens
        assert calls[0]["response_format"] == {"type": "json_object"}

    def test_get_generation_kwargs_applies_adjustments_before_probe(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """A learned `NO_LOGPROBS` adjustment is applied before the probe request.

        This is a regression test for `get_generation_kwargs()` sending a probe
        request with parameters (like `logprobs`) that we already know this model
        rejects, only converging via `_handle_exception` retries. The fix applies
        `self._parameter_adjustments` to `generation_kwargs` before the probe.
        """
        model = LiteLLMModel(
            model_config=dataclasses.replace(model_config, model_id="openai/gpt-4o"),
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        # Force `_setup_model_params` to add `logprobs`/`top_logprobs`, mirroring
        # what `generate()` does before calling `get_generation_kwargs()`.
        model.buffer["first_label_token_mapping"] = True
        model._parameter_adjustments = {ParameterAdjustment.NO_LOGPROBS}

        async def fake_acompletion(**kwargs: object) -> MagicMock:
            return _make_response()

        with patch(
            target="euroeval.benchmark_modules.litellm.Router.acompletion",
            new=AsyncMock(side_effect=fake_acompletion),
        ) as mock_acompletion:
            model.get_generation_kwargs(dataset_config=dataset_config)

        probe_kwargs = mock_acompletion.call_args.kwargs
        assert "logprobs" not in probe_kwargs
        assert "top_logprobs" not in probe_kwargs

    def test_get_generation_kwargs_keeps_logprobs_without_adjustments(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """With no persisted adjustments, `logprobs` is sent to the probe as usual."""
        model = LiteLLMModel(
            model_config=dataclasses.replace(model_config, model_id="openai/gpt-4o"),
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        model.buffer["first_label_token_mapping"] = True

        async def fake_acompletion(**kwargs: object) -> MagicMock:
            return _make_response()

        with patch(
            target="euroeval.benchmark_modules.litellm.Router.acompletion",
            new=AsyncMock(side_effect=fake_acompletion),
        ) as mock_acompletion:
            model.get_generation_kwargs(dataset_config=dataset_config)

        probe_kwargs = mock_acompletion.call_args.kwargs
        assert probe_kwargs["logprobs"] is True
        assert "top_logprobs" in probe_kwargs

    def test_get_generation_kwargs_reapplies_adjustments_after_reasoning_probe(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """A persisted `USE_MAX_TOKENS` adjustment survives the reasoning probe.

        This is a regression test for the reasoning-content detection block in
        `get_generation_kwargs()` unconditionally re-adding
        `max_completion_tokens` after the probe request, which would otherwise
        leave the returned kwargs holding both `max_tokens` and
        `max_completion_tokens` when `USE_MAX_TOKENS` is persisted. The fix
        re-applies `self._apply_parameter_adjustments()` after the probe loop.
        """
        model = LiteLLMModel(
            model_config=dataclasses.replace(model_config, model_id="openai/gpt-4o"),
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        model._parameter_adjustments.add(ParameterAdjustment.USE_MAX_TOKENS)

        response = _make_response()
        response.choices[0].message.reasoning_content = "thinking"

        async def fake_acompletion(**kwargs: object) -> MagicMock:
            return response

        with patch(
            target="euroeval.benchmark_modules.litellm.Router.acompletion",
            new=AsyncMock(side_effect=fake_acompletion),
        ):
            generation_kwargs = model.get_generation_kwargs(
                dataset_config=dataset_config
            )

        assert "max_completion_tokens" not in generation_kwargs
        assert generation_kwargs["max_tokens"] == REASONING_MAX_TOKENS
        assert model.buffer["uses_reasoning_content"] is True

    def test_logprobs_rejection_leaves_schema_response_format_for_new_dataset(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """A learned logprobs rejection does not strip another dataset's schema."""
        model = LiteLLMModel(
            model_config=dataclasses.replace(model_config, model_id="openai/gpt-4o"),
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )

        async def fake_acompletion_logprobs_unsupported(**kwargs: object) -> MagicMock:
            raise Exception("logprobs is not supported")

        with (
            patch.object(
                target=LiteLLMModel,
                attribute="get_generation_kwargs",
                autospec=True,
                side_effect=_fake_get_generation_kwargs(None),
            ),
            patch(
                target="euroeval.benchmark_modules.litellm.Router.acompletion",
                new=AsyncMock(side_effect=fake_acompletion_logprobs_unsupported),
            ),
        ):
            with pytest.raises(InvalidBenchmark):
                model.generate(
                    inputs={"messages": [[{"role": "user", "content": "hi"}]]}
                )

        assert model._parameter_adjustments == {ParameterAdjustment.NO_LOGPROBS}

        new_response_format = {"type": "json_schema", "json_schema": {}}

        def fake_get_generation_kwargs_with_logprobs(
            self: LiteLLMModel, dataset_config: DatasetConfig
        ) -> dict[str, t.Any]:
            return {
                "max_completion_tokens": dataset_config.max_generated_tokens,
                "response_format": new_response_format,
                "logprobs": True,
                "top_logprobs": 10,
            }

        with (
            patch.object(
                target=LiteLLMModel,
                attribute="get_generation_kwargs",
                autospec=True,
                side_effect=fake_get_generation_kwargs_with_logprobs,
            ),
            patch(
                target="euroeval.benchmark_modules.litellm.Router.acompletion",
                new=AsyncMock(side_effect=lambda **kwargs: _make_response()),
            ) as mock_acompletion,
        ):
            model.generate(inputs={"messages": [[{"role": "user", "content": "hi"}]]})

        sent_kwargs = mock_acompletion.call_args.kwargs
        assert sent_kwargs["response_format"] == new_response_format
        assert "logprobs" not in sent_kwargs
        assert "top_logprobs" not in sent_kwargs

    def test_malformed_schema_does_not_persist_and_new_dataset_schema_is_unchanged(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """A malformed-schema message is request-local and not persisted."""
        model = LiteLLMModel(
            model_config=dataclasses.replace(model_config, model_id="openai/gpt-4o"),
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )

        original_response_format = {"type": "json_schema", "json_schema": {}}

        async def fake_acompletion_malformed_schema(**kwargs: object) -> MagicMock:
            raise Exception("Property keys should match pattern")

        with (
            patch.object(
                target=LiteLLMModel,
                attribute="get_generation_kwargs",
                autospec=True,
                side_effect=_fake_get_generation_kwargs(original_response_format),
            ),
            patch(
                target="euroeval.benchmark_modules.litellm.Router.acompletion",
                new=AsyncMock(side_effect=fake_acompletion_malformed_schema),
            ),
        ):
            with pytest.raises(InvalidBenchmark):
                model.generate(
                    inputs={"messages": [[{"role": "user", "content": "hi"}]]}
                )

        assert ParameterAdjustment.NO_JSON_SCHEMA not in model._parameter_adjustments

        new_response_format = {"type": "json_schema", "json_schema": {"foo": "bar"}}

        def fake_get_generation_kwargs_new_schema(
            self: LiteLLMModel, dataset_config: DatasetConfig
        ) -> dict[str, t.Any]:
            return {
                "max_completion_tokens": dataset_config.max_generated_tokens,
                "response_format": new_response_format,
            }

        with (
            patch.object(
                target=LiteLLMModel,
                attribute="get_generation_kwargs",
                autospec=True,
                side_effect=fake_get_generation_kwargs_new_schema,
            ),
            patch(
                target="euroeval.benchmark_modules.litellm.Router.acompletion",
                new=AsyncMock(side_effect=lambda **kwargs: _make_response()),
            ) as mock_acompletion,
        ):
            model.generate(inputs={"messages": [[{"role": "user", "content": "hi"}]]})

        sent_kwargs = mock_acompletion.call_args.kwargs
        assert sent_kwargs["response_format"] == new_response_format

    def test_schema_rejection_does_not_add_response_format_to_unstructured_dataset(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """A learned JSON-schema rejection leaves an unstructured dataset alone."""
        model = LiteLLMModel(
            model_config=dataclasses.replace(model_config, model_id="openai/gpt-4o"),
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )

        original_response_format = {"type": "json_schema", "json_schema": {}}
        has_raised: list[bool] = [False]

        with _patch_retry_adjustment_dependencies(
            has_raised=has_raised, original_response_format=original_response_format
        ):
            model.generate(inputs={"messages": [[{"role": "user", "content": "hi"}]]})

        assert model._parameter_adjustments == {ParameterAdjustment.NO_JSON_SCHEMA}

        # A later dataset that does not request structured output must not have a
        # `response_format` injected by the persisted adjustment
        with _patch_retry_adjustment_dependencies(
            has_raised=has_raised, original_response_format=None
        ) as calls:
            model.generate(inputs={"messages": [[{"role": "user", "content": "hi"}]]})

        assert len(calls) == 1
        assert "response_format" not in calls[0]

    def test_use_max_tokens_preserves_new_dataset_token_limit(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """`USE_MAX_TOKENS` moves the current dataset's limit, not an old one."""
        model = LiteLLMModel(
            model_config=dataclasses.replace(model_config, model_id="openai/gpt-4o"),
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        model._parameter_adjustments.add(ParameterAdjustment.USE_MAX_TOKENS)

        new_dataset_config = copy.deepcopy(x=dataset_config)
        new_dataset_config.max_generated_tokens = (
            dataset_config.max_generated_tokens or 0
        ) + 1234
        model.update_dataset_config(dataset_config=new_dataset_config)

        has_raised: list[bool] = [True]
        with _patch_retry_adjustment_dependencies(
            has_raised=has_raised, original_response_format=None
        ) as calls:
            model.generate(inputs={"messages": [[{"role": "user", "content": "hi"}]]})

        assert len(calls) == 1
        assert "max_completion_tokens" not in calls[0]
        assert calls[0]["max_tokens"] == new_dataset_config.max_generated_tokens


@contextmanager
def _patch_retry_adjustment_dependencies(
    has_raised: list[bool], original_response_format: dict[str, t.Any] | None
) -> t.Iterator[list[dict[str, t.Any]]]:
    """Patch `get_generation_kwargs` and `Router.acompletion` for retry tests.

    The faked `acompletion` raises a 'response_format unavailable' error on its
    first call (unless `has_raised` is already `True`) and succeeds afterwards.

    Args:
        has_raised:
            A one-element mutable flag shared across patches, so that a single
            failure can be triggered across multiple `generate()` calls.
        original_response_format:
            The response format returned by the faked `get_generation_kwargs`.

    Yields:
        The list of kwargs each `acompletion` call was made with.
    """
    calls: list[dict[str, t.Any]] = []

    async def fake_acompletion(**kwargs: object) -> MagicMock:
        calls.append(kwargs)
        if not has_raised[0]:
            has_raised[0] = True
            raise _make_response_format_unavailable_error()
        return _make_response()

    with (
        patch.object(
            target=LiteLLMModel,
            attribute="get_generation_kwargs",
            autospec=True,
            side_effect=_fake_get_generation_kwargs(original_response_format),
        ),
        patch(
            target="euroeval.benchmark_modules.litellm.Router.acompletion",
            new=AsyncMock(side_effect=fake_acompletion),
        ),
    ):
        yield calls


class TestServiceErrorHandling:
    """Tests that service-type errors are not persisted as parameter fixes."""

    def test_service_error_is_not_persisted(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """A service-type error resolved with wait time 0 is not stored for replay."""
        benchmark_config = dataclasses.replace(
            benchmark_config, api_base="http://localhost:8000"
        )
        model = LiteLLMModel(
            model_config=dataclasses.replace(model_config, model_id="openai/gpt-4o"),
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )

        has_raised: list[bool] = [False]

        async def fake_acompletion(**kwargs: object) -> MagicMock:
            if not has_raised[0]:
                has_raised[0] = True
                raise BadRequestError(
                    message="Bad request", model="gpt-4o", llm_provider="openai"
                )
            return _make_response()

        with patch(
            target="euroeval.benchmark_modules.litellm.Router.acompletion",
            new=AsyncMock(side_effect=fake_acompletion),
        ):
            model.generate(inputs={"messages": [[{"role": "user", "content": "hi"}]]})

        # The api_base fix is a service-level adjustment and must not be persisted
        assert model._parameter_adjustments == set()
        assert model.benchmark_config.api_base == "http://localhost:8000/v1"

        # A second call must succeed without raising InvalidBenchmark
        has_raised[0] = True
        with patch(
            target="euroeval.benchmark_modules.litellm.Router.acompletion",
            new=AsyncMock(side_effect=fake_acompletion),
        ):
            model.generate(inputs={"messages": [[{"role": "user", "content": "hi"}]]})


class TestTransientParameterErrors:
    """Tests that transient parameter errors are fixed but not persisted."""

    def test_json_schema_unsupported_message_is_persisted(
        self, model_config: ModelConfig, dataset_config: DatasetConfig
    ) -> None:
        """The `'json_schema' is not supported` message is persisted."""
        model = object.__new__(LiteLLMModel)
        model.model_config = dataclasses.replace(model_config, model_id="openai/gpt-4o")
        model.dataset_config = dataset_config
        model.buffer = {"first_label_token_mapping": True}
        model._parameter_adjustments = set()
        model._max_thinking_budget = None

        error_message = "'json_schema' is not supported"
        result = model._handle_parameter_error(
            error=Exception(error_message),
            error_msg=error_message.lower(),
            model_id="test-model",
            generation_kwargs={"response_format": {"type": "json_schema"}},
        )

        assert result is not None
        kwargs, wait_time, adjustment = result
        assert wait_time == 0
        assert adjustment == ParameterAdjustment.NO_JSON_SCHEMA
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_logprobs_quota_message_is_not_persisted(
        self, model_config: ModelConfig, dataset_config: DatasetConfig
    ) -> None:
        """The temporary logprobs quota message does not permanently disable them."""
        model = object.__new__(LiteLLMModel)
        model.model_config = dataclasses.replace(model_config, model_id="openai/gpt-4o")
        model.dataset_config = dataset_config
        model.buffer = {"first_label_token_mapping": True}
        model._parameter_adjustments = set()
        model._max_thinking_budget = None

        kwargs, wait_time = model._handle_exception(
            error=Exception(
                "You've reached the maximum number of requests with logprobs"
            ),
            logprobs=True,
            top_logprobs=10,
        )

        assert wait_time == 0
        assert "logprobs" not in kwargs
        assert "top_logprobs" not in kwargs
        assert model._parameter_adjustments == set()

    def test_logprobs_unsupported_message_is_persisted(
        self, model_config: ModelConfig, dataset_config: DatasetConfig
    ) -> None:
        """An explicit unsupported-logprobs message is persisted across datasets."""
        model = object.__new__(LiteLLMModel)
        model.model_config = dataclasses.replace(model_config, model_id="openai/gpt-4o")
        model.dataset_config = dataset_config
        model.buffer = {"first_label_token_mapping": True}
        model._parameter_adjustments = set()
        model._max_thinking_budget = None

        model._handle_exception(
            error=Exception("logprobs is not supported"), logprobs=True
        )

        assert model._parameter_adjustments == {ParameterAdjustment.NO_LOGPROBS}

    def test_malformed_schema_message_is_not_persisted(
        self, model_config: ModelConfig, dataset_config: DatasetConfig
    ) -> None:
        """A malformed-schema message is request-local and not persisted."""
        model = object.__new__(LiteLLMModel)
        model.model_config = dataclasses.replace(model_config, model_id="openai/gpt-4o")
        model.dataset_config = dataset_config
        model.buffer = {"first_label_token_mapping": True}
        model._parameter_adjustments = set()
        model._max_thinking_budget = None

        error_message = "Property keys should match pattern"
        result = model._handle_parameter_error(
            error=Exception(error_message),
            error_msg=error_message.lower(),
            model_id="test-model",
            generation_kwargs={"response_format": {"type": "json_schema"}},
        )

        assert result is not None
        kwargs, wait_time, adjustment = result
        assert wait_time == 0
        assert adjustment is None
        assert kwargs["response_format"] == {"type": "json_object"}
        assert model._parameter_adjustments == set()

    @pytest.mark.parametrize(
        "error_message", ["'maxitems' is not supported", "must contain the word 'json'"]
    )
    def test_request_local_response_format_messages_are_not_persisted(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        error_message: str,
    ) -> None:
        """Request-local `response_format` errors drop the format without persisting."""
        model = object.__new__(LiteLLMModel)
        model.model_config = dataclasses.replace(model_config, model_id="openai/gpt-4o")
        model.dataset_config = dataset_config
        model.buffer = {"first_label_token_mapping": True}
        model._parameter_adjustments = set()
        model._max_thinking_budget = None

        result = model._handle_parameter_error(
            error=Exception(error_message),
            error_msg=error_message.lower(),
            model_id="test-model",
            generation_kwargs={"response_format": {"type": "json_object"}},
        )

        assert result is not None
        kwargs, wait_time, adjustment = result
        assert wait_time == 0
        assert adjustment is None
        assert "response_format" not in kwargs
        assert model._parameter_adjustments == set()

    def test_unexpected_keyword_response_format_message_is_persisted(
        self, model_config: ModelConfig, dataset_config: DatasetConfig
    ) -> None:
        """The generic `response_format` unsupported message is persisted."""
        model = object.__new__(LiteLLMModel)
        model.model_config = dataclasses.replace(model_config, model_id="openai/gpt-4o")
        model.dataset_config = dataset_config
        model.buffer = {"first_label_token_mapping": True}
        model._parameter_adjustments = set()
        model._max_thinking_budget = None

        error_message = "got an unexpected keyword argument 'response_format'"
        result = model._handle_parameter_error(
            error=Exception(error_message),
            error_msg=error_message.lower(),
            model_id="test-model",
            generation_kwargs={"response_format": {"type": "json_object"}},
        )

        assert result is not None
        kwargs, wait_time, adjustment = result
        assert wait_time == 0
        assert adjustment == ParameterAdjustment.NO_RESPONSE_FORMAT
        assert "response_format" not in kwargs
