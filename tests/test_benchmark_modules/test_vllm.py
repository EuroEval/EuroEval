"""Unit tests for the `vllm` module."""

import importlib.util
import shutil
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.version
from transformers.models.auto.image_processing_auto import AutoImageProcessor

from euroeval.benchmark_modules.vllm import (
    VLLMModel,
    _is_mistral_tokeniser_model,
    _safe_batch_decode,
    _skip_image_processor_context,
    compute_token_budget,
    load_model,
)
from euroeval.bpc_scoring import compute_bpc_scores
from euroeval.constants import MAX_CONTEXT_LENGTH, REASONING_MAX_TOKENS
from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig
from euroeval.enums import GenerativeType
from euroeval.exceptions import InvalidBenchmark, InvalidModel, NeedsSystemDependency


class TestNvccCheck:
    """Tests for the nvcc system-dependency check in VLLMModel.__init__."""

    @pytest.mark.parametrize(
        argnames=["cuda_available", "hip_version", "nvcc_path", "needs_nvcc"],
        argvalues=[
            (True, None, None, True),
            (True, "5.7.0", None, False),
            (False, None, None, False),
            (True, None, "/usr/bin/nvcc", False),
        ],
        ids=[
            "NVIDIA CUDA without nvcc needs nvcc",
            "ROCm without nvcc does not need nvcc",
            "No CUDA without nvcc does not need nvcc",
            "NVIDIA CUDA with nvcc does not need nvcc",
        ],
    )
    def test_nvcc_check_condition(
        self,
        cuda_available: bool,
        hip_version: str | None,
        nvcc_path: str | None,
        needs_nvcc: bool,
    ) -> None:
        """Test the condition used in the nvcc system-dependency check.

        The check should require nvcc only when CUDA is available on a non-ROCm
        (i.e. NVIDIA) build and nvcc is missing from PATH.
        """
        with (
            patch.object(torch.cuda, "is_available", return_value=cuda_available),
            patch.object(torch.version, "hip", new=hip_version),
            patch.object(shutil, "which", return_value=nvcc_path),
        ):
            condition = (
                torch.cuda.is_available()
                and torch.version.hip is None
                and shutil.which("nvcc") is None
            )

        assert condition == needs_nvcc

    def test_nvcc_check_raises_on_nvidia_without_nvcc(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Test that NeedsSystemDependency is raised on NVIDIA CUDA without nvcc."""
        original_find_spec = importlib.util.find_spec

        def mock_find_spec(name: str) -> object:
            if name == "vllm":
                return MagicMock()
            return original_find_spec(name)

        with (
            patch("importlib.util.find_spec", side_effect=mock_find_spec),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.version, "hip", new=None),
            patch.object(shutil, "which", return_value=None),
        ):
            with pytest.raises(NeedsSystemDependency):
                VLLMModel(
                    model_config=model_config,
                    dataset_config=dataset_config,
                    benchmark_config=benchmark_config,
                    log_metadata=False,
                )

    def test_nvcc_check_skipped_on_rocm(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Test that NeedsSystemDependency is not raised on ROCm without nvcc."""
        original_find_spec = importlib.util.find_spec

        def mock_find_spec(name: str) -> object:
            if name == "vllm":
                return MagicMock()
            return original_find_spec(name)

        with (
            patch("importlib.util.find_spec", side_effect=mock_find_spec),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.version, "hip", new="5.7.0"),
            patch.object(shutil, "which", return_value=None),
        ):
            try:
                VLLMModel(
                    model_config=model_config,
                    dataset_config=dataset_config,
                    benchmark_config=benchmark_config,
                    log_metadata=False,
                )
            except NeedsSystemDependency:
                pytest.fail(
                    "NeedsSystemDependency was raised unexpectedly on ROCm hardware "
                    "without nvcc"
                )
            except Exception:
                pass  # Other exceptions are acceptable in this test


class TestVLLMPromptTruncation:
    """Tests for the BOS-token guard in VLLMModel's instruction-tuned truncation path.

    Regression tests for the bug where `prompt.replace(bos_token, "")` raised a
    `TypeError` when `bos_token` is ``None`` (e.g., Qwen/Qwen3.5-2B).
    """

    @pytest.mark.parametrize(
        argnames=["bos_token"],
        argvalues=[(None,), ("<s>",)],
        ids=["bos_token_none", "bos_token_string"],
    )
    def test_generate_instruction_tuned_truncation_without_bos_token_error(
        self, bos_token: str | None
    ) -> None:
        """Test that the instruction-tuned truncation path handles bos_token=None.

        Regression test: when bos_token is None and generate() calls
        prompt.replace(bos_token, ""), a TypeError was raised. The fix guards
        the replace() call with an explicit None check.
        """
        end_of_chat_token = "<|im_end|>"
        bos_prefix = bos_token or ""
        prompt = (
            f"{bos_prefix}system{end_of_chat_token}"
            f"few_shot_q{end_of_chat_token}"
            f"few_shot_a{end_of_chat_token}"
            f"query"
        )
        max_model_length = 20
        max_generated_tokens = 5
        max_tokens_per_prompt = max_model_length - max_generated_tokens

        # The tokenizer is called multiple times during generate():
        #   1st call  – initial length check (prompt is too long → triggers truncation)
        #   2nd call  – length check after removing one few-shot pair (now fits)
        #   3rd call  – special-token removal on the final completions
        tokenize_call_count = [0]

        def _mock_tokenize(*args, **kwargs) -> MagicMock:
            result = MagicMock()
            tokenize_call_count[0] += 1
            if tokenize_call_count[0] == 1:
                result.input_ids = [list(range(max_tokens_per_prompt))]
            else:
                result.input_ids = [list(range(3))]
            return result

        mock_inner_output = MagicMock()
        mock_inner_output.token_ids = [1, 2, 3]
        mock_vllm_output = MagicMock()
        mock_vllm_output.outputs = [mock_inner_output]

        tokeniser = MagicMock()
        tokeniser.bos_token = bos_token
        tokeniser.pad_token_id = 1
        tokeniser.pad_token = "<pad>"
        tokeniser.eos_token_id = 2
        tokeniser.eos_token = "</s>"
        tokeniser.model_max_length = max_model_length
        tokeniser.decode.return_value = end_of_chat_token
        tokeniser.batch_decode.return_value = ["generated text"]
        tokeniser.side_effect = _mock_tokenize

        model = object.__new__(VLLMModel)
        model._tokeniser = tokeniser
        model.benchmark_config = MagicMock()
        model.benchmark_config.generative_type = GenerativeType.INSTRUCTION_TUNED
        model.end_of_chat_token_ids = (100,)
        model.end_of_reasoning_token = None
        model.custom_stop_tokens = []
        model.log_metadata = False
        model.buffer = {}
        model.model_config = MagicMock()
        model.model_config.model_id = "test-model"
        model.model_config.generation_config = None
        model.dataset_config = MagicMock()
        model.dataset_config.max_generated_tokens = max_generated_tokens
        model.dataset_config.num_few_shot_examples = 1
        model.dataset_config.prompt_label_mapping = {}
        model.dataset_config.labels = []
        model.dataset_config.task.uses_structured_output = False
        model.dataset_config.task.uses_logprobs = False
        model.dataset_config.task.requires_logprobs = False
        model._model = MagicMock()
        model._model.generate.return_value = [mock_vllm_output]

        def _make_sampling_params(**kwargs) -> MagicMock:
            # Mirror the constructor kwargs onto the mock so the code under test reads
            # back the real values (e.g. `prompt_logprobs=None` selects the generation
            # path rather than the BPC scoring path).
            sampling_params = MagicMock()
            for key, value in kwargs.items():
                setattr(sampling_params, key, value)
            return sampling_params

        with (
            patch(
                "euroeval.benchmark_modules.vllm.SamplingParams",
                create=True,
                side_effect=_make_sampling_params,
            ),
            patch(
                "euroeval.benchmark_modules.vllm.get_first_label_token_mapping",
                return_value={},
            ),
        ):
            result = model.generate(inputs={"text": [prompt]})

        assert len(result.sequences) == 1
        assert result.sequences[0] == "generated text"


class TestComputeTokenBudget:
    """Tests for `compute_token_budget`.

    Regression tests for the bug where a model whose context window could not fit
    both the prompt and the dataset's full generation budget (e.g. a 2,048-token
    model on IFEval, which reserves 2,048 generation tokens) had its prompt budget
    collapse to a single token, causing truncation to fail.
    """

    def test_large_context_keeps_full_generation_budget(self) -> None:
        """A model with ample context reserves the full generation budget."""
        generation_budget, max_tokens_per_prompt = compute_token_budget(
            model_max_length=MAX_CONTEXT_LENGTH, max_generated_tokens=2048
        )
        assert generation_budget == 2048
        assert max_tokens_per_prompt == MAX_CONTEXT_LENGTH - 2048

    def test_small_context_shrinks_generation_budget(self) -> None:
        """A model whose context equals the generation budget keeps prompt room.

        The generation budget is shrunk to half the context so the prompt is not
        truncated down to (nearly) nothing.
        """
        generation_budget, max_tokens_per_prompt = compute_token_budget(
            model_max_length=2048, max_generated_tokens=2048
        )
        assert generation_budget == 1024
        assert max_tokens_per_prompt == 1024

    def test_model_max_length_capped_at_max_context_length(self) -> None:
        """The context length is capped at MAX_CONTEXT_LENGTH."""
        generation_budget, max_tokens_per_prompt = compute_token_budget(
            model_max_length=10 * MAX_CONTEXT_LENGTH, max_generated_tokens=2048
        )
        assert generation_budget == 2048
        assert max_tokens_per_prompt == MAX_CONTEXT_LENGTH - 2048

    def test_too_small_context_raises(self) -> None:
        """A context too small to fit any prompt raises a clear benchmark error."""
        with pytest.raises(InvalidBenchmark):
            compute_token_budget(model_max_length=1, max_generated_tokens=2048)


class TestLoadModelMaxModelLen:
    """Tests that load_model passes the correct max_model_len to LLM.

    The max_model_len should be capped at MAX_CONTEXT_LENGTH + REASONING_MAX_TOKENS for
    reasoning models, and at MAX_CONTEXT_LENGTH for all other model types.
    """

    @pytest.mark.parametrize(
        argnames=["generative_type", "true_max_model_len", "expected_max_model_len"],
        argvalues=[
            (
                GenerativeType.REASONING,
                MAX_CONTEXT_LENGTH + REASONING_MAX_TOKENS + 1_000,
                MAX_CONTEXT_LENGTH + REASONING_MAX_TOKENS,
            ),
            (GenerativeType.REASONING, 100, 100),
            (
                GenerativeType.INSTRUCTION_TUNED,
                MAX_CONTEXT_LENGTH + 1_000,
                MAX_CONTEXT_LENGTH,
            ),
            (GenerativeType.INSTRUCTION_TUNED, 100, 100),
            (GenerativeType.BASE, MAX_CONTEXT_LENGTH + 1_000, MAX_CONTEXT_LENGTH),
        ],
        ids=[
            "reasoning_model_large_context",
            "reasoning_model_small_context",
            "instruction_tuned_model_large_context",
            "instruction_tuned_model_small_context",
            "base_model_large_context",
        ],
    )
    def test_load_model_passes_correct_max_model_len_to_llm(
        self,
        generative_type: GenerativeType,
        true_max_model_len: int,
        expected_max_model_len: int,
        model_config: ModelConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Test that load_model passes the correct max_model_len to LLM.

        For reasoning models, max_model_len is capped at
        MAX_CONTEXT_LENGTH + REASONING_MAX_TOKENS; for all other types it is capped at
        MAX_CONTEXT_LENGTH. When the model's true max length is smaller than the cap,
        the true max length is used instead.
        """
        mock_llm_instance = MagicMock()
        mock_hf_model_config = MagicMock(spec=["dtype", "architectures"])
        mock_hf_model_config.dtype = torch.float16
        mock_tokeniser = MagicMock()

        # Build a minimal mock for the vllm module so that vllm.config is accessible
        # inside load_model without requiring vllm to be installed.
        mock_vllm_module = MagicMock()
        mock_vllm_module.config = MagicMock(spec=[])  # no 'attention' attribute

        with (
            patch(
                "euroeval.benchmark_modules.vllm.LLM",
                return_value=mock_llm_instance,
                create=True,
            ) as mock_llm_cls,
            patch(
                "euroeval.benchmark_modules.vllm.vllm",
                new=mock_vllm_module,
                create=True,
            ),
            patch("euroeval.benchmark_modules.vllm.clear_vllm"),
            patch(
                "euroeval.benchmark_modules.vllm.select_backend_and_parallelism",
                return_value=("mp", 1, 1),
            ),
            patch(
                "euroeval.benchmark_modules.vllm.internet_connection_available",
                return_value=True,
            ),
            patch(
                "euroeval.benchmark_modules.vllm.get_vllm_tokenisation_params",
                return_value={},
            ),
        ):
            load_model(
                model_config=model_config,
                benchmark_config=benchmark_config,
                attention_backend=None,
                generative_type=generative_type,
                true_max_model_len=true_max_model_len,
                tokeniser=mock_tokeniser,
                hf_model_config=mock_hf_model_config,
            )

        mock_llm_cls.assert_called_once()
        call_kwargs = mock_llm_cls.call_args.kwargs
        assert call_kwargs["max_model_len"] == expected_max_model_len


class TestLoadModelDisableFlashinferAutotune:
    """Tests for the disable_flashinfer_autotune option."""

    def test_load_model_passes_flashinfer_autotune_when_disabled(
        self, model_config: ModelConfig, benchmark_config: BenchmarkConfig
    ) -> None:
        """Test that load_model passes enable_flashinfer_autotune=False when disabled.

        Verifies that when ``disable_flashinfer_autotune`` is True, the
        ``enable_flashinfer_autotune=False`` parameter is passed to the vLLM
        LLM constructor.
        """
        mock_llm_instance = MagicMock()
        mock_hf_model_config = MagicMock(spec=["dtype", "architectures"])
        mock_hf_model_config.dtype = torch.float16
        mock_hf_model_config.architectures = None
        mock_tokeniser = MagicMock()

        # Enable the disable_flashinfer_autotune option
        benchmark_config.disable_flashinfer_autotune = True

        mock_vllm_module = MagicMock()
        mock_vllm_module.config = MagicMock(spec=[])

        with (
            patch(
                "euroeval.benchmark_modules.vllm.LLM",
                return_value=mock_llm_instance,
                create=True,
            ) as mock_llm_cls,
            patch(
                "euroeval.benchmark_modules.vllm.vllm",
                new=mock_vllm_module,
                create=True,
            ),
            patch("euroeval.benchmark_modules.vllm.clear_vllm"),
            patch(
                "euroeval.benchmark_modules.vllm.select_backend_and_parallelism",
                return_value=("mp", 1, 1),
            ),
            patch(
                "euroeval.benchmark_modules.vllm.internet_connection_available",
                return_value=True,
            ),
            patch(
                "euroeval.benchmark_modules.vllm.get_vllm_tokenisation_params",
                return_value={},
            ),
        ):
            load_model(
                model_config=model_config,
                benchmark_config=benchmark_config,
                attention_backend=None,
                generative_type=GenerativeType.INSTRUCTION_TUNED,
                true_max_model_len=4096,
                tokeniser=mock_tokeniser,
                hf_model_config=mock_hf_model_config,
            )

        mock_llm_cls.assert_called_once()
        call_kwargs = mock_llm_cls.call_args.kwargs
        assert call_kwargs.get("enable_flashinfer_autotune") is False

    def test_load_model_omits_flashinfer_autotune_when_enabled(
        self, model_config: ModelConfig, benchmark_config: BenchmarkConfig
    ) -> None:
        """Test that load_model omits enable_flashinfer_autotune when enabled (default).

        Verifies that when ``disable_flashinfer_autotune`` is False (the
        default), the ``enable_flashinfer_autotune`` parameter is not passed
        to the vLLM LLM constructor.
        """
        mock_llm_instance = MagicMock()
        mock_hf_model_config = MagicMock(spec=["dtype", "architectures"])
        mock_hf_model_config.dtype = torch.float16
        mock_hf_model_config.architectures = None
        mock_tokeniser = MagicMock()

        # Keep default (False) - should not pass the parameter
        benchmark_config.disable_flashinfer_autotune = False

        mock_vllm_module = MagicMock()
        mock_vllm_module.config = MagicMock(spec=[])

        with (
            patch(
                "euroeval.benchmark_modules.vllm.LLM",
                return_value=mock_llm_instance,
                create=True,
            ) as mock_llm_cls,
            patch(
                "euroeval.benchmark_modules.vllm.vllm",
                new=mock_vllm_module,
                create=True,
            ),
            patch("euroeval.benchmark_modules.vllm.clear_vllm"),
            patch(
                "euroeval.benchmark_modules.vllm.select_backend_and_parallelism",
                return_value=("mp", 1, 1),
            ),
            patch(
                "euroeval.benchmark_modules.vllm.internet_connection_available",
                return_value=True,
            ),
            patch(
                "euroeval.benchmark_modules.vllm.get_vllm_tokenisation_params",
                return_value={},
            ),
        ):
            load_model(
                model_config=model_config,
                benchmark_config=benchmark_config,
                attention_backend=None,
                generative_type=GenerativeType.INSTRUCTION_TUNED,
                true_max_model_len=4096,
                tokeniser=mock_tokeniser,
                hf_model_config=mock_hf_model_config,
            )

        mock_llm_cls.assert_called_once()
        call_kwargs = mock_llm_cls.call_args.kwargs
        assert "enable_flashinfer_autotune" not in call_kwargs


class TestComputeBPCFromPromptLogprobs:
    """Tests for `compute_bpc_scores` using prompt_logprobs."""

    @staticmethod
    def _create_mock_output(
        prompt: str,
        prompt_logprobs: list[dict[int, float] | None],
        prompt_token_ids: list[int],
    ) -> MagicMock:
        """Create a mock vLLM output with prompt_logprobs.

        Args:
            prompt: The prompt text.
            prompt_logprobs: List of logprob dicts (or None) per token position.
            prompt_token_ids: The token ids vLLM scored, aligned with prompt_logprobs.

        Returns:
            A MagicMock with prompt, prompt_logprobs and prompt_token_ids attributes.
        """
        output = MagicMock()
        output.prompt = prompt
        output.prompt_logprobs = prompt_logprobs
        output.prompt_token_ids = prompt_token_ids
        return output

    @staticmethod
    def _lp(value: float) -> float:
        """Return a logprob value (natural log).

        Args:
            value: The log probability value.

        Returns:
            The same value (for compatibility with dict-based logprobs).
        """
        return value

    def test_basic_bpc_computation(self) -> None:
        """Test basic BPC computation with simple prompt_logprobs.

        Given a prompt where we know the answer portion and logprobs,
        verify the BPC score is computed correctly.
        """
        # Mock tokeniser
        tokeniser = MagicMock()
        tokeniser.encode.side_effect = (
            lambda text, add_special_tokens=False: (
                [
                    10,  # "Hello"
                    11,  # " world"
                    12,  # "!"
                    13,  # " The"
                    14,  # " answer"
                    15,  # " is"
                    16,  # " yes"
                ][0 : len(text.split())]
                if text
                else []
            )
        )

        # Simulate prompt: "Hello world! The answer is yes"
        # Answer: "yes" (last token)
        # Suppose logprob for "yes" (token 16) is -0.693 (log(0.5))
        prompt = "Hello world! The answer is yes"
        answer = "yes"

        # Create mock prompt_logprobs (one per token, first is None)
        # Positions: [None, {11: lp}, {12: lp}, {13: lp}, {14: lp}, {15: lp}, {16: lp}]
        prompt_logprobs = [
            None,
            {11: -0.1},
            {12: -0.2},
            {13: -0.3},
            {14: -0.4},
            {15: -0.5},
            {16: -0.693},  # log(0.5) ≈ -0.693
        ]

        mock_output = self._create_mock_output(
            prompt, prompt_logprobs, [10, 11, 12, 13, 14, 15, 16]
        )

        # Tokeniser should return 7 tokens for full prompt
        tokeniser.encode.return_value = [10, 11, 12, 13, 14, 15, 16]
        # For prompt without answer ("Hello world! The answer is ")
        tokeniser.encode.side_effect = lambda text, add_special_tokens=False: (
            [10, 11, 12, 13, 14, 15, 16] if "yes" in text else [10, 11, 12, 13, 14, 15]
        )

        bpc_scores = compute_bpc_scores(
            raw_outputs=[mock_output],
            prompts=[prompt],
            answer_texts=[answer],
            answer_start_indices=[6],  # 6 tokens before "yes"
            tokeniser=tokeniser,
        )

        # BPC = -log2(0.5) / len("yes") = 1.0 / 3 ≈ 0.333
        assert len(bpc_scores) == 1
        assert abs(bpc_scores[0] - (1.0 / 3)) < 0.01

    def test_null_prompt_logprobs_returns_inf(self) -> None:
        """Test that None prompt_logprobs returns infinite BPC (worst score)."""
        tokeniser = MagicMock()

        mock_output = MagicMock()
        mock_output.prompt = "test"
        mock_output.prompt_logprobs = None

        bpc_scores = compute_bpc_scores(
            raw_outputs=[mock_output],
            prompts=["test"],
            answer_texts=["answer"],
            answer_start_indices=[0],  # answer starts at token 0
            tokeniser=tokeniser,
        )

        assert bpc_scores == [float("inf")]

    def test_multiple_samples(self) -> None:
        """Test BPC computation for multiple samples."""
        tokeniser = MagicMock()

        # Sample 1: answer "yes" with logprob -0.693 (log(0.5))
        prompt1 = "Q: Is this true? A: yes"
        answer1 = "yes"
        prompt_logprobs1 = [
            None,
            {2: -0.1},
            {3: -0.2},
            {4: -0.693},  # "yes"
        ]

        # Sample 2: answer "no" with logprob -1.609 (log(0.2))
        prompt2 = "Q: Is this true? A: no"
        answer2 = "no"
        prompt_logprobs2 = [
            None,
            {5: -0.3},
            {6: -0.4},
            {7: -1.609},  # "no"
        ]

        mock_output1 = self._create_mock_output(prompt1, prompt_logprobs1, [1, 2, 3, 4])
        mock_output2 = self._create_mock_output(prompt2, prompt_logprobs2, [1, 5, 6, 7])

        # Setup tokeniser
        tokeniser.encode.side_effect = lambda text, add_special_tokens=False: (
            [1, 2, 3, 4]
            if "yes" in text
            else [1, 5, 6, 7]
            if "no" in text
            else [1, 2, 3]
            if answer1 in text
            else [1, 5, 6]
        )

        bpc_scores = compute_bpc_scores(
            raw_outputs=[mock_output1, mock_output2],
            prompts=[prompt1, prompt2],
            answer_texts=[answer1, answer2],
            answer_start_indices=[3, 3],  # answer at token 3 for both
            tokeniser=tokeniser,
        )

        assert len(bpc_scores) == 2
        # Sample 1: BPC = -log2(exp(-0.693)) / 3 = 0.693/ln(2) / 3 ≈ 0.333
        assert abs(bpc_scores[0] - (0.693 / 0.693 / 3)) < 0.01
        # Sample 2: BPC = -log2(exp(-1.609)) / 2 = 1.609/ln(2) / 2 ≈ 1.16
        assert abs(bpc_scores[1] - (1.609 / 0.693 / 2)) < 0.01

    def test_answer_char_counts_override_denominator(self) -> None:
        """An explicit ``answer_char_counts`` is used as the denominator.

        Mirrors NER, where the bits are divided by the entity-text character count
        rather than the full serialised-answer length.
        """
        tokeniser = MagicMock()

        prompt = "Q: Is this true? A: yes"
        answer = "yes"
        prompt_logprobs = [None, {2: -0.1}, {3: -0.2}, {4: -0.693}]  # "yes"
        mock_output = self._create_mock_output(prompt, prompt_logprobs, [1, 2, 3, 4])
        tokeniser.encode.side_effect = lambda text, add_special_tokens=False: (
            [1, 2, 3, 4] if "yes" in text else [1, 2, 3]
        )

        bpc_scores = compute_bpc_scores(
            raw_outputs=[mock_output],
            prompts=[prompt],
            answer_texts=[answer],
            answer_start_indices=[3],
            tokeniser=tokeniser,
            # Divide by 1 character instead of len("yes") == 3.
            answer_char_counts=[1],
        )

        # BPC = -log2(exp(-0.693)) / 1 = 1.0 / 1 ≈ 1.0 (3x the len-based 0.333).
        assert len(bpc_scores) == 1
        assert abs(bpc_scores[0] - 1.0) < 0.01

    def test_left_truncated_prefix_still_aligns_answer(self) -> None:
        """The answer is found by counting back from the end of the scored tokens.

        Simulates a prompt whose prefix was left-truncated to fit the context window:
        the full prompt is 6 tokens (answer_start_idx=5), but vLLM only scored the last
        3 tokens with the answer preserved at the end. Anchoring from the front would
        index past the truncated sequence and collapse to infinity; anchoring from the
        back recovers the answer regardless.
        """
        tokeniser = MagicMock()
        # The full (untruncated) prompt encodes to 6 tokens; the answer is the last one.
        tokeniser.encode.side_effect = lambda text, add_special_tokens=False: [
            10,
            11,
            12,
            13,
            14,
            16,
        ]

        prompt = "A B C D E yes"
        answer = "yes"
        # vLLM scored only the last 3 tokens (prefix truncated from the left).
        prompt_logprobs = [None, {14: -0.2}, {16: -0.693}]  # log(0.5) ≈ -0.693
        mock_output = self._create_mock_output(prompt, prompt_logprobs, [13, 14, 16])

        bpc_scores = compute_bpc_scores(
            raw_outputs=[mock_output],
            prompts=[prompt],
            answer_texts=[answer],
            answer_start_indices=[5],  # would overflow the 3-token scored sequence
            tokeniser=tokeniser,
        )

        # BPC = -log2(0.5) / len("yes") = 1.0 / 3 ≈ 0.333 (finite, not inf).
        assert len(bpc_scores) == 1
        assert abs(bpc_scores[0] - (1.0 / 3)) < 0.01


class TestSkipImageProcessorContext:
    """Tests for _skip_image_processor_context.

    Regression for: models such as IMISLab/Maistros-8B-Instruct-4bit that carry a
    multimodal architecture (Mistral3ForConditionalGeneration) but are released without
    a preprocessor_config.json.  vLLM calls AutoImageProcessor.from_pretrained() for
    such models, which previously raised an unhandled OSError and prevented loading.
    """

    def test_suppresses_missing_image_processor_error(self) -> None:
        """Inside the context, from_pretrained returns None instead of raising."""

        def _raise_missing(
            cls: type,
            pretrained_model_name_or_path: str,
            *args: object,
            **kwargs: object,
        ) -> object:
            raise OSError(
                f"Can't load image processor for '{pretrained_model_name_or_path}'. "
                "If you were trying to load it from 'https://huggingface.co/models', "
                "make sure you don't have a local directory with the same name. "
                "Otherwise, make sure the path contains a preprocessor_config.json"
                " file."
            )

        real_func = AutoImageProcessor.from_pretrained.__func__
        AutoImageProcessor.from_pretrained = classmethod(_raise_missing)  # ty: ignore[invalid-assignment]
        try:
            with pytest.raises(OSError, match="Can't load image processor"):
                AutoImageProcessor.from_pretrained("missing-model")

            with _skip_image_processor_context():
                result = AutoImageProcessor.from_pretrained("missing-model")
            assert result is None

            # context exited: original behaviour restored
            with pytest.raises(OSError, match="Can't load image processor"):
                AutoImageProcessor.from_pretrained("missing-model")
        finally:
            AutoImageProcessor.from_pretrained = classmethod(real_func)  # ty: ignore[invalid-assignment]

    def test_unrelated_oserrors_are_not_suppressed(self) -> None:
        """OSErrors unrelated to image processor loading propagate unchanged."""

        def _raise_other(
            cls: type,
            pretrained_model_name_or_path: str,
            *args: object,
            **kwargs: object,
        ) -> object:
            raise OSError("Some other file-system error")

        real_func = AutoImageProcessor.from_pretrained.__func__
        AutoImageProcessor.from_pretrained = classmethod(_raise_other)  # ty: ignore[invalid-assignment]
        try:
            with _skip_image_processor_context():
                with pytest.raises(OSError, match="Some other file-system error"):
                    AutoImageProcessor.from_pretrained("any-model")
        finally:
            AutoImageProcessor.from_pretrained = classmethod(real_func)  # ty: ignore[invalid-assignment]


class TestLoadModelImageProcessorRetry:
    """Tests that load_model retries with _skip_image_processor_context on OSError.

    Regression for: IMISLab/Maistros-8B-Instruct-4bit and similar models where
    LLM() raises OSError("Can't load image processor ...") because the model repo
    has no preprocessor_config.json.  Before the fix this propagated as InvalidModel;
    after the fix load_model retries once inside _skip_image_processor_context and
    succeeds for text-only inference.
    """

    def test_load_model_retries_on_missing_image_processor(
        self, model_config: ModelConfig, benchmark_config: BenchmarkConfig
    ) -> None:
        """load_model succeeds when LLM() raises the image-processor OSError once."""
        mock_llm_instance = MagicMock()
        mock_hf_model_config = MagicMock(spec=["dtype", "architectures"])
        mock_hf_model_config.dtype = torch.float16
        mock_hf_model_config.architectures = None
        mock_tokeniser = MagicMock()
        mock_vllm_module = MagicMock()
        mock_vllm_module.config = MagicMock(spec=[])

        call_count = [0]

        def _llm_constructor(**kwargs: object) -> object:
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError(
                    "Can't load image processor for 'test-model/4bit'. "
                    "If you were trying to load it from 'https://huggingface.co/models',"
                    " make sure you don't have a local directory with the same name. "
                    "Otherwise, make sure the path contains a preprocessor_config.json"
                    " file."
                )
            return mock_llm_instance

        with (
            patch(
                "euroeval.benchmark_modules.vllm.LLM",
                side_effect=_llm_constructor,
                create=True,
            ),
            patch(
                "euroeval.benchmark_modules.vllm.vllm",
                new=mock_vllm_module,
                create=True,
            ),
            patch("euroeval.benchmark_modules.vllm.clear_vllm"),
            patch(
                "euroeval.benchmark_modules.vllm.select_backend_and_parallelism",
                return_value=("mp", 1, 1),
            ),
            patch(
                "euroeval.benchmark_modules.vllm.internet_connection_available",
                return_value=True,
            ),
            patch(
                "euroeval.benchmark_modules.vllm.get_vllm_tokenisation_params",
                return_value={},
            ),
        ):
            result = load_model(
                model_config=model_config,
                benchmark_config=benchmark_config,
                attention_backend=None,
                generative_type=GenerativeType.INSTRUCTION_TUNED,
                true_max_model_len=4096,
                tokeniser=mock_tokeniser,
                hf_model_config=mock_hf_model_config,
            )

        assert call_count[0] == 2, "LLM should be called twice: first attempt + retry"
        assert result is mock_llm_instance

    def test_load_model_raises_invalid_model_when_retry_also_fails(
        self, model_config: ModelConfig, benchmark_config: BenchmarkConfig
    ) -> None:
        """load_model raises InvalidModel when both attempts fail."""
        mock_hf_model_config = MagicMock(spec=["dtype", "architectures"])
        mock_hf_model_config.dtype = torch.float16
        mock_hf_model_config.architectures = None
        mock_tokeniser = MagicMock()
        mock_vllm_module = MagicMock()
        mock_vllm_module.config = MagicMock(spec=[])

        def _always_fail(**kwargs: object) -> object:
            raise OSError(
                "Can't load image processor for 'test-model/4bit'. "
                "If you were trying to load it from 'https://huggingface.co/models',"
                " make sure you don't have a local directory with the same name."
            )

        with (
            patch(
                "euroeval.benchmark_modules.vllm.LLM",
                side_effect=_always_fail,
                create=True,
            ),
            patch(
                "euroeval.benchmark_modules.vllm.vllm",
                new=mock_vllm_module,
                create=True,
            ),
            patch("euroeval.benchmark_modules.vllm.clear_vllm"),
            patch(
                "euroeval.benchmark_modules.vllm.select_backend_and_parallelism",
                return_value=("mp", 1, 1),
            ),
            patch(
                "euroeval.benchmark_modules.vllm.internet_connection_available",
                return_value=True,
            ),
            patch(
                "euroeval.benchmark_modules.vllm.get_vllm_tokenisation_params",
                return_value={},
            ),
        ):
            with pytest.raises(InvalidModel):
                load_model(
                    model_config=model_config,
                    benchmark_config=benchmark_config,
                    attention_backend=None,
                    generative_type=GenerativeType.INSTRUCTION_TUNED,
                    true_max_model_len=4096,
                    tokeniser=mock_tokeniser,
                    hf_model_config=mock_hf_model_config,
                )


class TestLoadModelMultimodalBudgetRetry:
    """Tests that load_model retries on multimodal budget errors.

    Regression for: Mistral3 and similar models with multimodal architectures that
    raise multimodal budget errors during initialisation. Before the fix this
    propagated as InvalidModel; after the fix load_model retries once with multimodal
    inputs disabled and succeeds for text-only inference.
    """

    def test_load_model_retries_on_multimodal_budget_error(
        self, model_config: ModelConfig, benchmark_config: BenchmarkConfig
    ) -> None:
        """load_model succeeds when LLM() raises a multimodal budget error once."""
        mock_llm_instance = MagicMock()
        mock_hf_model_config = MagicMock(spec=["dtype", "architectures"])
        mock_hf_model_config.dtype = torch.float16
        mock_hf_model_config.architectures = ["Mistral3ForConditionalGeneration"]
        mock_tokeniser = MagicMock()
        mock_vllm_module = MagicMock()
        mock_vllm_module.config = MagicMock(spec=[])

        call_count = [0]

        def _llm_constructor(**kwargs: object) -> object:
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError(
                    "Failed to initialize model: multimodal budget exceeded. "
                    "MM input not allowed when limit_mm_per_prompt is not set."
                )
            return mock_llm_instance

        with (
            patch(
                "euroeval.benchmark_modules.vllm.LLM",
                side_effect=_llm_constructor,
                create=True,
            ),
            patch(
                "euroeval.benchmark_modules.vllm.vllm",
                new=mock_vllm_module,
                create=True,
            ),
            patch("euroeval.benchmark_modules.vllm.clear_vllm"),
            patch(
                "euroeval.benchmark_modules.vllm.select_backend_and_parallelism",
                return_value=("mp", 1, 1),
            ),
            patch(
                "euroeval.benchmark_modules.vllm.internet_connection_available",
                return_value=True,
            ),
            patch(
                "euroeval.benchmark_modules.vllm.get_vllm_tokenisation_params",
                return_value={},
            ),
        ):
            result = load_model(
                model_config=model_config,
                benchmark_config=benchmark_config,
                attention_backend=None,
                generative_type=GenerativeType.INSTRUCTION_TUNED,
                true_max_model_len=4096,
                tokeniser=mock_tokeniser,
                hf_model_config=mock_hf_model_config,
            )

        assert call_count[0] == 2, "LLM should be called twice: first attempt + retry"
        assert result is mock_llm_instance

    def test_load_model_raises_invalid_model_when_multimodal_retry_also_fails(
        self, model_config: ModelConfig, benchmark_config: BenchmarkConfig
    ) -> None:
        """load_model raises InvalidModel when multimodal retry also fails."""
        mock_hf_model_config = MagicMock(spec=["dtype", "architectures"])
        mock_hf_model_config.dtype = torch.float16
        mock_hf_model_config.architectures = ["Mistral3ForConditionalGeneration"]
        mock_tokeniser = MagicMock()
        mock_vllm_module = MagicMock()
        mock_vllm_module.config = MagicMock(spec=[])

        def _always_fail(**kwargs: object) -> object:
            raise RuntimeError(
                "Failed to initialize model: multimodal budget exceeded. "
                "MM input not allowed when limit_mm_per_prompt is not set."
            )

        with (
            patch(
                "euroeval.benchmark_modules.vllm.LLM",
                side_effect=_always_fail,
                create=True,
            ),
            patch(
                "euroeval.benchmark_modules.vllm.vllm",
                new=mock_vllm_module,
                create=True,
            ),
            patch("euroeval.benchmark_modules.vllm.clear_vllm"),
            patch(
                "euroeval.benchmark_modules.vllm.select_backend_and_parallelism",
                return_value=("mp", 1, 1),
            ),
            patch(
                "euroeval.benchmark_modules.vllm.internet_connection_available",
                return_value=True,
            ),
            patch(
                "euroeval.benchmark_modules.vllm.get_vllm_tokenisation_params",
                return_value={},
            ),
        ):
            with pytest.raises(InvalidModel):
                load_model(
                    model_config=model_config,
                    benchmark_config=benchmark_config,
                    attention_backend=None,
                    generative_type=GenerativeType.INSTRUCTION_TUNED,
                    true_max_model_len=4096,
                    tokeniser=mock_tokeniser,
                    hf_model_config=mock_hf_model_config,
                )


class TestSafeBatchDecode:
    """Tests for the `_safe_batch_decode` helper function."""

    def test_batch_decode_success(self) -> None:
        """Test that batch_decode is used when available."""
        mock_tokeniser = MagicMock()
        mock_tokeniser.batch_decode.return_value = ["hello", "world"]

        sequences = [[1, 2, 3], [4, 5, 6]]
        result = _safe_batch_decode(mock_tokeniser, sequences, skip_special_tokens=True)

        assert result == ["hello", "world"]
        mock_tokeniser.batch_decode.assert_called_once_with(
            sequences, skip_special_tokens=True
        )

    def test_fallback_to_individual_decode(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that decode falls back when batch_decode raises AttributeError."""
        mock_tokeniser = MagicMock()
        mock_tokeniser.batch_decode.side_effect = AttributeError(
            "list object has no attribute replace"
        )
        mock_tokeniser.decode.side_effect = lambda seq, skip_special_tokens: (
            f"decoded_{seq}"
        )

        sequences = [[1, 2, 3], [4, 5, 6]]
        result = _safe_batch_decode(
            mock_tokeniser, sequences, skip_special_tokens=False
        )

        assert result == ["decoded_[1, 2, 3]", "decoded_[4, 5, 6]"]
        mock_tokeniser.decode.assert_any_call([1, 2, 3], skip_special_tokens=False)
        mock_tokeniser.decode.assert_any_call([4, 5, 6], skip_special_tokens=False)


class TestIsMistralTokeniserModel:
    """Tests for the `_is_mistral_tokeniser_model` helper function."""

    def test_mistralai_instruct_model_returns_true(self) -> None:
        """A mistralai instruction-tuned model is identified as Mistral."""
        config = MagicMock()
        config.model_type = "llama"
        config.architectures = ["LlamaForCausalLM"]
        assert _is_mistral_tokeniser_model(
            model_id="mistralai/Mistral-7B-Instruct-v0.3", hf_model_config=config
        )

    def test_mistralai_base_model_returns_false(self) -> None:
        """A mistralai base model is not routed to the Mistral common tokeniser."""
        config = MagicMock()
        config.model_type = "mistral"
        config.architectures = ["MistralForCausalLM"]
        assert not _is_mistral_tokeniser_model(
            model_id="mistralai/Mistral-7B-Base-2412", hf_model_config=config
        )

    def test_model_type_mistral_returns_true(self) -> None:
        """A non-mistralai model with model_type 'mistral' is identified as Mistral."""
        config = MagicMock()
        config.model_type = "mistral"
        config.architectures = None
        assert _is_mistral_tokeniser_model(
            model_id="some-user/MyMistralModel", hf_model_config=config
        )

    def test_mistral_architecture_returns_true(self) -> None:
        """A model with a Mistral architecture is identified as Mistral."""
        config = MagicMock()
        config.model_type = "llama"
        config.architectures = ["MistralForCausalLM"]
        assert _is_mistral_tokeniser_model(
            model_id="some-user/MyModel", hf_model_config=config
        )

    def test_mistralai_versioned_base_model_returns_false(self) -> None:
        """A versioned base model like Mistral-7B-v0.1 is not identified as Mistral."""
        config = MagicMock()
        config.model_type = "mistral"
        config.architectures = ["MistralForCausalLM"]
        assert not _is_mistral_tokeniser_model(
            model_id="mistralai/Mistral-7B-v0.1", hf_model_config=config
        )

    def test_non_mistral_community_model_returns_false(self) -> None:
        """A non-Mistral community model is not identified as Mistral."""
        config = MagicMock()
        config.model_type = "gpt2"
        config.architectures = ["GPT2LMHeadModel"]
        assert not _is_mistral_tokeniser_model(
            model_id="gordicaleksa/SlovenianGPT", hf_model_config=config
        )
