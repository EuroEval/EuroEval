"""Unit tests for the `vllm` module."""

import importlib.util
import shutil
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.version

from euroeval.benchmark_modules.vllm import VLLMModel, load_model
from euroeval.constants import MAX_CONTEXT_LENGTH, REASONING_MAX_TOKENS
from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig
from euroeval.enums import GenerativeType
from euroeval.exceptions import NeedsSystemDependency


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

        with (
            patch("euroeval.benchmark_modules.vllm.SamplingParams", create=True),
            patch(
                "euroeval.benchmark_modules.vllm.get_first_label_token_mapping",
                return_value={},
            ),
        ):
            result = model.generate(inputs={"text": [prompt]})

        assert len(result.sequences) == 1
        assert result.sequences[0] == "generated text"


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
            (
                GenerativeType.REASONING,
                100,
                100,
            ),
            (
                GenerativeType.INSTRUCTION_TUNED,
                MAX_CONTEXT_LENGTH + 1_000,
                MAX_CONTEXT_LENGTH,
            ),
            (
                GenerativeType.INSTRUCTION_TUNED,
                100,
                100,
            ),
            (
                GenerativeType.BASE,
                MAX_CONTEXT_LENGTH + 1_000,
                MAX_CONTEXT_LENGTH,
            ),
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
        mock_hf_model_config = MagicMock(spec=["dtype"])
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
