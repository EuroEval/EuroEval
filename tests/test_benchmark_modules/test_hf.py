"""Unit tests for the `hf` module."""

import dataclasses
import hashlib
from unittest.mock import MagicMock, patch

import pytest
import torch
from huggingface_hub.hf_api import HfApi
from transformers.models.xlm_roberta import (
    XLMRobertaConfig,
    XLMRobertaForQuestionAnswering,
)

from euroeval.benchmark_modules.hf import (
    _load_model_from_pretrained,
    get_dtype,
    get_model_repo_info,
    setup_model_for_question_answering,
)
from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig
from euroeval.enums import TaskGroup
from euroeval.exceptions import InvalidModel
from euroeval.model_loading import load_model


class TestBPCGating:
    """Tests that BPC scoring is rejected for non-vLLM backends.

    BPC validation happens in load_model() before backend initialization.
    """

    def test_bpc_rejected_for_hf_encoder(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """BPC scoring raises InvalidModel for HF encoder backend."""
        bpc_config = dataclasses.replace(benchmark_config, use_bits_per_character=True)
        with pytest.raises(InvalidModel, match="vLLM backend"):
            load_model(
                model_config=model_config,
                dataset_config=dataset_config,
                benchmark_config=bpc_config,
            )


@pytest.mark.parametrize(
    argnames=["test_device", "dtype_is_set", "bf16_available", "expected"],
    argvalues=[
        ("cpu", True, True, torch.float32),
        ("cpu", True, False, torch.float32),
        ("cpu", False, True, torch.float32),
        ("cpu", False, False, torch.float32),
        ("mps", True, True, torch.float32),
        ("mps", True, False, torch.float32),
        ("mps", False, True, torch.float32),
        ("mps", False, False, torch.float32),
        ("cuda", True, True, "auto"),
        ("cuda", True, False, "auto"),
        ("cuda", False, True, torch.bfloat16),
        ("cuda", False, False, torch.float16),
    ],
)
def test_get_dtype(
    test_device: str, dtype_is_set: bool, bf16_available: bool, expected: torch.dtype
) -> None:
    """Test that the dtype is set correctly."""
    assert (
        get_dtype(
            device=torch.device(test_device),
            dtype_is_set=dtype_is_set,
            bf16_available=bf16_available,
        )
        == expected
    )


def test_load_model_from_pretrained_keyerror_retry_and_message() -> None:
    """Test KeyError retry and final message for _load_model_from_pretrained.

    KeyError('default') is retried once with ignore_mismatched_sizes, and the final
    InvalidModel includes the model ID, exception repr, and cause.

    Regression test for the EuroBERT/EuroBERT-210m transformers 5 RoPE loading bug.
    """
    mock_model_cls = MagicMock()
    # First call raises KeyError('default'), second call (with retry) also raises it.
    side_effects = [KeyError("default"), KeyError("default")]
    mock_model_cls.from_pretrained.side_effect = side_effects

    with pytest.raises(InvalidModel) as exc_info:
        _load_model_from_pretrained(
            model_cls=mock_model_cls,
            model_id="EuroBERT/EuroBERT-210m",
            model_kwargs={},
            task_group=TaskGroup.SEQUENCE_CLASSIFICATION,
        )

    # Verify the final error message contains model ID, exception repr, and cause.
    message = str(exc_info.value)
    assert "EuroBERT/EuroBERT-210m" in message
    assert "KeyError('default')" in message

    # Verify the exception chain is preserved.
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, KeyError)

    # Verify from_pretrained was called twice: initial attempt + one retry.
    assert mock_model_cls.from_pretrained.call_count == 2


@pytest.mark.parametrize(
    argnames=["repo_files", "requires_safetensors", "model_exists"],
    argvalues=[
        (["model.safetensors", "config.json"], True, True),
        (["pytorch_model.bin", "config.json"], True, False),
        (["pytorch_model.bin", "config.json"], False, True),
        ([], True, False),
    ],
    ids=[
        "Model with safetensors",
        "Model without safetensors",
        "Safetensors check disabled",
        "Empty repo files",
    ],
)
def test_safetensors_check(
    repo_files: list[str],
    requires_safetensors: bool,
    model_exists: bool,
    benchmark_config: BenchmarkConfig,
) -> None:
    """Test the safetensors availability check functionality."""
    with (
        patch.object(HfApi, "list_repo_files") as mock_list_files,
        patch.object(HfApi, "model_info") as mock_model_info,
    ):
        mock_list_files.return_value = repo_files
        mock_model_info.return_value = MagicMock(
            id="test-model", tags=["test"], pipeline_tag="fill-mask"
        )
        hash_model_id = hashlib.md5(
            ",".join(repo_files).encode("utf-8")
            + str(requires_safetensors).encode("utf-8")
        ).hexdigest()
        result = get_model_repo_info(
            model_id=f"model-{hash_model_id}",
            revision="main",
            api_key=benchmark_config.api_key,
            cache_dir=benchmark_config.cache_dir,
            trust_remote_code=benchmark_config.trust_remote_code,
            requires_safetensors=requires_safetensors,
            run_with_cli=benchmark_config.run_with_cli,
        )
        assert (result is not None) == model_exists


def test_setup_model_for_qa_expands_single_row_token_type_embeddings() -> None:
    """Test setup_model_for_qa expands single-row token-type embeddings to two rows.

    Regression test for the fresh-xlm-roberta-base bug: the model's token-type
    embeddings have shape (1, hidden_size) and must be expanded to (2, hidden_size).
    """
    config = XLMRobertaConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        type_vocab_size=1,
    )
    model = XLMRobertaForQuestionAnswering(config)

    # Verify initial state: single-row token-type embeddings
    assert model.config.type_vocab_size == 1
    token_type_embeddings = model.roberta.embeddings.token_type_embeddings
    assert token_type_embeddings.weight.data.shape[0] == 1
    original_row = token_type_embeddings.weight.data[0].clone()

    # Run the setup function
    result = setup_model_for_question_answering(model)

    # Assert the same model object is returned
    assert result is model

    # Assert token-type embeddings expanded to two rows
    assert token_type_embeddings.weight.data.shape[0] == 2

    # Assert the original row is preserved
    assert torch.equal(token_type_embeddings.weight.data[0], original_row)

    # Assert config and embedding sizes are updated
    assert model.config.type_vocab_size == 2
    assert token_type_embeddings.num_embeddings == 2
