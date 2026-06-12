"""Tests for the `tokenisation_utils` module."""

from unittest.mock import MagicMock, patch

import pytest
from transformers.models.auto.tokenization_auto import AutoTokenizer

from euroeval.benchmark_modules.hf import load_hf_model_config, load_tokeniser
from euroeval.data_models import BenchmarkConfig, HashableDict
from euroeval.tokenisation_utils import (
    get_end_of_chat_token_ids,
    should_prefix_space_be_added_to_labels,
    should_prompts_be_stripped,
)
from euroeval.types import Tokeniser


@pytest.mark.parametrize(
    argnames=["model_id", "expected"],
    argvalues=[("01-ai/Yi-6B", True), ("google-bert/bert-base-uncased", False)],
)
def test_should_prompts_be_stripped(model_id: str, expected: bool, auth: str) -> None:
    """Test that a model ID is a generative model."""
    config = load_hf_model_config(
        model_id=model_id,
        num_labels=0,
        id2label=HashableDict(),
        label2id=HashableDict(),
        revision="main",
        model_cache_dir=None,
        api_key=auth,
        trust_remote_code=True,
        run_with_cli=True,
    )
    tokeniser: Tokeniser = AutoTokenizer.from_pretrained(  # ty: ignore[invalid-assignment]
        model_id, config=config
    )
    labels = ["positiv", "negativ"]
    strip_prompts = should_prompts_be_stripped(
        labels_to_be_generated=labels, tokeniser=tokeniser
    )
    assert strip_prompts == expected


@pytest.mark.parametrize(
    argnames=["model_id", "expected"],
    argvalues=[("01-ai/Yi-6B", False), ("common-pile/comma-v0.1-2t", True)],
)
def test_should_prefix_space_be_added_to_labels(
    model_id: str, expected: bool, auth: str
) -> None:
    """Test whether a prefix space should be added to labels."""
    tokeniser: Tokeniser = AutoTokenizer.from_pretrained(  # ty: ignore[invalid-assignment]
        model_id, token=auth
    )
    labels = ["positiv", "negativ"]
    strip_prompts = should_prefix_space_be_added_to_labels(
        labels_to_be_generated=labels, tokeniser=tokeniser
    )
    assert strip_prompts == expected


@pytest.mark.parametrize(
    argnames=["model_id", "expected_token_ids", "expected_string"],
    argvalues=[
        ("occiglot/occiglot-7b-de-en", None, None),
        ("occiglot/occiglot-7b-de-en-instruct", [32001, 28705, 13], "<|im_end|>"),
        ("mhenrichsen/danskgpt-tiny", None, None),
        ("mhenrichsen/danskgpt-tiny-chat", [32000, 13], "<|im_end|>"),
        ("mayflowergmbh/Wiedervereinigung-7b-dpo", None, None),
        ("Qwen/Qwen1.5-0.5B-Chat", [151645, 198], "<|im_end|>"),
        ("norallm/normistral-7b-warm", None, None),
        ("norallm/normistral-7b-warm-instruct", [4, 217], "<|im_end|>"),
        ("ibm-granite/granite-3b-code-instruct-2k", [478], ""),
    ],
)
def test_get_end_of_chat_token_ids(
    model_id: str,
    expected_token_ids: list[int] | None,
    expected_string: str | None,
    auth: str,
) -> None:
    """Test ability to get the chat token IDs of a model."""
    tokeniser: Tokeniser = AutoTokenizer.from_pretrained(  # ty: ignore[invalid-assignment]
        model_id, token=auth, trust_remote_code=True
    )
    end_of_chat_token_ids = get_end_of_chat_token_ids(
        tokeniser=tokeniser, generative_type=None
    )
    assert end_of_chat_token_ids == expected_token_ids
    if expected_string is not None:
        assert end_of_chat_token_ids is not None
        end_of_chat_string = tokeniser.decode(list(end_of_chat_token_ids)).strip()
        assert end_of_chat_string == expected_string


def test_load_xlmr_tokeniser_with_fallback(
    auth: str, benchmark_config: BenchmarkConfig
) -> None:
    """Test that XLM-RoBERTa tokenizers load with use_fast=False fallback.

    Regression test for EMBEDDIA/litlat-bert and similar XLM-RoBERTa variants
    that raise TypeError when loading fast tokenizers.
    """
    model_id = "EMBEDDIA/litlat-bert"

    # Create a mock model config to avoid network calls
    mock_model_config = MagicMock()
    mock_model_config.param = None
    mock_model_config.model_cache_dir = None

    # Create a mock slow tokenizer
    mock_slow_tokeniser = MagicMock()
    mock_slow_tokeniser.is_fast = False
    mock_slow_tokeniser.bos_token = "<s>"
    mock_slow_tokeniser.eos_token = "</s>"
    mock_slow_tokeniser.bos_token_id = 0
    mock_slow_tokeniser.eos_token_id = 2
    mock_slow_tokeniser.pad_token = None
    mock_slow_tokeniser.pad_token_id = None

    # Mock AutoTokenizer.from_pretrained to:
    # 1. First raise TypeError when use_fast=True (simulating the XLM-R issue)
    # 2. Then return the mock slow tokenizer when use_fast=False (the fallback)
    with patch.object(
        AutoTokenizer,
        "from_pretrained",
        side_effect=[TypeError("fast tokenizer not supported"), mock_slow_tokeniser],
    ) as mock_from_pretrained:
        tokeniser: Tokeniser = load_tokeniser(
            model=None,
            model_id=model_id,
            trust_remote_code=benchmark_config.trust_remote_code,
            model_config=mock_model_config,
        )

    # Verify that AutoTokenizer.from_pretrained was called twice
    # (once with use_fast=True, once with use_fast=False)
    assert mock_from_pretrained.call_count == 2

    # Verify the first call had use_fast=True
    first_call_args = mock_from_pretrained.call_args_list[0]
    assert first_call_args.kwargs.get("use_fast") is True
    assert first_call_args.args[0] == model_id

    # Verify the second call had use_fast=False (the fallback)
    second_call_args = mock_from_pretrained.call_args_list[1]
    assert second_call_args.kwargs.get("use_fast") is False
    assert second_call_args.args[0] == model_id

    # Verify that the fallback to the slow tokenizer was used
    assert tokeniser.is_fast is False

    # Verify tokenizer attributes are set
    assert tokeniser.bos_token == "<s>"
    assert tokeniser.eos_token == "</s>"
