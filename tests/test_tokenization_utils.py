"""Unit tests for the `tokenization_utils` module."""

import pytest
from transformers.models.auto.tokenization_auto import AutoTokenizer

from euroeval.benchmark_modules.hf import load_hf_model_config
from euroeval.tokenization_utils import (
    get_end_of_chat_token_ids,
    should_prefix_space_be_added_to_labels,
    should_prompts_be_stripped,
)


@pytest.mark.parametrize(
    argnames=["model_id", "expected"],
    argvalues=[
        ("AI-Sweden-Models/gpt-sw3-6.7b-v2", True),
        ("01-ai/Yi-6B", True),
        ("google-bert/bert-base-uncased", False),
    ],
)
def test_should_prompts_be_stripped(model_id: str, expected: bool, auth: str) -> None:
    """Test that a model ID is a generative model."""
    config = load_hf_model_config(
        model_id=model_id,
        num_labels=0,
        id2label=dict(),
        label2id=dict(),
        revision="main",
        model_cache_dir=None,
        api_key=auth,
        trust_remote_code=True,
        run_with_cli=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, config=config)
    labels = ["positiv", "negativ"]
    strip_prompts = should_prompts_be_stripped(
        labels_to_be_generated=labels, tokenizer=tokenizer
    )
    assert strip_prompts == expected


@pytest.mark.parametrize(
    argnames=["model_id", "expected"],
    argvalues=[("AI-Sweden-Models/gpt-sw3-6.7b-v2", False), ("01-ai/Yi-6B", True)],
)
def test_should_prefix_space_be_added_to_labels(
    model_id: str, expected: bool, auth: str
) -> None:
    """Test whether a prefix space should be added to labels."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=auth)
    labels = ["positiv", "negativ"]
    strip_prompts = should_prefix_space_be_added_to_labels(
        labels_to_be_generated=labels, tokenizer=tokenizer
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=auth, trust_remote_code=True
    )
    end_of_chat_token_ids = get_end_of_chat_token_ids(tokenizer=tokenizer)
    assert end_of_chat_token_ids == expected_token_ids
    if expected_string is not None:
        end_of_chat_string = tokenizer.decode(end_of_chat_token_ids).strip()
        assert end_of_chat_string == expected_string
