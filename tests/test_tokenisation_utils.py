"""Tests for the `tokenisation_utils` module."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from transformers.models.auto.tokenization_auto import AutoTokenizer

from euroeval.benchmark_modules.hf import load_hf_model_config, load_tokeniser
from euroeval.data_models import BenchmarkConfig, HashableDict
from euroeval.enums import GenerativeType
from euroeval.tokenisation_utils import (
    _label_token_ids_via_chat_diff,
    _pick_matching_label_token,
    get_end_of_chat_token_ids,
    get_first_label_token_mapping,
    should_prefix_space_be_added_to_labels,
    should_prompts_be_stripped,
)
from euroeval.types import Tokeniser


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


def test_get_first_label_token_mapping_falls_back_to_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty chat-template diff should fall back to encoding the label."""
    mapping = _mapping_for_chat_tokeniser(
        tokeniser=_ContaminatedChatTokeniser(include_label_in_template=False),
        monkeypatch=monkeypatch,
    )
    assert mapping == {"négatif": "n", "positif": "pos"}


class _ContaminatedChatTokeniser:
    """Chat tokeniser whose system span contains ``p`` / ``n`` before labels."""

    chat_template = "non-empty"

    def __init__(
        self,
        *,
        include_label_in_template: bool = True,
        extra_span_token: str | None = None,
    ) -> None:
        self.include_label_in_template = include_label_in_template
        self.extra_span_token = extra_span_token
        self._tok2id: dict[str, int] = {}
        self._id2tok: dict[int, str] = {}
        for tok in (
            "sys",
            "p",
            "n",
            "end",
            "<user>",
            "</user>",
            "<assistant>",
            "</assistant>",
            "<gen>",
            "pos",
            "itif",
            "égatif",
        ):
            self._add(tok)
        if extra_span_token is not None:
            self._add(extra_span_token)

    def _add(self, tok: str) -> int:
        if tok not in self._tok2id:
            idx = len(self._tok2id)
            self._tok2id[tok] = idx
            self._id2tok[idx] = tok
        return self._tok2id[tok]

    def __call__(
        self, text: str, add_special_tokens: bool = False, **kwargs: object
    ) -> SimpleNamespace:
        return SimpleNamespace(input_ids=self.encode(text=text))

    def encode(
        self,
        text: str | None = None,
        add_special_tokens: bool = False,
        **kwargs: object,
    ) -> list[int]:
        if text is None:
            text = str(kwargs.get("text", ""))
        text = text.lstrip(" ")
        if text == "positif":
            return [self._add("pos"), self._add("itif")]
        if text == "négatif":
            return [self._add("n"), self._add("égatif")]
        return [self._add(ch) for ch in text]

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        **kwargs: object,
    ) -> list[int] | str:
        toks = ["sys", "p", "n", "end"]
        for msg in conversation:
            role = msg["role"]
            content = msg.get("content") or ""
            toks.append(f"<{role}>")
            if content in {"positif", "négatif"}:
                if self.extra_span_token is not None:
                    toks.append(self.extra_span_token)
                if self.include_label_in_template:
                    if content == "positif":
                        toks.extend(["pos", "itif"])
                    else:
                        toks.extend(["n", "égatif"])
            elif self.include_label_in_template and content:
                toks.append(content)
            toks.append(f"</{role}>")
        if add_generation_prompt:
            toks.append("<gen>")
        ids = [self._add(tok) for tok in toks]
        return ids if tokenize else " ".join(toks)

    def convert_ids_to_tokens(self, ids: list[int], **kwargs: object) -> list[str]:
        return [self._id2tok[int(i)] for i in ids]


def _mapping_for_chat_tokeniser(
    tokeniser: _ContaminatedChatTokeniser, monkeypatch: pytest.MonkeyPatch
) -> dict[str, str] | bool:
    """Return first-label-token mapping for a fake chat tokeniser."""
    monkeypatch.setattr(
        "euroeval.tokenisation_utils.should_prefix_space_be_added_to_labels",
        lambda **kwargs: False,
    )
    dataset_config = SimpleNamespace(
        task=SimpleNamespace(uses_logprobs=True),
        labels=["negative", "positive"],
        prompt_label_mapping={"negative": "négatif", "positive": "positif"},
    )
    model_config = SimpleNamespace(model_id="fake/chat-model")
    # Bypass cache key hashing on SimpleNamespace configs.
    return get_first_label_token_mapping.__wrapped__(
        dataset_config=dataset_config,
        model_config=model_config,
        tokeniser=tokeniser,  # ty: ignore[invalid-argument-type]
        generative_type=GenerativeType.INSTRUCTION_TUNED,
        log_metadata=False,
    )


def test_get_first_label_token_mapping_ignores_system_prompt_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """System tokens like ``p`` must not win over the real label prefix (``pos``)."""
    mapping = _mapping_for_chat_tokeniser(
        tokeniser=_ContaminatedChatTokeniser(), monkeypatch=monkeypatch
    )
    assert mapping == {"négatif": "n", "positif": "pos"}


def test_get_first_label_token_mapping_skips_non_matching_span_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extra tokens in the isolated span must not hide the first label prefix."""
    mapping = _mapping_for_chat_tokeniser(
        tokeniser=_ContaminatedChatTokeniser(extra_span_token="asst"),
        monkeypatch=monkeypatch,
    )
    assert mapping == {"négatif": "n", "positif": "pos"}


def test_label_token_ids_via_chat_diff_isolates_label_span() -> None:
    """Chat-template diff should drop contaminated system tokens."""
    tokeniser = _ContaminatedChatTokeniser()
    ids = _label_token_ids_via_chat_diff(
        label="positif",
        tokeniser=tokeniser,  # ty: ignore[invalid-argument-type]
        enable_thinking=False,
    )
    assert ids is not None
    tokens = tokeniser.convert_ids_to_tokens(ids=ids)
    assert tokens == ["pos", "itif"]


def test_label_token_ids_via_chat_diff_returns_none_when_span_empty() -> None:
    """Identical empty/filled templates should yield no isolated span."""
    tokeniser = _ContaminatedChatTokeniser(include_label_in_template=False)
    ids = _label_token_ids_via_chat_diff(
        label="positif",
        tokeniser=tokeniser,  # ty: ignore[invalid-argument-type]
        enable_thinking=False,
    )
    assert ids is None


@pytest.mark.skipif(
    condition=not os.getenv("HF_TOKEN"),
    reason="HF_TOKEN not set, required for loading tokenizers",
)
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
    assert first_call_args.kwargs.get("pretrained_model_name_or_path") == model_id

    # Verify the second call had use_fast=False (the fallback)
    second_call_args = mock_from_pretrained.call_args_list[1]
    assert second_call_args.kwargs.get("use_fast") is False
    assert second_call_args.kwargs.get("pretrained_model_name_or_path") == model_id

    # Verify that the fallback to the slow tokenizer was used
    assert tokeniser.is_fast is False

    # Verify tokenizer attributes are set
    assert tokeniser.bos_token == "<s>"
    assert tokeniser.eos_token == "</s>"


def test_pick_matching_label_token_uses_first_prefix() -> None:
    """Use the first matching prefix, not a later longer one."""
    assert _pick_matching_label_token(["a", "aa"], "aaa") == "a"
    assert _pick_matching_label_token(["sys", "pos", "itif"], "positif") == "pos"
    assert _pick_matching_label_token(["", "  "], "positif") is None


@pytest.mark.parametrize(
    argnames=["model_id", "expected"],
    argvalues=[("01-ai/Yi-6B", False), ("common-pile/comma-v0.1-2t", True)],
)
@pytest.mark.skipif(
    condition=not os.getenv("HF_TOKEN"),
    reason="HF_TOKEN not set, required for loading tokenizers",
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
    argnames=["model_id", "expected"],
    argvalues=[("01-ai/Yi-6B", True), ("google-bert/bert-base-uncased", False)],
)
@pytest.mark.skipif(
    condition=not os.getenv("HF_TOKEN"),
    reason="HF_TOKEN not set, required for loading tokenizers",
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
