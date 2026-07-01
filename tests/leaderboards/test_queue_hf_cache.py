"""Tests for the `leaderboards.queue_hf_cache` module."""

import pytest
from huggingface_hub import ModelInfo

from leaderboards.queue_hf_cache import is_generative_model, is_gguf_model


@pytest.mark.parametrize(
    argnames=["info", "expected"],
    argvalues=[
        # The "gguf" tag alone is enough, regardless of file layout.
        (ModelInfo(id="org/m", tags=["gguf", "qwen"]), True),
        # Tag matching is case-insensitive.
        (ModelInfo(id="org/m", tags=["GGUF"]), True),
        # library_name set to gguf.
        (ModelInfo(id="org/m", library_name="gguf"), True),
        # A .gguf file nested in a per-quant subfolder, no tag.
        (
            ModelInfo(
                id="org/m", siblings=[{"rfilename": "Q4_K_M/model-00001-of-2.gguf"}]
            ),
            True,
        ),
        # A plain safetensors model is not GGUF.
        (
            ModelInfo(
                id="org/m",
                tags=["text-generation"],
                library_name="transformers",
                siblings=[{"rfilename": "model.safetensors"}],
            ),
            False,
        ),
        # Missing/None attributes must not raise.
        (ModelInfo(id="org/m"), False),
    ],
)
def test_is_gguf_model(info: ModelInfo, expected: bool) -> None:
    """GGUF repos are detected via tag, library_name, or any .gguf sibling."""
    assert is_gguf_model(info=info) is expected


@pytest.mark.parametrize(
    argnames=["info", "expected"],
    argvalues=[
        # Generative pipeline tags.
        (ModelInfo(id="org/m", pipeline_tag="text-generation"), True),
        (ModelInfo(id="org/m", pipeline_tag="text2text-generation"), True),
        (ModelInfo(id="org/m", pipeline_tag="image-text-to-text"), True),
        # Encoder pipeline tags.
        (ModelInfo(id="org/m", pipeline_tag="fill-mask"), False),
        (ModelInfo(id="org/m", pipeline_tag="feature-extraction"), False),
        (ModelInfo(id="org/m", pipeline_tag="sentence-similarity"), False),
        # No pipeline tag: fall back to the architecture.
        (ModelInfo(id="org/m", config={"architectures": ["LlamaForCausalLM"]}), True),
        (
            ModelInfo(
                id="org/m", config={"architectures": ["T5ForConditionalGeneration"]}
            ),
            True,
        ),
        (ModelInfo(id="org/m", config={"architectures": ["BertForMaskedLM"]}), False),
        # No pipeline tag and no config: treated as encoder.
        (ModelInfo(id="org/m"), False),
    ],
)
def test_is_generative_model(info: ModelInfo, expected: bool) -> None:
    """Generative models are detected via pipeline tag or LM architecture."""
    assert is_generative_model(info=info) is expected
