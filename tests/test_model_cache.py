"""Tests for the `model_cache` module."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from datasets import Dataset

from euroeval.data_models import GenerativeModelOutput, SingleGenerativeModelOutput
from euroeval.model_cache import (
    ModelCache,
    create_model_cache_dir,
    load_cached_model_outputs,
    split_dataset_into_cached_and_non_cached,
)


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    return tmp_path / "cache"


@pytest.fixture
def model_cache(cache_dir: Path) -> ModelCache:
    """Create a ModelCache instance for testing."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return ModelCache(
        model_cache_dir=cache_dir,
        cache_name="test_cache.json",
        max_generated_tokens=50,
        progress_bar=False,
        hash_inputs=False,
    )


def test_model_cache_initialization(cache_dir: Path) -> None:
    """Test ModelCache initialization."""
    cache = ModelCache(
        model_cache_dir=cache_dir,
        cache_name="test.json",
        max_generated_tokens=100,
        progress_bar=True,
        hash_inputs=True,
    )
    assert cache.model_cache_dir == cache_dir
    assert cache.max_generated_tokens == 100
    assert cache.progress_bar is True
    assert cache.hash_inputs is True


def test_model_cache_load_empty(model_cache: ModelCache) -> None:
    """Test loading an empty cache."""
    model_cache.load()
    assert isinstance(model_cache.cache, dict)
    assert len(model_cache.cache) == 0


def test_model_cache_save_and_load(model_cache: ModelCache) -> None:
    """Test saving and loading the cache."""
    model_cache.load()
    model_cache["key1"] = SingleGenerativeModelOutput(sequence="output1", scores=None)
    model_cache.save()

    # Create new cache instance and load
    new_cache = ModelCache(
        model_cache_dir=model_cache.model_cache_dir,
        cache_name="test_cache.json",
        max_generated_tokens=50,
        progress_bar=False,
        hash_inputs=False,
    )
    new_cache.load()

    assert "key1" in new_cache
    assert new_cache["key1"].sequence == "output1"


def test_model_cache_getitem_setitem(model_cache: ModelCache) -> None:
    """Test getting and setting items in the cache."""
    model_cache.load()

    output = SingleGenerativeModelOutput(sequence="test_output", scores=[0.1, 0.9])
    model_cache["test_key"] = output

    retrieved = model_cache["test_key"]
    assert retrieved.sequence == "test_output"
    assert retrieved.scores == [0.1, 0.9]


def test_model_cache_contains(model_cache: ModelCache) -> None:
    """Test the __contains__ method."""
    model_cache.load()

    assert "missing_key" not in model_cache

    model_cache["existing_key"] = SingleGenerativeModelOutput(
        sequence="output", scores=None
    )

    assert "existing_key" in model_cache


def test_model_cache_hash_key(cache_dir: Path) -> None:
    """Test that hashing is applied when enabled."""
    cache_with_hash = ModelCache(
        model_cache_dir=cache_dir,
        cache_name="hash_test.json",
        max_generated_tokens=50,
        progress_bar=False,
        hash_inputs=True,
    )
    cache_with_hash.load()

    key = "test_input"
    cache_with_hash[key] = SingleGenerativeModelOutput(sequence="output", scores=None)

    # The actual key stored should be hashed
    assert key not in cache_with_hash.cache
    # But the cache should still contain the entry
    assert key in cache_with_hash


def test_model_cache_remove(model_cache: ModelCache) -> None:
    """Test removing the cache."""
    model_cache.load()
    model_cache["key"] = SingleGenerativeModelOutput(sequence="output", scores=None)
    model_cache.save()

    cache_file = model_cache.cache_path
    assert cache_file.exists()

    model_cache.remove()
    assert not cache_file.exists()


def test_model_cache_add_to_cache(model_cache: ModelCache) -> None:
    """Test adding model outputs to the cache."""
    model_cache.load()

    model_inputs = {"text": ["input1", "input2"]}
    model_output = GenerativeModelOutput(
        sequences=["output1", "output2"], scores=None
    )

    model_cache.add_to_cache(model_inputs, model_output)

    assert "input1" in model_cache
    assert "input2" in model_cache
    assert model_cache["input1"].sequence == "output1"
    assert model_cache["input2"].sequence == "output2"


def test_model_cache_add_to_cache_with_messages(model_cache: ModelCache) -> None:
    """Test adding model outputs with messages format."""
    model_cache.load()

    model_inputs = {"messages": [["msg1"], ["msg2"]]}
    model_output = GenerativeModelOutput(sequences=["out1", "out2"], scores=None)

    model_cache.add_to_cache(model_inputs, model_output)

    assert ["msg1"] in model_cache
    assert ["msg2"] in model_cache


def test_split_dataset_into_cached_and_non_cached(model_cache: ModelCache) -> None:
    """Test splitting a dataset based on cache."""
    model_cache.load()

    # Add some entries to cache
    model_cache["cached_text"] = SingleGenerativeModelOutput(
        sequence="cached_output", scores=None
    )

    # Create a dataset with both cached and non-cached items
    dataset = Dataset.from_dict(
        {"text": ["cached_text", "new_text1", "new_text2", "cached_text"]}
    )

    cached, non_cached = split_dataset_into_cached_and_non_cached(dataset, model_cache)

    assert len(cached) == 2  # "cached_text" appears twice
    assert len(non_cached) == 2  # "new_text1" and "new_text2"


def test_load_cached_model_outputs(model_cache: ModelCache) -> None:
    """Test loading cached model outputs from a dataset."""
    model_cache.load()

    # Add entries to cache
    model_cache["text1"] = SingleGenerativeModelOutput(sequence="output1", scores=None)
    model_cache["text2"] = SingleGenerativeModelOutput(sequence="output2", scores=None)

    # Create a cached dataset
    cached_dataset = Dataset.from_dict({"text": ["text1", "text2"]})

    result = load_cached_model_outputs(cached_dataset, model_cache)

    assert isinstance(result, GenerativeModelOutput)
    assert result.sequences == ["output1", "output2"]


def test_create_model_cache_dir_from_path(tmp_path: Path) -> None:
    """Test creating cache dir for a local model path."""
    model_path = tmp_path / "local_model"
    model_path.mkdir(parents=True, exist_ok=True)

    result = create_model_cache_dir(cache_dir="cache", model_id=str(model_path))

    # Should return the model path itself
    assert result == str(model_path)


def test_create_model_cache_dir_from_hub_id() -> None:
    """Test creating cache dir for a HuggingFace Hub model ID."""
    cache_dir = "test_cache"
    model_id = "org/model-name"

    result = create_model_cache_dir(cache_dir=cache_dir, model_id=model_id)

    # Should create a path based on the model ID
    assert "model_cache" in result
    assert "org--model-name" in result


def test_model_cache_load_corrupted(model_cache: ModelCache, caplog: pytest.LogCaptureFixture) -> None:
    """Test loading a corrupted cache file."""
    # Create a corrupted JSON file
    model_cache.cache_path.parent.mkdir(parents=True, exist_ok=True)
    model_cache.cache_path.write_text("{ this is not valid json }")

    with caplog.at_level("WARNING"):
        model_cache.load()

    # Should re-initialize with empty cache
    assert isinstance(model_cache.cache, dict)
    assert len(model_cache.cache) == 0
    assert "Failed to load the cache" in caplog.text
