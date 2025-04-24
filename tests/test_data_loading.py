"""Tests for the `data_loading` module."""

from collections.abc import Generator

import pytest
from datasets import DatasetDict
from numpy.random import default_rng
from transformers.models.auto.tokenization_auto import AutoTokenizer

from euroeval.benchmark_modules.litellm import LiteLLMModel
from euroeval.constants import MAX_CONTEXT_LENGTH
from euroeval.data_loading import load_data, load_raw_data
from euroeval.data_models import BenchmarkConfig, DatasetConfig
from euroeval.dataset_configs import get_all_dataset_configs, get_dataset_config


@pytest.fixture(scope="module")
def tokenizer_id() -> Generator[str, None, None]:
    """Fixture for the tokenizer ID."""
    yield "google/gemma-3-27b-it"


class TestLoadData:
    """Tests for the `load_data` function."""

    @pytest.fixture(scope="class")
    def datasets(
        self, benchmark_config: BenchmarkConfig
    ) -> Generator[list[DatasetDict], None, None]:
        """A loaded dataset."""
        yield load_data(
            rng=default_rng(seed=4242),
            dataset_config=get_dataset_config("angry-tweets"),
            benchmark_config=benchmark_config,
        )

    def test_load_data_is_list_of_dataset_dicts(
        self, datasets: list[DatasetDict]
    ) -> None:
        """Test that the `load_data` function returns a list of `DatasetDict`."""
        assert isinstance(datasets, list)
        assert all(isinstance(d, DatasetDict) for d in datasets)

    def test_split_names_are_correct(self, datasets: list[DatasetDict]) -> None:
        """Test that the split names are correct."""
        assert all(set(d.keys()) == {"train", "val", "test"} for d in datasets)

    def test_number_of_iterations_is_correct(
        self, datasets: list[DatasetDict], benchmark_config: BenchmarkConfig
    ) -> None:
        """Test that the number of iterations is correct."""
        assert len(datasets) == benchmark_config.num_iterations

    def test_no_empty_examples(self, datasets: list[DatasetDict]) -> None:
        """Test that there are no empty examples in the datasets."""
        for dataset in datasets:
            for split in dataset.values():
                for feature in ["text", "tokens"]:
                    if feature in split.features:
                        assert all(len(x) > 0 for x in split[feature])


@pytest.mark.parametrize(
    argnames="dataset_config",
    argvalues=[
        cfg
        for cfg in get_all_dataset_configs().values()
        if not cfg.unofficial and not cfg.name == "speed"
    ],
    ids=[
        dataset_name
        for dataset_name, cfg in get_all_dataset_configs().items()
        if not cfg.unofficial and not cfg.name == "speed"
    ],
)
def test_examples_in_official_datasets_are_not_too_long(
    dataset_config: DatasetConfig, benchmark_config: BenchmarkConfig, tokenizer_id: str
) -> None:
    """Test that the examples are not too long in official datasets."""
    dummy_model_config = LiteLLMModel.get_model_config(
        model_id="", benchmark_config=benchmark_config
    )
    dummy_model = LiteLLMModel(
        model_config=dummy_model_config,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    dataset = load_raw_data(
        dataset_config=dataset_config, cache_dir=benchmark_config.cache_dir
    )
    prepared_dataset = dummy_model.prepare_datasets(
        datasets=[dataset], task=dataset_config.task
    )[0]
    for split_name in ["train", "val", "test"]:
        split = prepared_dataset[split_name]
        if "messages" not in split.features:
            continue
        encoded_texts = [
            tokenizer.apply_chat_template(messages) for messages in split["messages"]
        ]
        max_length = max(len(x) for x in encoded_texts)
        assert max_length <= MAX_CONTEXT_LENGTH, (
            f"Max length of {max_length:,} exceeds the maximum context length "
            f"({MAX_CONTEXT_LENGTH:,}) for dataset {dataset_config.name} in split "
            f"{split_name!r}."
        )
