"""Tests for the `data_loading` module."""

import os
from collections.abc import Generator
from functools import partial
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict
from numpy.random import default_rng
from transformers.models.auto.tokenization_auto import AutoTokenizer

from euroeval.benchmark_modules.litellm import LiteLLMModel
from euroeval.constants import MAX_CONTEXT_LENGTH
from euroeval.data_loading import load_data, load_raw_data
from euroeval.data_models import BenchmarkConfig, DatasetConfig
from euroeval.dataset_configs import get_all_dataset_configs
from euroeval.enums import GenerativeType
from euroeval.exceptions import InvalidBenchmark
from euroeval.generation_utils import apply_prompt, extract_few_shot_examples
from euroeval.tasks import RC


@pytest.mark.parametrize(
    argnames="dataset_config",
    argvalues=[
        dataset_config
        for dataset_config in get_all_dataset_configs(
            custom_datasets_file=Path("custom_datasets.py"),
            dataset_ids=[],
            api_key=os.getenv("HF_TOKEN"),
            cache_dir=Path(".euroeval_cache"),
            trust_remote_code=True,
            run_with_cli=True,
        ).values()
        if os.getenv("CHECK_DATASET") is not None
        and (
            dataset_config.name in os.environ["CHECK_DATASET"].split(",")
            or any(
                language.code in os.environ["CHECK_DATASET"].split(",")
                for language in dataset_config.languages
            )
            or "all" in os.environ["CHECK_DATASET"].split(",")
        )
    ],
    ids=lambda dc: dc.name,
)
class TestAllDatasets:
    """Tests that are run on all datasets."""

    def test_examples_in_official_datasets_are_not_too_long(
        self,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
        tokeniser_id: str,
    ) -> None:
        """Test that the examples are not too long in official datasets."""
        dummy_model_config = LiteLLMModel.get_model_config(
            model_id="model", benchmark_config=benchmark_config
        )
        tokeniser = AutoTokenizer.from_pretrained(tokeniser_id)
        dataset = load_raw_data(
            dataset_config=dataset_config,
            cache_dir=benchmark_config.cache_dir,
            api_key=benchmark_config.api_key,
        )

        for itr_idx in range(10):
            if dataset_config.train_split is not None:
                few_shot_examples = (
                    extract_few_shot_examples(
                        dataset=dataset,
                        dataset_config=dataset_config,
                        benchmark_config=benchmark_config,
                        itr_idx=itr_idx,
                    )
                    if not dataset_config.task.requires_zero_shot
                    else []
                )
            else:
                few_shot_examples = []
            for instruction_model in [True, False]:
                prepared_test = dataset["test"].map(
                    partial(
                        apply_prompt,
                        few_shot_examples=few_shot_examples,
                        model_config=dummy_model_config,
                        dataset_config=dataset_config,
                        generative_type=(
                            GenerativeType.INSTRUCTION_TUNED
                            if instruction_model
                            else GenerativeType.BASE
                        ),
                        always_populate_text_field=True,
                        tokeniser=tokeniser,
                    ),
                    batched=True,
                    load_from_cache_file=False,
                    keep_in_memory=True,
                )

                max_input_length = max(
                    len(tokeniser(prompt)["input_ids"])  # ty: ignore[call-non-callable]
                    for prompt in prepared_test["text"]
                )
                max_output_length = dataset_config.max_generated_tokens
                max_length = max_input_length + max_output_length

                assert max_length <= MAX_CONTEXT_LENGTH, (
                    f"Max length of {max_length:,} exceeds the maximum context length "
                    f"({MAX_CONTEXT_LENGTH:,}) for dataset {dataset_config.name} in "
                    f"iteration {itr_idx} and when instruction_model="
                    f"{instruction_model}."
                )

    def test_reading_comprehension_datasets_have_id_column(
        self, dataset_config: DatasetConfig, benchmark_config: BenchmarkConfig
    ) -> None:
        """Test that reading comprehension datasets have an ID column."""
        # Skip if the dataset is not a reading comprehension dataset
        if dataset_config.task != RC:
            pytest.skip(reason="Skipping test for non-reading comprehension dataset.")

        dataset = load_raw_data(
            dataset_config=dataset_config,
            cache_dir=benchmark_config.cache_dir,
            api_key=benchmark_config.api_key,
        )

        splits = [
            split
            for split in [
                dataset_config.train_split,
                dataset_config.val_split,
                dataset_config.test_split,
            ]
            if split is not None
        ]

        for split in splits:
            assert "id" in dataset[split].features, (
                f"Dataset {dataset_config.name} is a reading comprehension dataset but "
                f"the {split} split does not have an 'id' column."
            )


class TestLoadData:
    """Tests for the `load_data` function."""

    @pytest.fixture(scope="class")
    def datasets(
        self, benchmark_config: BenchmarkConfig
    ) -> Generator[list[DatasetDict], None, None]:
        """A loaded dataset.

        Yields:
            A loaded dataset.
        """
        yield load_data(
            rng=default_rng(seed=4242),
            dataset_config=get_all_dataset_configs(
                custom_datasets_file=Path("custom_datasets.py"),
                dataset_ids=[],
                api_key=os.getenv("HF_TOKEN"),
                cache_dir=Path(".euroeval_cache"),
                trust_remote_code=True,
                run_with_cli=True,
            )["multi-wiki-qa-da"],
            benchmark_config=benchmark_config,
        )

    def test_derived_validation_split_is_deterministic_and_disjoint(
        self,
        benchmark_config: BenchmarkConfig,
        dataset_config: DatasetConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Derive a stable train holdout before bootstrapping, leaving test intact."""
        dataset_config = dataset_config.model_copy(update={"val_split": None})
        monkeypatch.setattr(
            "euroeval.data_loading.load_raw_data",
            lambda **_: DatasetDict(
                {
                    "train": Dataset.from_dict({"text": [str(i) for i in range(16)]}),
                    "test": Dataset.from_dict({"text": ["test-0", "test-1"]}),
                }
            ),
        )
        config = benchmark_config.model_copy(
            update={"num_iterations": 2, "evaluate_test_split": True}
        )

        result = load_data(
            rng=default_rng(seed=42),
            dataset_config=dataset_config,
            benchmark_config=config,
            create_validation_split=True,
        )
        second_result = load_data(
            rng=default_rng(seed=42),
            dataset_config=dataset_config,
            benchmark_config=config,
            create_validation_split=True,
        )

        for iteration in range(2):
            train_indices = set(result[iteration]["train"]["index"])
            validation_indices = set(result[iteration]["val"]["index"])
            assert train_indices.isdisjoint(validation_indices)
            assert (
                result[iteration]["val"]["index"]
                == second_result[iteration]["val"]["index"]
            )
            assert set(result[iteration]["test"]["text"]) == {"test-0", "test-1"}

    def test_derived_validation_split_rejects_tiny_training_data(
        self,
        benchmark_config: BenchmarkConfig,
        dataset_config: DatasetConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reject a one-row training split rather than creating overlapping data."""
        dataset_config = dataset_config.model_copy(update={"val_split": None})
        source_dataset = DatasetDict(
            {
                "train": Dataset.from_dict({"text": ["only"]}),
                "test": Dataset.from_dict({"text": ["test"]}),
            }
        )
        monkeypatch.setattr(
            "euroeval.data_loading.load_raw_data", lambda **_: source_dataset
        )

        with pytest.raises(InvalidBenchmark, match="At least two training examples"):
            load_data(
                rng=default_rng(seed=42),
                dataset_config=dataset_config,
                benchmark_config=benchmark_config,
                create_validation_split=True,
            )

    def test_load_data_is_list_of_dataset_dicts(
        self, datasets: list[DatasetDict]
    ) -> None:
        """Test that the `load_data` function returns a list of `DatasetDict`."""
        assert isinstance(datasets, list)
        assert all(isinstance(d, DatasetDict) for d in datasets)

    def test_no_empty_examples(self, datasets: list[DatasetDict]) -> None:
        """Test that there are no empty examples in the datasets."""
        for dataset in datasets:
            for split in dataset.values():
                for feature in ["text", "tokens"]:
                    if feature in split.features:
                        assert all(len(x) > 0 for x in split[feature])

    def test_number_of_iterations_is_correct(
        self, datasets: list[DatasetDict], benchmark_config: BenchmarkConfig
    ) -> None:
        """Test that the number of iterations is correct."""
        assert len(datasets) == benchmark_config.num_iterations

    def test_split_names_are_correct(self, datasets: list[DatasetDict]) -> None:
        """Test that the split names are correct."""
        assert all(set(d.keys()) == {"train", "val", "test"} for d in datasets)


@pytest.fixture(scope="module")
def tokeniser_id() -> Generator[str, None, None]:
    """Fixture for the tokeniser ID.

    Yields:
        A tokeniser ID.
    """
    yield "EuroEval/gemma-3-tokenizer"
