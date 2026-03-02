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
from euroeval.languages import DANISH
from euroeval.tasks import NER, RC, SENT, SUMM


@pytest.fixture(scope="module")
def tokeniser_id() -> Generator[str, None, None]:
    """Fixture for the tokeniser ID.

    Yields:
        A tokeniser ID.
    """
    yield "EuroEval/gemma-3-tokenizer"


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
            if "train" in dataset_config.splits:
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
                    len(tokeniser(prompt)["input_ids"])
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
        for split in dataset_config.splits:
            assert "id" in dataset[split].features, (
                f"Dataset {dataset_config.name} is a reading comprehension dataset but "
                f"the {split} split does not have an 'id' column."
            )


def make_dataset(col: str, value: str = "positive") -> DatasetDict:
    """Build a minimal three-split DatasetDict with a 'text' column and a custom column.

    Args:
        col:
            The name of the custom column to add.
        value:
            The value to use in the custom column. Defaults to "positive".

    Returns:
        A DatasetDict with "train", "val", and "test" splits, each containing a single
        row with a "text" column and the specified custom column.
    """
    split = Dataset.from_dict({"text": ["hello"], col: [value]})
    return DatasetDict({"train": split, "val": split, "test": split})


class TestPreprocessingFunc:
    """Tests for the preprocessing function built from column arguments."""

    def _config(
        self,
        task,
        target_column: str | None = None,
        input_column: str = "text",
        choices_column: str | None = None,
    ) -> DatasetConfig:
        return DatasetConfig(
            name="test-dataset",
            pretty_name="Test Dataset",
            source="dummy/source",
            task=task,
            languages=[DANISH],
            target_column=target_column,
            input_column=input_column,
            choices_column=choices_column,
        )

    def test_sequence_classification_renames_to_label(self) -> None:
        """target_column is renamed to 'label' for sequence classification tasks."""
        raw = make_dataset(col="sentiment")
        config = self._config(SENT, target_column="sentiment")
        assert config.preprocessing_func is not None
        result = config.preprocessing_func(raw)
        assert "label" in result["test"].column_names
        assert "sentiment" not in result["test"].column_names

    def test_token_classification_renames_to_labels(self) -> None:
        """target_column is renamed to 'labels' for token classification tasks."""
        raw = make_dataset(col="ner_tags", value="O")
        config = self._config(NER, target_column="ner_tags")
        assert config.preprocessing_func is not None
        result = config.preprocessing_func(raw)
        assert "labels" in result["test"].column_names
        assert "ner_tags" not in result["test"].column_names

    def test_text_to_text_renames_to_target_text(self) -> None:
        """target_column is renamed to 'target_text' for text-to-text tasks."""
        raw = make_dataset(col="summary")
        config = self._config(SUMM, target_column="summary")
        assert config.preprocessing_func is not None
        result = config.preprocessing_func(raw)
        assert "target_text" in result["test"].column_names
        assert "summary" not in result["test"].column_names

    def test_conflict_existing_target_column_is_replaced(self) -> None:
        """When target column already exists, it is removed before renaming."""
        split = Dataset.from_dict(
            {"text": ["hello"], "sentiment": ["positive"], "label": ["negative"]}
        )
        raw = DatasetDict({"train": split, "val": split, "test": split})
        config = self._config(SENT, target_column="sentiment")
        assert config.preprocessing_func is not None
        result = config.preprocessing_func(raw)
        assert "label" in result["test"].column_names
        assert result["test"]["label"] == ["positive"]
        assert "sentiment" not in result["test"].column_names

    def test_missing_target_column_raises_invalid_benchmark(self) -> None:
        """Raises InvalidBenchmark when the configured target_column is absent."""
        raw = make_dataset(col="label")  # does NOT have "sentiment" column
        config = self._config(SENT, target_column="sentiment")
        assert config.preprocessing_func is not None
        with pytest.raises(InvalidBenchmark, match="sentiment"):
            config.preprocessing_func(raw)

    def test_input_column_renamed_to_text(self) -> None:
        """input_column is renamed to 'text'."""
        split = Dataset.from_dict({"question": ["what?"], "label": ["a"]})
        raw = DatasetDict({"train": split, "val": split, "test": split})
        config = self._config(SENT, input_column="question")
        assert config.preprocessing_func is not None
        result = config.preprocessing_func(raw)
        assert "text" in result["test"].column_names
        assert "question" not in result["test"].column_names

    def test_choices_column_merged_with_input_column(self) -> None:
        """choices_column is merged with input_column into 'text'."""
        split = Dataset.from_dict(
            {
                "question": ["What is 1+1?"],
                "choices": [["1", "2", "3", "4"]],
                "label": ["b"],
            }
        )
        raw = DatasetDict({"train": split, "val": split, "test": split})
        config = self._config(SENT, input_column="question", choices_column="choices")
        assert config.preprocessing_func is not None
        result = config.preprocessing_func(raw)
        assert "text" in result["test"].column_names
        assert "question" not in result["test"].column_names
        assert "choices" not in result["test"].column_names
        text = result["test"]["text"][0]
        assert "What is 1+1?" in text
        assert "a. 1" in text
        assert "b. 2" in text
