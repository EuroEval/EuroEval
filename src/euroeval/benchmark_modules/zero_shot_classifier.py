"""A benchmark module wrapping non-generative zero-shot "decision" models."""

import logging
import math
import typing as t
from functools import cached_property

from ..data_models import (
    BenchmarkConfig,
    DatasetConfig,
    GenerativeModelOutput,
    ModelConfig,
    Task,
)
from ..enums import (
    BatchingPreference,
    GenerativeType,
    InferenceBackend,
    ModelType,
    TaskGroup,
)
from ..exceptions import InvalidBenchmark, InvalidModel, NeedsExtraInstalled
from ..logging_utils import log_once
from ..model_cache import create_model_cache_dir
from ..string_utils import split_model_id
from ..task_group_utils.cloze import parse_bare_question_and_choices_with_markers
from ..tokenisation_utils import get_first_label_token_mapping
from ..types import ExtractLabelsFunction
from ..zero_shot_adapters import ZeroShotClassifierAdapter, get_adapter
from .base import (
    PRIORITY_ZERO_SHOT_CLASSIFIER,
    BenchmarkModule,
    NonFinetunableModuleMixin,
    _extract_labels_from_generation_helper,
    _prepare_dataset_helper,
)

if t.TYPE_CHECKING:
    from datasets import DatasetDict

# A probability floor used when taking the log of an adapter-reported probability,
# to avoid `math.log(0.0)` raising `ValueError` for a label the adapter assigned zero
# probability.
_MIN_PROBABILITY = 1e-12


class ZeroShotClassifierModel(NonFinetunableModuleMixin, BenchmarkModule):
    """A benchmark module wrapping a non-generative zero-shot "decision" model.

    These models (e.g. Laya) are encoders with trained heads that answer typed
    questions with calibrated probabilities. They are loaded through their own
    package, via one of the adapters in `euroeval.zero_shot_adapters.ADAPTERS`, and
    are evaluated zero-shot only: no finetuning, no few-shot demonstrations.
    """

    fresh_model = False
    batching_preference = BatchingPreference.ALL_AT_ONCE

    # Checked before `FreshEncoderModel`/`HuggingFaceEncoderModel`/`VLLMModel` (see
    # the priority table in `benchmark_modules.base`), since a matching zero-shot
    # classifier repo (e.g. Laya) would otherwise be misidentified as a plain
    # encoder or generative model.
    priority = PRIORITY_ZERO_SHOT_CLASSIFIER

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
        log_metadata: bool = True,
    ) -> None:
        """Initialise the model.

        Args:
            model_config:
                The model configuration.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.
            log_metadata:
                Whether to log the model metadata.

        Raises:
            InvalidModel:
                If no adapter matches the model.
        """
        adapter_cls = get_adapter(
            model_id=model_config.model_id, benchmark_config=benchmark_config
        )
        if adapter_cls is None:
            raise InvalidModel(
                f"No zero-shot classifier adapter matches the model "
                f"{model_config.model_id!r}."
            )
        self.adapter: ZeroShotClassifierAdapter = adapter_cls(
            model_config=model_config, benchmark_config=benchmark_config
        )

        super().__init__(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=log_metadata,
        )
        self.buffer["first_label_token_mapping"] = get_first_label_token_mapping(
            dataset_config=self.dataset_config,
            model_config=self.model_config,
            tokeniser=None,
            generative_type=self.generative_type,
            log_metadata=self.log_metadata,
        )
        self.buffer["instructions"] = self._build_instructions()

    def _build_instructions(self) -> str:
        """Build the classification instructions from the dataset's templates.

        Returns:
            The instructions describing the classification task, with the
            `{text}` placeholder (which is filled in per-sample by the `texts`
            argument to `adapter.classify`) removed.
        """
        return self.dataset_config.instruction_prompt.format(
            text="", labels_str=self.dataset_config.get_labels_str()
        ).strip()

    @property
    def extract_labels_from_generation(self) -> ExtractLabelsFunction:
        """The function used to extract the labels from the generated output.

        Returns:
            The function used to extract the labels from the generated output.
        """
        return _extract_labels_from_generation_helper(
            dataset_config=self.dataset_config,
            model_config=self.model_config,
            first_label_token_mapping=self.buffer["first_label_token_mapping"],
        )

    def generate(self, inputs: dict) -> GenerativeModelOutput:
        """Generate outputs from the model.

        Args:
            inputs:
                A batch of inputs to pass through the model.

        Returns:
            The generated model outputs.

        Raises:
            InvalidBenchmark:
                If the inputs do not contain a 'text' key, or if the dataset's task
                group is not supported.
        """
        if self.dataset_config.task.task_group not in (
            TaskGroup.SEQUENCE_CLASSIFICATION,
            TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
        ):
            raise InvalidBenchmark(
                "Zero-shot classifier models only support sequence classification "
                "and multiple-choice classification tasks, but the task group of "
                f"the dataset {self.dataset_config.name!r} is "
                f"{self.dataset_config.task.task_group!r}."
            )

        if "text" not in inputs:
            raise InvalidBenchmark("The inputs must contain a 'text' key.")

        texts = list(inputs["text"])

        candidate_labels = [
            self.dataset_config.prompt_label_mapping[label]
            for label in self.dataset_config.id2label.values()
        ]
        if not candidate_labels:
            raise InvalidBenchmark(
                "No candidate labels found for this dataset. Set "
                "DatasetConfig.labels/prompt_label_mapping for classification tasks "
                "before using a zero-shot classifier model."
            )

        task_group = self.dataset_config.task.task_group
        if task_group == TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION:
            label_probs = self._classify_multiple_choice(
                texts=texts, letter_labels=candidate_labels
            )
        else:
            label_probs = self.adapter.classify(
                texts=texts,
                candidate_labels=candidate_labels,
                instructions=self.buffer["instructions"],
            )

        sequences: list[str] = []
        scores: list[list[list[tuple[str, float]]]] = []
        for probs in label_probs:
            sample_scores: list[tuple[str, float]] = [
                (label, math.log(max(probs.get(label, 0.0), _MIN_PROBABILITY)))
                for label in candidate_labels
            ]
            # `sequence_classification.get_closest_logprobs_labels` (the generic
            # label extractor used for all generative-style `scores` output, e.g.
            # vLLM/LiteLLM top-logprobs) sorts each sample's list by logprob
            # descending itself, so we don't need to pre-sort here.
            best_label = max(
                sample_scores, key=lambda label_and_logprob: label_and_logprob[1]
            )[0]
            sequences.append(best_label)
            scores.append([sample_scores])

        return GenerativeModelOutput(sequences=sequences, scores=scores)

    def _classify_multiple_choice(
        self, texts: list[str], letter_labels: list[str]
    ) -> list[dict[str, float]]:
        """Classify multiple-choice samples, using the option texts as labels.

        The adapter is a text classifier, not an MCQ solver: passing it the bare
        letter labels ("a", "b", ...) gives it nothing to compare the sample text
        against. Instead, each sample's actual option texts (parsed back out of the
        formatted 'text' column, which stores the question followed by its
        enumerated options) are used as the candidate labels, and the returned
        probabilities are mapped back to the letter labels afterwards, so label
        extraction and metrics -- which are keyed by letter -- are unaffected.

        Args:
            texts:
                The formatted multiple-choice prompts (question plus options).
            letter_labels:
                The letter labels ("a", "b", ...), in order, that the parsed options
                are expected to map onto.

        Returns:
            One letter-label-keyed probability dictionary per text.
        """
        label_probs: list[dict[str, float]] = []
        for text in texts:
            _, option_texts, markers = parse_bare_question_and_choices_with_markers(
                text
            )

            fallback_reason: str | None = None
            if len(option_texts) != len(letter_labels):
                fallback_reason = (
                    f"the sample has {len(option_texts)} parsed option(s), but "
                    f"{len(letter_labels)} letter label(s) were expected"
                )
            elif markers != letter_labels:
                fallback_reason = (
                    f"the parsed option markers {markers!r} do not match the "
                    f"expected letter labels {letter_labels!r} in order"
                )
            elif len(set(option_texts)) != len(option_texts):
                fallback_reason = (
                    f"the parsed option texts {option_texts!r} contain duplicates, "
                    "which would make two letters share the same probability"
                )

            if fallback_reason is not None:
                # Couldn't reliably use this sample's parsed options -- fall back to
                # classifying against the letter labels directly, rather than
                # dropping the sample or silently mismapping probabilities.
                log_once(
                    "Falling back to classifying multiple-choice options by their "
                    f"letter labels for dataset {self.dataset_config.name!r}, "
                    f"since {fallback_reason}.",
                    level=logging.WARNING,
                )
                option_texts = letter_labels

            sample_probs = self.adapter.classify(
                texts=[text],
                candidate_labels=option_texts,
                instructions=self.buffer["instructions"],
            )[0]

            if option_texts is letter_labels:
                label_probs.append(sample_probs)
            else:
                label_probs.append(
                    {
                        letter: sample_probs.get(option_text, 0.0)
                        for letter, option_text in zip(letter_labels, option_texts)
                    }
                )
        return label_probs

    @property
    def generative_type(self) -> GenerativeType | None:
        """The generative type of the model.

        Zero-shot classifier models are not generative, so this is always None.

        Returns:
            The generative type of the model.
        """
        return None

    @classmethod
    def get_model_config(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> ModelConfig:
        """Fetch the model configuration.

        Args:
            model_id:
                The model ID.
            benchmark_config:
                The benchmark configuration.

        Returns:
            The model configuration.

        Raises:
            InvalidModel:
                If no adapter matches the model, or the given parameter is not
                allowed for the matching adapter.
        """
        model_id_components = split_model_id(model_id=model_id)
        adapter_cls = get_adapter(
            model_id=model_id_components.model_id, benchmark_config=benchmark_config
        )
        if adapter_cls is None:
            raise InvalidModel(
                f"No zero-shot classifier adapter matches the model {model_id!r}."
            )

        param = model_id_components.param
        if param is not None and param not in adapter_cls.variants:
            msg = f"Invalid parameter {param!r} for model {model_id!r}."
            if adapter_cls.variants:
                msg += f" Allowed parameters are: {', '.join(adapter_cls.variants)}."
            else:
                msg += " No parameters are allowed."
            raise InvalidModel(msg)

        return ModelConfig(
            model_id=model_id_components.model_id,
            revision=model_id_components.revision,
            param=param,
            task="text-classification",
            languages=list(),
            merge=False,
            inference_backend=InferenceBackend.ZERO_SHOT_CLASSIFIER,
            model_type=ModelType.ZERO_SHOT_CLASSIFIER,
            fresh=False,
            model_cache_dir=create_model_cache_dir(
                cache_dir=benchmark_config.cache_dir,
                model_id=model_id_components.model_id,
            ),
            adapter_base_model_id=None,
        )

    @classmethod
    def model_exists(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> bool | NeedsExtraInstalled:
        """Check if a model exists.

        Args:
            model_id:
                The model ID.
            benchmark_config:
                The benchmark configuration.

        Returns:
            Whether the model exists.
        """
        model_id_components = split_model_id(model_id=model_id)
        try:
            adapter_cls = get_adapter(
                model_id=model_id_components.model_id, benchmark_config=benchmark_config
            )
        except NeedsExtraInstalled as error:
            return error
        return adapter_cls is not None

    @cached_property
    def model_max_length(self) -> int:
        """The maximum length of the model.

        Returns:
            The maximum length of the model.
        """
        return self.adapter.max_length

    @cached_property
    def num_params(self) -> int:
        """The number of parameters in the model.

        Returns:
            The number of parameters in the model.
        """
        if self.benchmark_config.num_parameters is not None:
            return self.benchmark_config.num_parameters
        adapter_cls = type(self.adapter)
        return adapter_cls.num_params(
            model_id=self.model_config.model_id,
            param=self.model_config.param,
            api_key=self.benchmark_config.api_key,
        )

    def prepare_dataset(
        self, dataset: "DatasetDict", task: Task, itr_idx: int
    ) -> "DatasetDict":
        """Prepare the dataset for the model.

        Args:
            dataset:
                The dataset to prepare.
            task:
                The task to prepare the dataset for.
            itr_idx:
                The index of the dataset in the iterator.

        Returns:
            The prepared dataset.
        """
        # Zero-shot classifier adapters (e.g. Laya) build their own instructions
        # from `_build_instructions` and expect the raw sample text -- for
        # multiple-choice tasks, the bare question plus its options, as it appears
        # in the dataset's 'text' column -- rather than the decoder prompt
        # template `_prepare_dataset_helper` would otherwise render into 'text'.
        return _prepare_dataset_helper(
            dataset=dataset,
            task=task,
            model_config=self.model_config,
            dataset_config=self.dataset_config,
            benchmark_config=self.benchmark_config,
            generative_type=self.generative_type,
            itr_idx=itr_idx,
            always_populate_text_field=False,
            tokeniser=None,
            preserve_raw_text=True,
        )

    def update_dataset_config(self, dataset_config: "DatasetConfig") -> t.Self:
        """Update the dataset config registered in the benchmark module.

        The model is reused across datasets (see `Benchmarker`), so per-dataset
        state derived from `dataset_config` -- here, `first_label_token_mapping`
        and `instructions`, both computed once in `__init__` -- must be
        recomputed for the new dataset, the same way `VLLMModel.
        update_dataset_config` does.

        Args:
            dataset_config:
                The new dataset config.

        Returns:
            The benchmark module.
        """
        self.dataset_config = dataset_config
        self.buffer["first_label_token_mapping"] = get_first_label_token_mapping(
            dataset_config=self.dataset_config,
            model_config=self.model_config,
            tokeniser=None,
            generative_type=self.generative_type,
            log_metadata=self.log_metadata,
        )
        self.buffer["instructions"] = self._build_instructions()
        return self

    @cached_property
    def vocab_size(self) -> int:
        """The vocabulary size of the model.

        Returns:
            The vocabulary size of the model.
        """
        return -1
