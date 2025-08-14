"""A custom generative model currently being trained."""

import collections.abc as c
import logging
import typing as t
from functools import cached_property, partial

import numpy as np
import xgrammar as xgr
from datasets import DatasetDict
from pydantic import conlist, create_model
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    BatchEncoding,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedModel,
    Trainer,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.tokenization_utils import PreTrainedTokenizer

from ..benchmark_modules.base import BenchmarkModule
from ..constants import (
    GENERATIVE_PIPELINE_TAGS,
    MAX_CONTEXT_LENGTH,
    POTENTIAL_MAX_MODEL_LENGTH_CONFIG_NAMES,
    TASKS_USING_JSON,
)
from ..data_models import BenchmarkConfig, GenerativeModelOutput, ModelConfig, Task
from ..enums import (
    BatchingPreference,
    GenerativeType,
    InferenceBackend,
    ModelType,
    TaskGroup,
)
from ..exceptions import NeedsEnvironmentVariable, NeedsExtraInstalled
from ..generation_utils import apply_prompt, extract_few_shot_examples
from ..languages import get_all_languages
from ..task_group_utils import (
    question_answering,
    sequence_classification,
    text_to_text,
    token_classification,
)
from ..types import ExtractLabelsFunction
from ..utils import create_model_cache_dir

logger = logging.getLogger("euroeval")


class CustomGenerativeModel(BenchmarkModule):
    """A custom generative model that is currently being trained."""

    fresh_model = False
    batching_preference = BatchingPreference.NO_PREFERENCE
    high_priority = True
    model_id = "custom"

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
        """Initialise the custom model.

        Args:
            model:
                The model to use for the custom model.
            tokenizer:
                The tokenizer to use for the custom model.
        """
        self._model = model
        self._tokenizer = tokenizer

    @cached_property
    def num_params(self) -> int:
        """Get the number of parameters in the model.

        Returns:
            The number of parameters in the model.
        """
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    # TODO: We currently do not support reasoning models here
    @property
    def generative_type(self) -> "GenerativeType | None":
        """Get the generative type of the model.

        Returns:
            The generative type of the model.
        """
        if self._tokenizer.chat_template is None:
            return GenerativeType.BASE
        return GenerativeType.INSTRUCTION_TUNED

    @cached_property
    def vocab_size(self) -> int:
        """Get the vocabulary size of the model.

        Returns:
            The vocabulary size of the model.
        """
        return len(self._tokenizer.get_vocab())

    @cached_property
    def model_max_length(self) -> int:
        """Get the maximum length of the model.

        Returns:
            The maximum length of the model.
        """
        candidate_max_lengths = [
            value
            for key in POTENTIAL_MAX_MODEL_LENGTH_CONFIG_NAMES
            if (value := getattr(self._tokenizer, key)) is not None
        ]
        if not candidate_max_lengths:
            return MAX_CONTEXT_LENGTH
        return min(candidate_max_lengths)

    @property
    def data_collator(self) -> c.Callable[[list[t.Any]], dict[str, t.Any]]:
        """The data collator used to prepare samples during finetuning.

        Returns:
            The data collator.
        """
        match self.dataset_config.task.task_group:
            case (
                TaskGroup.SEQUENCE_CLASSIFICATION
                | TaskGroup.TEXT_TO_TEXT
                | TaskGroup.QUESTION_ANSWERING
                | TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
            ):
                return DataCollatorWithPadding(self._tokenizer, padding="longest")
            case TaskGroup.TOKEN_CLASSIFICATION:
                return DataCollatorForTokenClassification(
                    tokenizer=self._tokenizer, label_pad_token_id=-100
                )
            case _:
                raise NotImplementedError(
                    f"Unsupported task group: {self.dataset_config.task.task_group}."
                )

    @property
    def extract_labels_from_generation(self) -> ExtractLabelsFunction:
        """The function used to extract the labels from the generated output.

        Returns:
            The function used to extract the labels from the generated output.
        """
        match self.dataset_config.task.task_group:
            case (
                TaskGroup.SEQUENCE_CLASSIFICATION
                | TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
            ):
                return partial(
                    sequence_classification.extract_labels_from_generation,
                    dataset_config=self.dataset_config,
                    first_label_token_mapping=self.buffer["first_label_token_mapping"],
                )
            case TaskGroup.TEXT_TO_TEXT:
                return text_to_text.extract_labels_from_generation
            case TaskGroup.TOKEN_CLASSIFICATION:
                return partial(
                    token_classification.extract_labels_from_generation,
                    dataset_config=self.dataset_config,
                )
            case TaskGroup.QUESTION_ANSWERING:
                return question_answering.extract_labels_from_generation
            case _:
                raise NotImplementedError(
                    f"Unsupported task group: {self.dataset_config.task.task_group}."
                )

    @property
    def trainer_class(self) -> t.Type["Trainer"]:
        """The trainer class used to train the model.

        Returns:
            The trainer class.
        """
        return Trainer

    def prepare_dataset(
        self, dataset: DatasetDict, task: "Task", itr_idx: int
    ) -> DatasetDict:
        """Prepare the dataset for evaluation.

        Args:
            dataset:
                The dataset to prepare.
            task:
                The task to prepare the dataset for.
            itr_idx:
                The iteration index.

        Returns:
            The prepared dataset.
        """
        if task.task_group == TaskGroup.QUESTION_ANSWERING:
            dataset = dataset.map(
                lambda examples: dict(
                    label=[
                        dict(
                            id=id,
                            answers=dict(
                                answer_start=answer_dct["answer_start"],
                                text=[
                                    answer_text.lower()
                                    for answer_text in answer_dct["text"]
                                ],
                            ),
                        )
                        for id, answer_dct in zip(examples["id"], examples["answers"])
                    ]
                ),
                batched=True,
                load_from_cache_file=False,
                keep_in_memory=True,
            )

        if self.benchmark_config.few_shot:
            few_shot_examples = extract_few_shot_examples(
                dataset=dataset, dataset_config=self.dataset_config, itr_idx=itr_idx
            )
        else:
            few_shot_examples = list()

        dataset["test"] = dataset["test"].map(
            partial(
                apply_prompt,
                few_shot_examples=few_shot_examples,
                model_config=self.model_config,
                dataset_config=self.dataset_config,
                instruction_model=self.buffer["instruction_model"],
                always_populate_text_field=True,
                tokenizer=self._tokenizer,
            ),
            batched=True,
            load_from_cache_file=False,
            keep_in_memory=True,
        )

        return dataset

    def generate(self, inputs: dict) -> "GenerativeModelOutput":
        """Generate output from the model.

        Args:
            inputs:
                The inputs to the model.

        Returns:
            The output of the model.
        """
        # Prepare the model inputs
        model_inputs = self._tokenizer.apply_chat_template(
            conversation=inputs["messages"],
            return_tensors="pt",
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
        )
        assert isinstance(model_inputs, BatchEncoding), (
            "The model inputs should be a BatchEncoding object, "
            "but got: {model_inputs!r}."
        )
        model_inputs = model_inputs.to(self._model.device)

        # Create a logits processor if we're doing structured generation
        logits_processor = None
        if self.dataset_config.task in TASKS_USING_JSON:
            ner_tag_names = list(self.dataset_config.prompt_label_mapping.values())
            keys_and_their_types: dict[str, t.Any] = {
                tag_name: (conlist(str, max_length=5), ...)
                for tag_name in ner_tag_names
            }
            answer_format_class = create_model("AnswerFormat", **keys_and_their_types)
            json_schema_string = answer_format_class.model_json_schema()

            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self._tokenizer, vocab_size=self.vocab_size
            )
            grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
            compiled_grammar = grammar_compiler.compile_json_schema(json_schema_string)
            logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)

        # Create the generation configuration, containing the parameters controlling the
        # text generation
        generation_config = GenerationConfig(
            max_new_tokens=self.dataset_config.max_generated_tokens,
            temperature=0.0,
            output_scores=True,
        )

        # Generate the outputs
        outputs = self._model.generate(
            **model_inputs,
            generation_config=generation_config,
            logits_processor=[logits_processor],
        )
        assert isinstance(outputs, GenerateDecoderOnlyOutput), (
            "The outputs of the model should be a GenerateDecoderOnlyOutput object, "
            f"but got: {outputs!r}."
        )

        # Extract the scores from the outputs
        raw_scores = outputs.scores
        scores = None
        if raw_scores is not None:
            # Convert the raw scores to a numpy array
            # list[(batch_size, vocab_size)] -> (batch_size, num_tokens, vocab_size)
            scores_arr = pad_sequence(
                sequences=list(raw_scores), batch_first=False
            ).numpy()

            # Add a last dimension of size 2, containing the token and the logprob
            # (batch_size, num_tokens, vocab_size)
            # -> (batch_size, num_tokens, vocab_size, 2)
            scores_arr = scores_arr[..., np.newaxis].repeat(repeats=2, axis=-1)
            scores_arr[..., 0] = [
                token
                for token, _ in sorted(
                    self._tokenizer.get_vocab().items(), key=lambda x: x[1]
                )
            ]

            # Convert the array to a nested list
            scores = scores_arr.tolist()

        # Convert the outputs to a GenerativeModelOutput object and return it
        return GenerativeModelOutput(
            sequences=self._tokenizer.batch_decode(sequences=outputs.sequences),
            scores=scores,
        )

    @classmethod
    def model_exists(
        cls, model_id: str, benchmark_config: "BenchmarkConfig"
    ) -> bool | NeedsExtraInstalled | NeedsEnvironmentVariable:
        """Check if the model exists.

        Args:
            model_id:
                The ID of the model to check.
            benchmark_config:
                The benchmark configuration.

        Returns:
            True if the model exists, False otherwise. If the model requires an extra
            to be installed or an environment variable to be set, returns the
            appropriate exception.
        """
        # return model_id == cls.model_id
        return False

    @classmethod
    def get_model_config(
        cls, model_id: str, benchmark_config: "BenchmarkConfig"
    ) -> "ModelConfig":
        """Get the model configuration.

        Args:
            model_id:
                The ID of the model.
            benchmark_config:
                The benchmark configuration.

        Returns:
            The model configuration.
        """
        return ModelConfig(
            model_id=model_id,
            revision="main",
            task=GENERATIVE_PIPELINE_TAGS[0],
            languages=list(get_all_languages().values()),
            inference_backend=InferenceBackend.TRANSFORMERS,
            merge=False,
            model_type=ModelType.GENERATIVE,
            fresh=False,
            model_cache_dir=create_model_cache_dir(
                cache_dir=benchmark_config.cache_dir, model_id=model_id
            ),
            adapter_base_model_id=None,
        )
