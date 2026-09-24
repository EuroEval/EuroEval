"""A benchmark module wrapping the Laya zero-shot classifier model."""

import collections.abc as c
import math
import typing as t
from functools import cached_property
from pathlib import Path

from huggingface_hub.errors import NotASafetensorsRepoError

from ..constants import (
    LAYA_BUNDLED_REPO_ID,
    LAYA_CHECKPOINTS,
    LAYA_DEFAULT_MAX_LENGTH,
    LAYA_MIN_PROBABILITY,
)
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
from ..model_cache import create_model_cache_dir
from ..safetensors_utils import get_num_params_from_safetensors_metadata
from ..string_utils import split_model_id
from ..task_group_utils.cloze import parse_bare_question_and_choices
from ..tokenisation_utils import get_first_label_token_mapping
from ..types import ExtractLabelsFunction
from ..utils import get_hf_token
from .base import (
    BenchmarkModule,
    _extract_labels_from_generation_helper,
    _prepare_dataset_helper,
)

if t.TYPE_CHECKING:
    from datasets import DatasetDict
    from transformers.trainer import Trainer


class ZeroShotClassifierModel(BenchmarkModule):
    """Laya, a non-generative zero-shot "decision" model.

    Laya (https://pypi.org/project/laya/) is an encoder with trained decision heads,
    loaded through its own `laya` package rather than `transformers`. It answers
    typed `choice` questions with calibrated per-label probabilities, and is
    evaluated zero-shot only: no finetuning, no few-shot demonstrations.
    """

    fresh_model = False
    batching_preference = BatchingPreference.ALL_AT_ONCE

    # Checked before the generic encoder/generative modules, so that a Laya repo
    # isn't misidentified as a plain encoder.
    high_priority = True

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
                If a revision other than "main" is requested, since the `laya`
                package cannot load a specific revision.
        """
        import laya  # noqa: PLC0415

        model_id = model_config.model_id
        param = model_config.param
        revision = model_config.revision
        if revision not in ("main", ""):
            raise InvalidModel(
                f"The model {model_id!r} was requested at revision {revision!r}, "
                "but the `laya` package does not support loading a specific "
                "revision -- it always loads the repo's default branch ('main')."
            )

        subfolder = _resolve_subfolder(model_id=model_id, param=param)
        checkpoint_name = _checkpoint_name(model_id=model_id, param=param)
        token = get_hf_token(api_key=benchmark_config.api_key)

        self.agent = laya.Agent(
            model_id_or_path=model_id, subfolder=subfolder, token=token
        )
        self.max_length = LAYA_CHECKPOINTS.get(checkpoint_name, LAYA_DEFAULT_MAX_LENGTH)

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
            The instructions describing the classification task.
        """
        return self.dataset_config.instruction_prompt.format(
            text="", labels_str=self.dataset_config.get_labels_str()
        ).strip()

    @property
    def data_collator(self) -> c.Callable[[list[dict[str, t.Any]]], dict[str, t.Any]]:
        """The data collator used to prepare samples during finetuning.

        Raises:
            NotImplementedError:
                Always; Laya is not finetuned.
        """
        raise NotImplementedError(
            "The `data_collator` property has not been implemented for Laya, as "
            "it is not finetuned."
        )

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
                "Laya only supports sequence classification and multiple-choice "
                f"classification tasks, but the task group of the dataset "
                f"{self.dataset_config.name!r} is "
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
                "DatasetConfig.labels/prompt_label_mapping for classification "
                "tasks before using Laya."
            )

        task_group = self.dataset_config.task.task_group
        if task_group == TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION:
            label_probs = self._classify_multiple_choice(
                texts=texts, letter_labels=candidate_labels
            )
        else:
            label_probs = self._classify(texts=texts, candidate_labels=candidate_labels)

        sequences: list[str] = []
        scores: list[list[list[tuple[str, float]]]] = []
        for probs in label_probs:
            sample_scores = sorted(
                (
                    (label, math.log(max(probs.get(label, 0.0), LAYA_MIN_PROBABILITY)))
                    for label in candidate_labels
                ),
                key=lambda pair: pair[1],
                reverse=True,
            )
            sequences.append(sample_scores[0][0])
            scores.append([sample_scores])

        return GenerativeModelOutput(sequences=sequences, scores=scores)

    def _classify(
        self, texts: list[str], candidate_labels: list[str]
    ) -> list[dict[str, float]]:
        """Classify each text against the candidate labels, using Laya.

        `laya.Agent.system_one` only batches multiple questions for a single text,
        not multiple texts in one call, so each text still needs its own call.

        Args:
            texts:
                The texts to classify.
            candidate_labels:
                The candidate labels to classify each text into.

        Returns:
            A list, with one dictionary per text, mapping each candidate label to
            its predicted probability.
        """
        return [
            self._classify_one(text=text, candidate_labels=candidate_labels)
            for text in texts
        ]

    def _classify_one(self, text: str, candidate_labels: list[str]) -> dict[str, float]:
        """Classify a single text against the candidate labels, using Laya.

        Args:
            text:
                The text to classify.
            candidate_labels:
                The candidate labels to classify the text into.

        Returns:
            A dictionary mapping each candidate label to its predicted probability.

        Raises:
            InvalidBenchmark:
                If Laya's response is missing a probability for one of the
                candidate labels.
        """
        question = {
            "type": "choice",
            "instructions": self.buffer["instructions"],
            "criteria": {label: None for label in candidate_labels},
        }
        output = self.agent.system_one(state=text, questions={"q": question})
        probabilities = output["answers"]["q"]["probabilities"]
        missing_labels = [
            label for label in candidate_labels if label not in probabilities
        ]
        if missing_labels:
            raise InvalidBenchmark(
                "Laya did not return a probability for the candidate "
                f"label(s) {missing_labels!r}."
            )
        return {label: float(probabilities[label]) for label in candidate_labels}

    def _classify_multiple_choice(
        self, texts: list[str], letter_labels: list[str]
    ) -> list[dict[str, float]]:
        """Classify multiple-choice samples, using the option texts as labels.

        Args:
            texts:
                The formatted multiple-choice prompts (question plus options).
            letter_labels:
                The letter labels ("a", "b", ...), in order.

        Returns:
            One letter-label-keyed probability dictionary per text.
        """
        label_probs: list[dict[str, float]] = []
        for text in texts:
            _, option_texts = parse_bare_question_and_choices(text=text)
            unparseable = len(option_texts) != len(letter_labels) or len(
                set(option_texts)
            ) != len(option_texts)
            used_letter_fallback = unparseable
            if used_letter_fallback:
                # Couldn't reliably parse this sample's options (or two options share
                # the same text) -- fall back to classifying against the letters.
                option_texts = letter_labels

            sample_probs = self._classify_one(text=text, candidate_labels=option_texts)
            if used_letter_fallback:
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

        Returns:
            None, since Laya is not generative.
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
                If the given parameter is not allowed for the model.
        """
        model_id_components = split_model_id(model_id=model_id)
        param = model_id_components.param
        variants = [name for name in LAYA_CHECKPOINTS if name]
        if param is not None and param not in variants:
            raise InvalidModel(
                f"Invalid parameter {param!r} for model {model_id!r}. Allowed "
                f"parameters are: {', '.join(variants)}."
            )

        return ModelConfig(
            model_id=model_id_components.model_id,
            revision=model_id_components.revision,
            param=param,
            task="text-classification",
            languages=list(),
            merge=False,
            inference_backend=InferenceBackend.LAYA,
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
        bare_model_id = model_id_components.model_id

        is_known_hub_repo = bare_model_id == LAYA_BUNDLED_REPO_ID or (
            bare_model_id.startswith(f"{LAYA_BUNDLED_REPO_ID}-")
        )
        is_local_checkpoint_dir = (
            Path(bare_model_id).is_dir()
            and (Path(bare_model_id) / "rl_agent_config.json").is_file()
        )
        if not (is_known_hub_repo or is_local_checkpoint_dir):
            return False

        try:
            import laya  # noqa: F401,PLC0415
        except ImportError:
            return NeedsExtraInstalled(extra="laya")
        return True

    @cached_property
    def model_max_length(self) -> int:
        """The maximum length of the model.

        Returns:
            The maximum length of the model.
        """
        return self.max_length

    @cached_property
    def num_params(self) -> int:
        """The number of parameters in the model.

        Returns:
            The number of parameters in the model.
        """
        if self.benchmark_config.num_parameters is not None:
            return self.benchmark_config.num_parameters

        model_id = self.model_config.model_id
        if Path(model_id).is_dir():
            return _num_params_from_local_checkpoint(checkpoint_dir=Path(model_id))

        try:
            num_params = get_num_params_from_safetensors_metadata(
                model_id=model_id,
                revision="main",
                api_key=get_hf_token(api_key=self.benchmark_config.api_key),
            )
        except (NotASafetensorsRepoError, OSError):
            return -1
        return num_params if num_params is not None else -1

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
        # Laya builds its own instructions and expects the raw sample text, so the
        # rendered decoder prompt that `_prepare_dataset_helper` writes to 'text' is
        # swapped back out for the original text afterwards.
        raw_text = list(dataset["test"]["text"])
        prepared = _prepare_dataset_helper(
            dataset=dataset,
            task=task,
            model_config=self.model_config,
            dataset_config=self.dataset_config,
            benchmark_config=self.benchmark_config,
            generative_type=self.generative_type,
            itr_idx=itr_idx,
            always_populate_text_field=False,
            tokeniser=None,
        )
        prepared["test"] = (
            prepared["test"].remove_columns("text").add_column("text", raw_text)
        )
        return prepared

    @property
    def trainer_class(self) -> t.Type["Trainer"]:
        """The Trainer class to use for finetuning.

        Raises:
            NotImplementedError:
                Always; Laya is not finetuned.
        """
        raise NotImplementedError(
            "The `trainer_class` property has not been implemented for Laya, as "
            "it is not finetuned."
        )

    def update_dataset_config(self, dataset_config: "DatasetConfig") -> t.Self:
        """Update the dataset config registered in the benchmark module.

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


def _checkpoint_name(model_id: str, param: str | None) -> str:
    """Resolve the checkpoint name (a key into `LAYA_CHECKPOINTS`).

    Args:
        model_id:
            The Hub repo ID, or a local checkpoint directory.
        param:
            The parameter (variant) requested through `model_id#param`.

    Returns:
        The checkpoint name.
    """
    if param is not None:
        return param
    prefix = f"{LAYA_BUNDLED_REPO_ID}-"
    if model_id.startswith(prefix):
        return model_id.removeprefix(prefix)
    return ""


def _num_params_from_local_checkpoint(checkpoint_dir: Path) -> int:
    """Get the number of parameters of a local Laya checkpoint directory.

    Args:
        checkpoint_dir:
            The local checkpoint directory, containing `model.safetensors`.

    Returns:
        The number of parameters in the model.
    """
    from safetensors import safe_open  # noqa: PLC0415

    weights_path = checkpoint_dir / "model.safetensors"
    num_params = 0
    with safe_open(str(weights_path), framework="numpy") as f:
        for key in f.keys():
            shape = f.get_slice(key).get_shape()
            n = 1
            for dim in shape:
                n *= dim
            num_params += n
    return num_params


def _resolve_subfolder(model_id: str, param: str | None) -> str | None:
    """Resolve the subfolder to load within a Laya Hub repo.

    Args:
        model_id:
            The Hub repo ID.
        param:
            The parameter (variant) requested through `model_id#param`.

    Returns:
        The subfolder to pass to `laya.Agent`, or None for the repo root.

    Raises:
        InvalidModel:
            If a parameter is given for a repo other than the bundled
            `convaiinnovations/laya` repo.
    """
    if param is None:
        return None
    if model_id != LAYA_BUNDLED_REPO_ID:
        raise InvalidModel(
            f"The model {model_id!r} does not accept a parameter (only the "
            f"bundled {LAYA_BUNDLED_REPO_ID!r} repo does)."
        )
    return param
