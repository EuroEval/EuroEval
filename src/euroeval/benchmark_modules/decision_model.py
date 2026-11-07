"""Typed decision-model adapter for Hugging Face decision-model checkpoints."""

import collections.abc as c
import json
import math
import typing as t
from functools import cached_property
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download

from ..data_models import (
    BenchmarkConfig,
    DatasetConfig,
    GenerativeModelOutput,
    ModelConfig,
)
from ..enums import (
    BatchingPreference,
    GenerativeType,
    InferenceBackend,
    ModelType,
    TaskGroup,
)
from ..exceptions import InvalidBenchmark, InvalidModel
from ..model_cache import create_model_cache_dir
from ..string_utils import split_model_id
from ..task_group_utils.cloze import parse_bare_question_and_choices
from ..types import ExtractLabelsFunction
from ..utils import get_hf_token
from .base import BenchmarkModule, _extract_labels_from_generation_helper
from .zero_shot_classifier import ZeroShotClassifierModel

_SUPPORTED_FORMATS = {"systemone-v1"}


class DecisionModel(ZeroShotClassifierModel):
    """Run Hub decision-models through the typed System One wire format."""

    fresh_model = False
    batching_preference = BatchingPreference.SINGLE_SAMPLE
    high_priority = True

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
        log_metadata: bool = True,
    ) -> None:
        """Initialise the remote typed classifier.

        Args:
            model_config: Resolved Hub model and revision.
            dataset_config: Dataset and label configuration.
            benchmark_config: Evaluation and endpoint settings.
            log_metadata (optional): Whether to log model metadata.

        Raises:
            InvalidModel: If the typed endpoint is not configured.
            InvalidBenchmark: If this task group is unsupported.
        """
        if benchmark_config.api_base is None:
            raise InvalidModel(
                "This decision-model requires a typed System One endpoint. Set "
                "--api-base to the endpoint base URL; generic generation fallback "
                "is not supported."
            )
        if dataset_config.task.task_group not in {
            TaskGroup.SEQUENCE_CLASSIFICATION,
            TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
        }:
            raise InvalidBenchmark(
                "Typed decision models support sequence and multiple-choice "
                "classification tasks only."
            )
        BenchmarkModule.__init__(
            self,
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=log_metadata,
        )
        self.buffer["first_label_token_mapping"] = True
        self._validate_labels()
        self.buffer["instructions"] = self._build_instructions()

    @property
    def data_collator(self) -> c.Callable[[list[dict[str, t.Any]]], dict[str, t.Any]]:
        """The unsupported finetuning collator.

        Raises:
            NotImplementedError: Remote decision models cannot be finetuned here.
        """
        raise NotImplementedError("Remote decision models do not support finetuning.")

    @property
    def extract_labels_from_generation(self) -> ExtractLabelsFunction:
        """EuroEval's standard classification label extractor.

        Returns:
            The output label extractor.
        """
        return _extract_labels_from_generation_helper(
            dataset_config=self.dataset_config,
            model_config=self.model_config,
            first_label_token_mapping=True,
        )

    def generate(self, inputs: dict) -> GenerativeModelOutput:
        """Classify batch inputs and expose direct choice probabilities as scores.

        Args:
            inputs: Batch containing the prepared text values.

        Returns:
            Predicted labels and log-probability scores.

        Raises:
            InvalidBenchmark: If inputs or typed response are malformed.
        """
        if "text" not in inputs:
            raise InvalidBenchmark("The inputs must contain a 'text' key.")
        labels = [
            self.dataset_config.prompt_label_mapping[label]
            for label in self.dataset_config.id2label.values()
        ]
        scores: list[list[list[tuple[str, float]]]] = []
        sequences: list[str] = []
        for text in inputs["text"]:
            criteria: dict[str, str | None] = {label: None for label in labels}
            state = text
            if (
                self.dataset_config.task.task_group
                == TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
            ):
                question, options = parse_bare_question_and_choices(text=text)
                if len(options) == len(labels) and len(set(options)) == len(options):
                    state = question
                    criteria = dict(zip(labels, options, strict=True))
            probabilities = self._request_classification(state=state, criteria=criteria)
            label_scores = sorted(
                (
                    (label, math.log(max(probabilities[label], 1e-12)))
                    for label in labels
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            sequences.append(label_scores[0][0])
            scores.append([label_scores])
        return GenerativeModelOutput(sequences=sequences, scores=scores)

    def _request_classification(
        self, *, state: str, criteria: dict[str, str | None]
    ) -> dict[str, float]:
        """Request and validate a typed choice distribution.

        Args:
            state: Text to classify.
            criteria: Candidate labels and optional descriptions.

        Returns:
            A validated probability for each candidate.

        Raises:
            InvalidBenchmark: If the returned probabilities or identity are invalid.
            InvalidModel: If the endpoint cannot complete the request.
        """
        assert self.benchmark_config.api_base is not None
        try:
            response = requests.post(
                f"{self.benchmark_config.api_base.rstrip('/')}/v1/systemone",
                json={
                    "state": state,
                    "model": self.benchmark_config.served_model
                    or self.model_config.model_id,
                    "questions": {
                        "name": {
                            "type": "choice",
                            "instructions": self.buffer["instructions"],
                            "criteria": criteria,
                        }
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise InvalidModel(
                f"Decision-model endpoint request failed ({type(error).__name__})."
            ) from None
        try:
            body = response.json()
            returned_model = body["model"]
            probabilities = body["answers"]["name"]["probabilities"]
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidBenchmark(
                "Typed decision-model response omitted its model identity or choice "
                "probabilities."
            ) from error
        expected_model = (
            self.benchmark_config.served_model or self.model_config.model_id
        )
        if returned_model != expected_model:
            raise InvalidBenchmark(
                "Typed decision-model endpoint returned an unexpected served model."
            )
        if not isinstance(probabilities, dict) or set(probabilities) != set(criteria):
            raise InvalidBenchmark(
                "Typed decision-model returned an invalid label set."
            )
        parsed: dict[str, float] = {}
        for label in criteria:
            try:
                probability = float(probabilities[label])
            except (TypeError, ValueError) as error:
                raise InvalidBenchmark(
                    "Typed decision-model returned a non-numeric probability."
                ) from error
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise InvalidBenchmark(
                    "Typed decision-model returned an invalid probability."
                )
            parsed[label] = probability
        if not math.isclose(sum(parsed.values()), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise InvalidBenchmark(
                "Typed decision-model probabilities must sum to one."
            )
        return parsed

    @property
    def generative_type(self) -> GenerativeType | None:
        """None, because decision models classify rather than generate."""
        return None

    @classmethod
    def get_model_config(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> ModelConfig:
        """Resolve decision format metadata and build classifier configuration.

        Args:
            model_id: Hub repository and optional revision.
            benchmark_config: Evaluation settings and cache location.

        Returns:
            The typed decision-model configuration.

        Raises:
            InvalidModel: If metadata is missing or its format is unsupported.
        """
        components = split_model_id(model_id=model_id)
        try:
            config_path = hf_hub_download(
                repo_id=components.model_id,
                filename="decision_config.json",
                revision=components.revision,
                token=get_hf_token(api_key=benchmark_config.api_key),
                cache_dir=benchmark_config.cache_dir,
            )
            metadata = json.loads(Path(config_path).read_text())
        except Exception as error:  # noqa: BLE001 - normalise Hub metadata failures
            raise InvalidModel(
                f"Could not read decision_config.json for {components.model_id!r}."
            ) from error
        format_id = metadata.get("format_id") if isinstance(metadata, dict) else None
        if format_id not in _SUPPORTED_FORMATS:
            raise InvalidModel(
                f"Unsupported decision-model format {format_id!r}; supported formats "
                f"are {sorted(_SUPPORTED_FORMATS)!r}."
            )
        return ModelConfig(
            model_id=components.model_id,
            revision=components.revision,
            param=None,
            task="text-classification",
            languages=[],
            merge=False,
            inference_backend=InferenceBackend.SYSTEM_ONE,
            model_type=ModelType.ZERO_SHOT_CLASSIFIER,
            fresh=False,
            model_cache_dir=create_model_cache_dir(
                cache_dir=benchmark_config.cache_dir, model_id=components.model_id
            ),
            adapter_base_model_id=None,
        )

    @classmethod
    def model_exists(cls, model_id: str, benchmark_config: BenchmarkConfig) -> bool:
        """Select tagged repositories independently of repository identity.

        Args:
            model_id: Hub repository and optional revision.
            benchmark_config: Evaluation settings and Hugging Face credentials.

        Returns:
            Whether repository metadata contains the decision-model tag.
        """
        components = split_model_id(model_id=model_id)
        if Path(components.model_id).is_dir():
            return (Path(components.model_id) / "decision_config.json").is_file()
        try:
            info = HfApi(
                token=get_hf_token(api_key=benchmark_config.api_key)
            ).model_info(repo_id=components.model_id, revision=components.revision)
        except Exception:  # noqa: BLE001 - missing and inaccessible repos are not matches
            return False
        return "decision-model" in (info.tags or [])

    @cached_property
    def model_max_length(self) -> int:
        """Unknown context length."""
        return -1

    @cached_property
    def num_params(self) -> int:
        """Unknown parameter count."""
        return -1

    @property
    def trainer_class(self) -> t.Type["Trainer"]:
        """Expose the unavailable remote trainer."""
        raise NotImplementedError("Remote decision models do not support finetuning.")

    def update_dataset_config(self, dataset_config: DatasetConfig) -> t.Self:
        """Update the active dataset instructions.

        Args:
            dataset_config: New dataset configuration.

        Returns:
            This adapter.
        """
        self.dataset_config = dataset_config
        self._validate_labels()
        self.buffer["instructions"] = self._build_instructions()
        return self

    @cached_property
    def vocab_size(self) -> int:
        """Unknown vocabulary size."""
        return -1


if t.TYPE_CHECKING:
    from transformers.trainer import Trainer
