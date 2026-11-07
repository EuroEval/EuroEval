"""Direct client for Typesafe's hosted System One classifier."""

import collections.abc as c
import math
import os
import re
import time
import typing as t
from functools import cached_property

import requests

from ..data_models import BenchmarkConfig, DatasetConfig, ModelConfig
from ..enums import (
    BatchingPreference,
    GenerativeType,
    InferenceBackend,
    ModelType,
    TaskGroup,
)
from ..exceptions import InvalidBenchmark, InvalidModel, NeedsAdditionalArgument
from ..model_cache import create_model_cache_dir
from ..string_utils import split_model_id
from .base import BenchmarkModule
from .typed_choice_classification import TypedChoiceClassificationModel

_SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone"
_MODEL_ID_PATTERN = re.compile(r"typesafe/jev-(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")
_LOGPROB_FLOOR = 1e-12
_MAX_REQUEST_ATTEMPTS = 3
_MAX_RETRY_AFTER = 10.0

if t.TYPE_CHECKING:
    from transformers.trainer import Trainer


class TypesafeSystemOneModel(TypedChoiceClassificationModel):
    """Hosted zero-shot classifier using Typesafe System One."""

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
        """Initialise the hosted classifier without loading local model weights.

        Args:
            model_config:
                The hosted model configuration.
            dataset_config:
                The dataset and label configuration.
            benchmark_config:
                The evaluation settings and credentials.
            log_metadata (optional):
                Whether to log model metadata. Defaults to True.

        Raises:
            NeedsAdditionalArgument:
                If TYPESAFE_API_KEY is not set.
            InvalidBenchmark:
                If the dataset task is unsupported.
        """
        self.api_key = os.getenv("TYPESAFE_API_KEY")
        if not self.api_key:
            raise NeedsAdditionalArgument(
                cli_argument="TYPESAFE_API_KEY environment variable",
                script_argument=(
                    'os.environ["TYPESAFE_API_KEY"] = "<your-typesafe-api-key>"'
                ),
                run_with_cli=benchmark_config.run_with_cli,
            )
        if dataset_config.task.task_group not in {
            TaskGroup.SEQUENCE_CLASSIFICATION,
            TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
        }:
            raise InvalidBenchmark(
                "Typesafe System One supports sequence and multiple-choice "
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

    def _classify_one(
        self, state: str, criteria: dict[str, str | None]
    ) -> dict[str, float]:
        """Send one typed choice question to the System One endpoint.

        Args:
            state:
                The text state to classify.
            criteria:
                Candidate labels and optional descriptions.

        Returns:
            A probability for every candidate label.

        Raises:
            InvalidBenchmark:
                If the response does not contain probabilities for every label.
        """
        for attempt in range(1, _MAX_REQUEST_ATTEMPTS + 1):
            try:
                response = requests.post(
                    _SYSTEM_ONE_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "state": state,
                        "model": self.model_config.model_id.removeprefix("typesafe/"),
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
            except requests.HTTPError as error:
                response = error.response
                status = response.status_code if response is not None else None
                if status not in {429} and (status is None or status < 500):
                    raise self._request_failure(
                        cause=f"HTTP status {status}", attempts=attempt, input_count=1
                    ) from None
                if attempt == _MAX_REQUEST_ATTEMPTS:
                    raise self._request_failure(
                        cause=f"HTTP status {status}", attempts=attempt, input_count=1
                    ) from None
                self._wait_before_retry(
                    attempt=attempt,
                    retry_after=(
                        response.headers.get("Retry-After")
                        if response is not None
                        else None
                    ),
                )
            except (requests.ConnectionError, requests.Timeout) as error:
                if attempt == _MAX_REQUEST_ATTEMPTS:
                    raise self._request_failure(
                        cause=type(error).__name__, attempts=attempt, input_count=1
                    ) from None
                self._wait_before_retry(attempt=attempt, retry_after=None)
            else:
                break
        try:
            probabilities = response.json()["answers"]["name"]["probabilities"]
        except (KeyError, TypeError) as error:
            raise InvalidBenchmark(
                "Typesafe System One returned a response without choice probabilities."
            ) from error
        if not isinstance(probabilities, dict):
            raise InvalidBenchmark(
                "Typesafe System One returned malformed choice probabilities."
            )
        missing_labels = [label for label in criteria if label not in probabilities]
        if missing_labels:
            raise InvalidBenchmark(
                "Typesafe System One did not return probabilities for labels "
                f"{missing_labels!r}."
            )
        parsed: dict[str, float] = {}
        for label in criteria:
            try:
                probability = float(probabilities[label])
            except (TypeError, ValueError) as error:
                raise InvalidBenchmark(
                    "Typesafe System One returned a non-numeric choice probability."
                ) from error
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise InvalidBenchmark(
                    "Typesafe System One returned a probability outside [0, 1]."
                )
            parsed[label] = probability
        return parsed

    def _request_failure(
        self, *, cause: str, attempts: int, input_count: int
    ) -> InvalidModel:
        """Create a safe, contextual error for a failed service request.

        Args:
            cause:
                HTTP status or exception class describing the failure.
            attempts:
                Number of requests made.
            input_count:
                Number of text inputs in the generation batch.

        Returns:
            An informative model error without request content or credentials.
        """
        return InvalidModel(
            f"Typesafe model '{self.model_config.model_id}' request failed "
            f"after {attempts} "
            f"attempt{'s' if attempts != 1 else ''} for {input_count} text "
            f"input: {cause}."
        )

    @staticmethod
    def _wait_before_retry(*, attempt: int, retry_after: str | None) -> None:
        """Pause briefly before retrying, honouring a bounded numeric Retry-After.

        Args:
            attempt:
                Failed attempt number, starting at one.
            retry_after:
                Optional Retry-After header value in seconds.
        """
        delay = min(0.5 * (2 ** (attempt - 1)), _MAX_RETRY_AFTER)
        if retry_after is not None:
            try:
                requested_delay = float(retry_after)
            except ValueError:
                pass
            else:
                if math.isfinite(requested_delay):
                    delay = min(max(requested_delay, 0.0), _MAX_RETRY_AFTER)
        time.sleep(delay)

    @property
    def data_collator(self) -> c.Callable[[list[dict[str, t.Any]]], dict[str, t.Any]]:
        """The unavailable finetuning data collator.

        Returns:
            The data collator, if supported.

        Raises:
            NotImplementedError:
                Typesafe System One does not support finetuning.
        """
        raise NotImplementedError("Typesafe System One does not support finetuning.")

    @property
    def generative_type(self) -> GenerativeType | None:
        """None, because System One is a typed classifier, not a generator.

        Returns:
            None, since this model does not generate text.
        """
        return None

    @classmethod
    def get_model_config(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> ModelConfig:
        """Create configuration for the exact hosted model identifier.

        Args:
            model_id:
                The Typesafe model identifier.
            benchmark_config:
                The evaluation configuration, including cache location.

        Returns:
            The dedicated Typesafe model configuration.

        Raises:
            InvalidModel:
                If the ID does not contain a valid three-part numeric version.
        """
        if _MODEL_ID_PATTERN.fullmatch(model_id) is None:
            raise InvalidModel(
                "Typesafe System One model IDs must use the form "
                "'typesafe/jev-<version>' with a three-part numeric version, "
                "for example 'typesafe/jev-1.13.0'."
            )
        components = split_model_id(model_id=model_id)
        return ModelConfig(
            model_id=components.model_id,
            revision=components.revision,
            param=None,
            task="text-classification",
            languages=list(),
            merge=False,
            inference_backend=InferenceBackend.TYPESAFE,
            model_type=ModelType.ZERO_SHOT_CLASSIFIER,
            fresh=False,
            model_cache_dir=create_model_cache_dir(
                cache_dir=benchmark_config.cache_dir, model_id=components.model_id
            ),
            adapter_base_model_id=None,
            release_date=(
                "2026-09-15" if components.model_id == "typesafe/jev-1.13.0" else None
            ),
        )

    @classmethod
    def model_exists(cls, model_id: str, benchmark_config: BenchmarkConfig) -> bool:
        """Recognise the documented model ID without making a billable request.

        Args:
            model_id:
                The model identifier to check.
            benchmark_config:
                The evaluation configuration.

        Returns:
            Whether the ID selects Jev System One.
        """
        return _MODEL_ID_PATTERN.fullmatch(model_id) is not None

    @cached_property
    def model_max_length(self) -> int:
        """The unknown context limit.

        Returns:
            -1 because Typesafe publishes no verified context limit.
        """
        return -1

    @cached_property
    def num_params(self) -> int:
        """The unknown parameter count.

        Returns:
            -1 because Typesafe publishes no verified parameter count.
        """
        return -1

    @property
    def trainer_class(self) -> t.Type["Trainer"]:
        """The unavailable finetuning trainer class.

        Returns:
            The trainer class, if supported.

        Raises:
            NotImplementedError:
                Typesafe System One does not support finetuning.
        """
        raise NotImplementedError("Typesafe System One does not support finetuning.")

    @cached_property
    def vocab_size(self) -> int:
        """The unknown vocabulary size.

        Returns:
            -1 because Typesafe publishes no verified vocabulary size.
        """
        return -1
