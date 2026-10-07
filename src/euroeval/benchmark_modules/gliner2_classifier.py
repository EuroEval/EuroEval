"""GLiNER2-backed zero-shot classification benchmark module."""

import json
import typing as t
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from huggingface_hub.utils import validate_repo_id

from ..data_models import (
    BenchmarkConfig,
    DatasetConfig,
    GenerativeModelOutput,
    ModelConfig,
)
from ..enums import InferenceBackend, ModelType, TaskGroup
from ..exceptions import InvalidBenchmark, InvalidModel, NeedsExtraInstalled
from ..model_cache import create_model_cache_dir
from ..string_utils import split_model_id
from ..utils import get_hf_token
from .base import BenchmarkModule
from .zero_shot_classifier import ZeroShotClassifierModel


class GLiNER2ClassifierModel(ZeroShotClassifierModel):
    """Zero-shot sequence classifier using GLiNER2's schema classification API."""

    high_priority = True

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
        log_metadata: bool = True,
    ) -> None:
        """Load GLiNER2 without importing its optional package until needed.

        Args:
            model_config: The model configuration.
            dataset_config: The dataset configuration.
            benchmark_config: The benchmark configuration.
            log_metadata: Whether to log model metadata.

        Raises:
            InvalidModel: If a parameterized or revision-specific ID is requested.
        """
        if model_config.param is not None:
            raise InvalidModel(
                "GLiNER2 classifier checkpoints do not support # parameters."
            )
        if model_config.revision not in ("", "main"):
            raise InvalidModel(
                "GLiNER2 AutoExtractor does not expose revision selection; "
                "use the main revision."
            )
        try:
            from gliner2 import AutoExtractor  # noqa: PLC0415
        except ImportError as exc:
            raise InvalidModel(
                "Install EuroEval with `pip install euroeval[gliner2]`."
            ) from exc

        token = get_hf_token(api_key=benchmark_config.api_key)
        local_path = Path(model_config.model_id)
        if local_path.is_dir():
            checkpoint = str(local_path)
        else:
            checkpoint = snapshot_download(
                repo_id=model_config.model_id,
                cache_dir=model_config.model_cache_dir,
                token=token,
            )
        self.model = AutoExtractor.from_pretrained(
            checkpoint, map_location=str(benchmark_config.device)
        )
        BenchmarkModule.__init__(
            self,
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=log_metadata,
        )

    def generate(self, inputs: dict[str, t.Any]) -> GenerativeModelOutput:
        """Classify text and return only the selected label, without invented scores.

        Args:
            inputs: Batch of examples containing raw ``text`` strings.

        Returns:
            One selected label per sample. GLiNER2's classification API only returns
            the selected label and optional selected-label confidence, not a score for
            every candidate, so ``scores`` is intentionally omitted.

        Raises:
            InvalidBenchmark: If the task or batch is unsupported or a result is absent.
        """
        task_group = self.dataset_config.task.task_group
        if task_group is not TaskGroup.SEQUENCE_CLASSIFICATION:
            raise InvalidBenchmark(
                "GLiNER2 classifier supports sequence classification only; "
                f"{task_group!r} is not supported."
            )
        if "text" not in inputs:
            raise InvalidBenchmark("The inputs must contain a 'text' key.")
        labels = [
            self.dataset_config.prompt_label_mapping[label]
            for label in self.dataset_config.id2label.values()
        ]
        if not labels:
            raise InvalidBenchmark("GLiNER2 classification requires candidate labels.")
        sequences: list[str] = []
        for text in inputs["text"]:
            result = self.model.classify_text(
                str(text), {"label": labels}, include_confidence=False
            )
            value = result.get("label") if isinstance(result, dict) else None
            if isinstance(value, dict):
                value = value.get("label")
            if not isinstance(value, str) or value not in labels:
                raise InvalidBenchmark(
                    f"GLiNER2 returned no valid selected label for {text!r}: "
                    f"{result!r}."
                )
            sequences.append(value)
        return GenerativeModelOutput(sequences=sequences)

    @classmethod
    def get_model_config(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> ModelConfig:
        """Build a classifier configuration for a verified checkpoint.

        Args:
            model_id: Hub model ID or local checkpoint path.
            benchmark_config: The active benchmark configuration.

        Returns:
            The generated model configuration.

        Raises:
            InvalidModel: If the model ID contains a parameter/subfolder.
        """
        components = split_model_id(model_id=model_id)
        if components.param is not None:
            raise InvalidModel(
                "GLiNER2 classifier checkpoints do not support # parameters."
            )
        return ModelConfig(
            model_id=components.model_id,
            revision=components.revision,
            param=None,
            task="text-classification",
            languages=list(),
            merge=False,
            inference_backend=InferenceBackend.GLINER2,
            model_type=ModelType.ZERO_SHOT_CLASSIFIER,
            fresh=False,
            model_cache_dir=create_model_cache_dir(
                cache_dir=benchmark_config.cache_dir, model_id=components.model_id
            ),
            adapter_base_model_id=None,
        )

    @classmethod
    def model_exists(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> bool | NeedsExtraInstalled:
        """Identify GLiNER2 checkpoints using only small config metadata.

        Args:
            model_id: Hub model ID or local checkpoint path.
            benchmark_config: The active benchmark configuration.

        Returns:
            True for compatible checkpoint metadata, otherwise False or a missing-extra
            diagnostic. This method never downloads model weights.
        """
        components = split_model_id(model_id=model_id)
        bare_id = components.model_id
        local = Path(bare_id)
        token = get_hf_token(api_key=benchmark_config.api_key)
        if local.is_dir():
            config_path = local / "config.json"
        else:
            try:
                validate_repo_id(bare_id)
                config_path = Path(
                    hf_hub_download(
                        repo_id=bare_id,
                        filename="config.json",
                        revision=components.revision or None,
                        cache_dir=create_model_cache_dir(
                            cache_dir=benchmark_config.cache_dir, model_id=bare_id
                        ),
                        token=token,
                    )
                )
            except (HFValidationError, RepositoryNotFoundError, OSError):
                return False
        try:
            config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(config, dict) or not _is_gliner2_config(config):
            return False
        try:
            import gliner2  # noqa: F401,PLC0415
        except ImportError:
            return NeedsExtraInstalled(extra="gliner2")
        return True


def _is_gliner2_config(config: dict[str, t.Any]) -> bool:
    """Return whether checkpoint metadata identifies a GLiNER2 classifier.

    Args:
        config: Parsed Hugging Face config metadata.

    Returns:
        Whether the checkpoint is known to be a GLiNER2-compatible classifier.
    """
    architecture = " ".join(str(value) for value in config.get("architectures", []))
    model_type = str(config.get("model_type", ""))
    metadata = f"{architecture} {model_type}".casefold()
    # Do not identify by repository name alone: the Hub ID can be a typo, a fork,
    # or a repository whose config was replaced. Both architecture/model_type are
    # checkpoint metadata, unlike the arbitrary underlying encoder name.
    return "gliner2" in metadata
