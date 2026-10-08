"""In-process adapter for local Kev pointer-head zero-shot checkpoints."""

from __future__ import annotations

import math
import typing as t
from pathlib import Path

import torch
from datasets import DatasetDict
from huggingface_hub import HfApi
from transformers import Trainer

from ..data_models import (
    BenchmarkConfig,
    DatasetConfig,
    GenerativeModelOutput,
    ModelConfig,
    Task,
)
from ..enums import BatchingPreference, InferenceBackend, ModelType, TaskGroup
from ..exceptions import InvalidBenchmark, InvalidModel, NeedsExtraInstalled
from ..model_cache import create_model_cache_dir
from ..string_utils import split_model_id
from ..task_group_utils.cloze import parse_bare_question_and_choices
from .base import BenchmarkModule


class KevModel(BenchmarkModule):
    """Standalone in-process backend for Kev decision checkpoints.

    Kev checkpoints contain a ``head.pt`` plus either a LoRA adapter or a full
    backbone. Kev itself constructs the causal backbone and trained pointer head;
    this backend never substitutes a generic text-classification head.
    """

    batching_preference = BatchingPreference.SINGLE_SAMPLE
    high_priority = True

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
        log_metadata: bool = True,
    ) -> None:
        """Load a Kev checkpoint using Kev's canonical checkpoint loader.

        Raises:
            InvalidModel: If the checkpoint dependency is unavailable or ID is invalid.
        """
        if model_config.param is not None:
            raise InvalidModel("Kev checkpoints do not support # parameters.")
        try:
            from kev.checkpoint import Checkpoint, LoadOptions  # noqa: PLC0415
        except ImportError as exc:
            raise InvalidModel(
                "Install the upstream Kev repository package to use this backend."
            ) from exc
        local = Path(model_config.model_id)
        # Checkpoint.resolve_run handles Hub IDs and downloads only checkpoint files.
        self.tokenizer, self.model = Checkpoint(
            str(local) if local.is_dir() else model_config.model_id
        ).load(
            str(benchmark_config.device),
            LoadOptions(backend="torch", dtype=torch.float32),
        )
        self.model.eval()
        self.buffer["first_label_token_mapping"] = False
        super().__init__(model_config, dataset_config, benchmark_config, log_metadata)

    @property
    def data_collator(self) -> t.Callable:
        """Kev is evaluated zero-shot and has no finetuning collator."""
        raise NotImplementedError("Kev checkpoints cannot be finetuned by EuroEval.")

    @property
    def extract_labels_from_generation(self) -> t.Callable:
        """The function that returns labels directly from the pointer head."""
        return lambda **kwargs: kwargs

    def generate(self, inputs: dict[str, t.Any]) -> GenerativeModelOutput:
        """Score each sample with Kev's trained pointer head.

        Returns:
            Class predictions and their log-probability scores.

        Raises:
            InvalidBenchmark: If task type or Kev probability output is invalid.
        """
        task_group = self.dataset_config.task.task_group
        if task_group not in (
            TaskGroup.SEQUENCE_CLASSIFICATION,
            TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
        ):
            raise InvalidBenchmark(f"Kev does not support task group {task_group!r}.")
        if self.dataset_config.task.requires_logprobs:
            raise InvalidBenchmark("Kev does not expose token-level log probabilities.")
        texts = inputs.get("text")
        if texts is None:
            raise InvalidBenchmark("The inputs must contain a 'text' key.")
        labels = [
            self.dataset_config.prompt_label_mapping[label]
            for label in self.dataset_config.id2label.values()
        ]
        if len(labels) < 2:
            raise InvalidBenchmark("Kev classification requires at least two labels.")
        sequences: list[str] = []
        scores: list[list[list[tuple[str, float]]]] = []
        for text in texts:
            options = labels
            output_labels = labels
            state = str(text)
            if task_group is TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION:
                state, parsed = parse_bare_question_and_choices(text=state)
                if not state or len(parsed) < 2 or len(set(parsed)) != len(parsed):
                    raise InvalidBenchmark(
                        f"Kev could not parse multiple-choice options from {text!r}."
                    )
                options = parsed
                output_labels = labels[: len(parsed)]
            record = {
                "state": state,
                "questions": [
                    {
                        "instr": self.dataset_config.instruction_prompt.replace(
                            "{text}", ""
                        ).strip(),
                        "options": options,
                        "label": options[0],
                    }
                ],
            }
            try:
                encoded = self.model.encode(self.tokenizer, record, strict=True)
                probs = self.model.probs(encoded)
                probabilities = probs[0].detach().cpu().tolist()
            except (ValueError, TypeError, IndexError, AttributeError) as exc:
                raise InvalidBenchmark(
                    f"Kev returned malformed classifier output: {exc}"
                ) from exc
            if (
                len(probabilities) != len(options)
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                    for value in probabilities
                )
                or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-3, abs_tol=1e-3)
            ):
                raise InvalidBenchmark("Kev returned malformed class probabilities.")
            ranked = sorted(
                zip(output_labels, probabilities, strict=True),
                key=lambda pair: pair[1],
                reverse=True,
            )
            sequences.append(ranked[0][0])
            scores.append(
                [
                    [
                        (label, math.log(max(probability, 1e-30)))
                        for label, probability in ranked
                    ]
                ]
            )
        return GenerativeModelOutput(sequences=sequences, scores=scores)

    @property
    def generative_type(self) -> None:
        """Kev is a classifier, not a text generator."""
        return None

    @classmethod
    def get_model_config(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> ModelConfig:
        """Build a Kev model configuration.

        Returns:
            The generated model configuration.

        Raises:
            InvalidModel: If the model ID includes an unsupported parameter.
        """
        components = split_model_id(model_id=model_id)
        if components.param is not None:
            raise InvalidModel("Kev checkpoints do not support # parameters.")
        return ModelConfig(
            model_id=components.model_id,
            revision=components.revision,
            param=None,
            task="text-classification",
            languages=[],
            merge=False,
            inference_backend=InferenceBackend.KEV,
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
        """Recognize Kev's complete checkpoint layout without loading weights.

        Returns:
            True if metadata matches Kev, otherwise False or a missing-extra diagnostic.
        """
        bare = split_model_id(model_id=model_id).model_id
        path = Path(bare)
        if path.is_dir():
            files = {p.name for p in path.iterdir() if p.is_file()}
        else:
            try:
                files = set(
                    HfApi().list_repo_files(
                        repo_id=bare,
                        revision=split_model_id(model_id=model_id).revision or None,
                        token=benchmark_config.api_key,
                    )
                )
            except Exception:
                return False
        # A PEFT config alone is common and must not claim arbitrary adapters.
        has_adapter = (
            "adapter_config.json" in files and "adapter_model.safetensors" in files
        )
        has_full = "config.json" in files and any(
            name.startswith("model") and name.endswith(".safetensors") for name in files
        )
        if "head.pt" not in files or not (has_adapter or has_full):
            return False
        try:
            import kev.checkpoint  # noqa: F401,PLC0415
        except ImportError:
            return NeedsExtraInstalled(extra="kev")
        return True

    @property
    def model_max_length(self) -> int:
        """Kev supports extended context; defer to its checkpoint's native limits."""
        return 65_536

    @property
    def num_params(self) -> int:
        """The loaded model's parameter count."""
        return sum(parameter.numel() for parameter in self.model.parameters())

    def prepare_dataset(
        self, dataset: DatasetDict, task: Task, itr_idx: int
    ) -> DatasetDict:
        """Return a dataset unchanged because Kev consumes raw text.

        Returns:
            The original dataset.
        """
        return dataset

    @property
    def trainer_class(self) -> t.Type[Trainer]:
        """Kev does not support EuroEval finetuning."""
        raise NotImplementedError("Kev checkpoints cannot be finetuned by EuroEval.")

    @property
    def vocab_size(self) -> int:
        """The checkpoint tokenizer vocabulary size."""
        return len(self.tokenizer)
