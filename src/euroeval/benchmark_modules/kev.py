"""In-process adapter for local Kev pointer-head zero-shot checkpoints."""

from __future__ import annotations

import math
import typing as t
from pathlib import Path

import torch
from datasets import DatasetDict
from huggingface_hub import HfApi, snapshot_download
from transformers import Trainer

from ..data_models import (
    BenchmarkConfig,
    DatasetConfig,
    GenerativeModelOutput,
    ModelConfig,
    Task,
)
from ..enums import BatchingPreference, InferenceBackend, ModelType, TaskGroup
from ..exceptions import InvalidBenchmark, InvalidModel
from ..model_cache import create_model_cache_dir
from ..string_utils import split_model_id
from ..task_group_utils.cloze import parse_bare_question_and_choices
from ..types import ExtractLabelsFunction
from ..utils import get_hf_token
from .base import (
    BenchmarkModule,
    _extract_labels_from_generation_helper,
    _prepare_dataset_helper,
)


class KevModel(BenchmarkModule):
    """Standalone in-process backend for Kev decision checkpoints.

    Kev checkpoints contain a ``head.pt`` plus either a LoRA adapter or a full
    backbone. Kev itself constructs the causal backbone and trained pointer head;
    this backend never substitutes a generic text-classification head.
    """

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
        """Load a Kev checkpoint using Kev's canonical checkpoint loader.

        Args:
            model_config: The checkpoint identity and local model-cache directory.
            dataset_config: The dataset and label configuration.
            benchmark_config: The device, credentials, and evaluation settings.
            log_metadata: Whether to log the loaded model's metadata.

        Raises:
            InvalidModel: If the checkpoint dependency is unavailable or ID is invalid.
        """
        if model_config.param is not None:
            raise InvalidModel("Kev checkpoints do not support # parameters.")
        try:
            from kev.checkpoint import Checkpoint, LoadOptions  # noqa: PLC0415
        except ImportError as exc:
            raise InvalidModel(
                "Kev is not installed. Please install it with "
                "`pip install euroeval[kev]` or `pip install euroeval[all]`."
            ) from exc
        checkpoint_path = resolve_checkpoint_path(
            model_id=model_config.model_id,
            revision=model_config.revision,
            cache_dir=model_config.model_cache_dir,
            token=get_hf_token(api_key=benchmark_config.api_key),
        )
        self.tokenizer, self.model = Checkpoint(checkpoint_path).load(
            str(benchmark_config.device),
            LoadOptions(backend="torch", dtype=torch.float32),
        )
        self.model.eval()
        super().__init__(model_config, dataset_config, benchmark_config, log_metadata)
        self.buffer["first_label_token_mapping"] = True

    @property
    def data_collator(self) -> t.Callable:
        """Kev is evaluated zero-shot and has no finetuning collator."""
        raise NotImplementedError("Kev checkpoints cannot be finetuned by EuroEval.")

    @property
    def extract_labels_from_generation(self) -> ExtractLabelsFunction:
        """The standard classifier output-label extractor.

        Returns:
            The label extractor configured for this dataset.
        """
        return _extract_labels_from_generation_helper(
            dataset_config=self.dataset_config,
            model_config=self.model_config,
            first_label_token_mapping=self.buffer["first_label_token_mapping"],
        )

    def generate(self, inputs: dict[str, t.Any]) -> GenerativeModelOutput:
        """Score each sample with Kev's trained pointer head.

        Args:
            inputs: Batched examples containing the raw text to classify.

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
        texts = inputs.get("text")
        if texts is None:
            raise InvalidBenchmark("The inputs must contain a 'text' key.")
        labels = [
            self.dataset_config.prompt_label_mapping[label]
            for label in self.dataset_config.id2label.values()
        ]
        if task_group is TaskGroup.SEQUENCE_CLASSIFICATION and len(labels) < 2:
            raise InvalidBenchmark("Kev classification requires at least two labels.")
        sequences: list[str] = []
        scores: list[list[list[tuple[str, float]]]] = []
        for text in texts:
            state = str(text)
            if task_group is TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION:
                state, options = parse_bare_question_and_choices(text=state)
                if not state or len(options) < 2:
                    raise InvalidBenchmark(
                        f"Kev could not parse multiple-choice options from {text!r}."
                    )
                if labels and len(options) > len(labels):
                    raise InvalidBenchmark(
                        "Kev multiple-choice option count exceeds the dataset labels "
                        f"({len(options)} options, {len(labels)} labels)."
                    )
                output_labels = [chr(ord("a") + index) for index in range(len(options))]
            else:
                options = labels
                output_labels = labels
            record = {
                "state": state,
                "questions": [
                    {
                        "instr": self.dataset_config.instruction_prompt.format(
                            text="",
                            labels_str=self.dataset_config.get_labels_str(
                                labels=output_labels
                                if task_group
                                is TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
                                else None
                            ),
                        ).strip(),
                        "options": options,
                        "label": 0,
                    }
                ],
            }
            try:
                encoded = self.model.encode(
                    self.tokenizer,
                    record,
                    max_state=self.model_max_length,
                    max_branch=self.model_max_length + 8192,
                    strict=True,
                )
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
                        (label, math.log(max(probability, 1e-12)))
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

        Args:
            model_id: The checkpoint ID, optionally including its revision.
            benchmark_config: The evaluation settings and cache location.

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
    def model_exists(cls, model_id: str, benchmark_config: BenchmarkConfig) -> bool:
        """Recognize Kev's complete checkpoint layout without loading weights.

        Args:
            model_id: A local checkpoint path or Hugging Face repository ID.
            benchmark_config: The credentials used to inspect a remote checkpoint.

        Returns:
            Whether the directory or repository contains a complete Kev checkpoint.
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
        return True

    @property
    def model_max_length(self) -> int:
        """The maximum context length supported by Kev.

        Returns:
            The context limit in tokens.
        """
        return 65_536

    @property
    def num_params(self) -> int:
        """The loaded model's parameter count.

        Returns:
            The number of model parameters.
        """
        return sum(parameter.numel() for parameter in self.model.parameters())

    def prepare_dataset(
        self, dataset: DatasetDict, task: Task, itr_idx: int
    ) -> DatasetDict:
        """Prepare task fields while restoring the raw text Kev consumes.

        Args:
            dataset: The loaded dataset splits.
            task: The task whose fields should be prepared.
            itr_idx: The current evaluation iteration.

        Returns:
            The prepared dataset with the original sample text restored.

        Raises:
            InvalidBenchmark: If few-shot mode is enabled.
        """
        if self.benchmark_config.few_shot:
            raise InvalidBenchmark("Kev does not support few-shot evaluation.")
        raw_text = list(dataset["test"]["text"])
        prepared = _prepare_dataset_helper(
            dataset=dataset,
            task=task,
            model_config=self.model_config,
            dataset_config=self.dataset_config,
            benchmark_config=self.benchmark_config,
            generative_type=None,
            itr_idx=itr_idx,
            always_populate_text_field=False,
            tokeniser=None,
        )
        prepared["test"] = (
            prepared["test"].remove_columns("text").add_column("text", raw_text)
        )
        return prepared

    @property
    def trainer_class(self) -> t.Type[Trainer]:
        """Kev does not support EuroEval finetuning."""
        raise NotImplementedError("Kev checkpoints cannot be finetuned by EuroEval.")

    @property
    def vocab_size(self) -> int:
        """The checkpoint tokenizer vocabulary size.

        Returns:
            The number of tokens in the vocabulary.
        """
        return len(self.tokenizer)


def resolve_checkpoint_path(
    *, model_id: str, revision: str | None, cache_dir: str, token: str | None
) -> str:
    """Resolve a local Kev checkpoint or download the requested Hub revision.

    Args:
        model_id: The local directory or Hub repository ID.
        revision: The Hub revision to download, if specified.
        cache_dir: The directory for downloaded checkpoint files.
        token: The optional Hub access token.

    Returns:
        The local checkpoint directory.
    """
    if Path(model_id).is_dir():
        return model_id
    return snapshot_download(
        repo_id=model_id, revision=revision, cache_dir=cache_dir, token=token
    )
