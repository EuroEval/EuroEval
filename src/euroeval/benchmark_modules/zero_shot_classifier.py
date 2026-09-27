"""A benchmark module wrapping the Laya zero-shot classifier model."""

import collections.abc as c
import json
import math
import typing as t
from functools import cached_property
from pathlib import Path

from huggingface_hub import (
    HfApi,
    hf_hub_download,
    snapshot_download,
    try_to_load_from_cache,
)
from huggingface_hub.errors import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    SafetensorsParsingError,
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


# A probability floor, to avoid taking the log of zero.
_LOGPROB_FLOOR = 1e-12

# The marker file that ships with every Laya checkpoint (root or subfolder), used to
# detect Laya checkpoints structurally rather than through a fixed repo/name list.
_LAYA_CONFIG_FILENAME = "rl_agent_config.json"

# Laya's own fallback when a checkpoint config doesn't set `max_len` (see
# `self.cfg.get("max_len", 512)` in `laya.agent.Agent.system_one`).
_DEFAULT_MAX_LENGTH = 512


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

        subfolder = param
        token = get_hf_token(api_key=benchmark_config.api_key)

        # Resolve/download the requested repo (or `#subfolder`) through EuroEval's
        # configured cache, so `--download-only`, evaluation, and `clear_model_cache`
        # all agree on where the checkpoint lives, and so that other variants bundled
        # in the same repo aren't downloaded.
        checkpoint_path = _resolve_checkpoint_path(
            model_id=model_id,
            subfolder=subfolder,
            cache_dir=model_config.model_cache_dir,
            token=token,
        )
        config = _local_laya_config(
            directory=Path(checkpoint_path), subfolder=subfolder
        )
        self.max_length = (config or {}).get("max_len", _DEFAULT_MAX_LENGTH)

        self.agent = laya.Agent(
            model_id_or_path=checkpoint_path,
            subfolder=subfolder,
            token=token,
            device=str(benchmark_config.device),
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
        self._validate_labels()
        self.buffer["instructions"] = self._build_instructions()

    def _build_instructions(self) -> str:
        """Build the classification instructions from the dataset's templates.

        Returns:
            The instructions describing the classification task.
        """
        return self.dataset_config.instruction_prompt.format(
            text="", labels_str=self.dataset_config.get_labels_str()
        ).strip()

    def _validate_labels(self) -> None:
        """Check that a supported dataset has candidate labels.

        Raises:
            InvalidBenchmark:
                If the dataset has no candidate labels.
        """
        supported_task_groups = (
            TaskGroup.SEQUENCE_CLASSIFICATION,
            TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
        )
        if (
            self.dataset_config.task.task_group in supported_task_groups
            and not self.dataset_config.id2label
        ):
            raise InvalidBenchmark(
                "No candidate labels found for this dataset. Set "
                "DatasetConfig.labels/prompt_label_mapping for classification "
                "tasks before using Laya."
            )

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
                    (label, math.log(max(probs.get(label, 0.0), _LOGPROB_FLOOR)))
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
        criteria = {label: None for label in candidate_labels}
        return [self._classify_one(state=text, criteria=criteria) for text in texts]

    def _classify_one(
        self, state: str, criteria: dict[str, str | None]
    ) -> dict[str, float]:
        """Classify a single state against the candidate criteria, using Laya.

        This is the only place that talks to Laya, so it's the only place that
        translates a backend-agnostic representation -- stable labels mapped to
        (optional) candidate descriptions -- into Laya's own `{label: description}`
        `criteria` format. The returned probabilities stay keyed by those same
        stable labels.

        Args:
            state:
                The state (text) to classify.
            criteria:
                A mapping from each stable candidate label to its (optional)
                description.

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
            "criteria": criteria,
        }
        output = self.agent.system_one(state=state, questions={"q": question})
        probabilities = output["answers"]["q"]["probabilities"]
        missing_labels = [label for label in criteria if label not in probabilities]
        if missing_labels:
            raise InvalidBenchmark(
                "Laya did not return a probability for the candidate "
                f"label(s) {missing_labels!r}."
            )
        return {label: float(probabilities[label]) for label in criteria}

    def _classify_multiple_choice(
        self, texts: list[str], letter_labels: list[str]
    ) -> list[dict[str, float]]:
        """Classify multiple-choice samples, using the option texts as descriptions.

        The stable label for each option is its letter ("a", "b", ...); the option
        text is only ever passed once, as that label's description in `criteria` --
        not also baked into the question text, which would otherwise pass the same
        options to Laya twice.

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
            question_text, option_texts = parse_bare_question_and_choices(text=text)
            unparseable = len(option_texts) != len(letter_labels) or len(
                set(option_texts)
            ) != len(option_texts)
            if unparseable:
                # Couldn't reliably parse this sample's options (or two options share
                # the same text) -- fall back to classifying against the letters,
                # with the original (unparsed) text as the question.
                question_text = text
                criteria: dict[str, str | None] = {
                    letter: None for letter in letter_labels
                }
            else:
                criteria = {
                    letter: t.cast(str | None, option_text)
                    for letter, option_text in zip(
                        letter_labels, option_texts, strict=True
                    )
                }

            label_probs.append(
                self._classify_one(state=question_text, criteria=criteria)
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
        bare_model_id = model_id_components.model_id
        model_cache_dir = create_model_cache_dir(
            cache_dir=benchmark_config.cache_dir, model_id=bare_model_id
        )
        if param is not None:
            token = get_hf_token(api_key=benchmark_config.api_key)
            config, definitely_absent = _find_laya_config_with_certainty(
                model_id=bare_model_id,
                subfolder=param,
                token=token,
                cache_dir=model_cache_dir,
            )
            # Only hard-fail when we positively know the marker file is absent (a
            # reachable Hub said so, or a local/cached lookup came up empty). If the
            # Hub is merely unreachable, we don't yet know either way, so we let the
            # benchmark proceed and defer to the checkpoint download/load later.
            if config is None and definitely_absent:
                raise InvalidModel(
                    f"Invalid parameter {param!r} for model {bare_model_id!r}: no "
                    f"{_laya_config_filename(subfolder=param)!r} was found in that "
                    "repo."
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
            model_cache_dir=model_cache_dir,
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
        subfolder = model_id_components.param
        token = get_hf_token(api_key=benchmark_config.api_key)
        cache_dir = create_model_cache_dir(
            cache_dir=benchmark_config.cache_dir, model_id=bare_model_id
        )

        config = _find_laya_config(
            model_id=bare_model_id,
            subfolder=subfolder,
            token=token,
            cache_dir=cache_dir,
        )
        if config is None:
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
        param = self.model_config.param
        if Path(model_id).is_dir():
            return _num_params_from_local_checkpoint(checkpoint_dir=Path(model_id))

        token = get_hf_token(api_key=self.benchmark_config.api_key)
        if param is not None:
            # A variant's weights live in its own subfolder; read only the header.
            try:
                metadata = HfApi().parse_safetensors_file_metadata(
                    repo_id=model_id, filename=f"{param}/model.safetensors", token=token
                )
            except (OSError, EntryNotFoundError, SafetensorsParsingError):
                return -1
            return sum(metadata.parameter_count.values())

        try:
            num_params = get_num_params_from_safetensors_metadata(
                model_id=model_id, revision="main", api_key=token
            )
        except OSError:
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
        self._validate_labels()
        self.buffer["instructions"] = self._build_instructions()
        return self

    @cached_property
    def vocab_size(self) -> int:
        """The vocabulary size of the model.

        Returns:
            The vocabulary size of the model.
        """
        return -1


def _find_laya_config(
    model_id: str,
    subfolder: str | None,
    token: str | None,
    cache_dir: str | None = None,
) -> dict[str, t.Any] | None:
    """Locate a Laya checkpoint's config, structurally, wherever it lives.

    Checks, in order, a local directory, the local Hub cache, and finally the Hub
    itself -- the same `rl_agent_config.json` marker in every case.

    Args:
        model_id:
            The Hub repo ID, or a local checkpoint directory.
        subfolder:
            The requested `#subfolder`, if any.
        token:
            The Hugging Face Hub API token, if any.
        cache_dir:
            EuroEval's configured model cache directory for this model, checked
            before falling back to the default Hub cache. Optional.

    Returns:
        The parsed config, or None if no Laya checkpoint was found.
    """
    config, _ = _find_laya_config_with_certainty(
        model_id=model_id, subfolder=subfolder, token=token, cache_dir=cache_dir
    )
    return config


def _find_laya_config_with_certainty(
    model_id: str,
    subfolder: str | None,
    token: str | None,
    cache_dir: str | None = None,
) -> tuple[dict[str, t.Any] | None, bool]:
    """Locate a Laya checkpoint's config, also reporting how certain that is.

    Args:
        model_id:
            The Hub repo ID, or a local checkpoint directory.
        subfolder:
            The requested `#subfolder`, if any.
        token:
            The Hugging Face Hub API token, if any.
        cache_dir:
            EuroEval's configured model cache directory for this model, checked
            before falling back to the default Hub cache. Optional.

    Returns:
        A tuple `(config, definitely_absent)`. `definitely_absent` is only True when
        we positively know no Laya checkpoint exists here -- a local directory or a
        reachable Hub said so -- as opposed to merely being unable to tell, e.g.
        because the Hub is unreachable.
    """
    if Path(model_id).is_dir():
        return _local_laya_config(directory=Path(model_id), subfolder=subfolder), True
    cached = _cached_laya_config(
        repo_id=model_id, subfolder=subfolder, cache_dir=cache_dir
    )
    if cached is not None:
        return cached, True
    return _remote_laya_config(repo_id=model_id, subfolder=subfolder, token=token)


def _cached_laya_config(
    repo_id: str, subfolder: str | None, cache_dir: str | None = None
) -> dict[str, t.Any] | None:
    """Read a Laya checkpoint config from the local Hub cache, without a network call.

    Checks EuroEval's own configured cache directory first -- the same one
    `_resolve_checkpoint_path` downloads into -- falling back to the default Hub
    cache, so a checkpoint downloaded by EuroEval is found offline either way.

    Args:
        repo_id:
            The Hub repo ID.
        subfolder:
            The requested `#subfolder`, if any.
        cache_dir:
            EuroEval's configured model cache directory for this model, if any.

    Returns:
        The parsed config, or None if it isn't cached locally.
    """
    filename = _laya_config_filename(subfolder=subfolder)
    candidate_cache_dirs = [cache_dir, None] if cache_dir is not None else [None]
    for candidate_cache_dir in candidate_cache_dirs:
        cached = try_to_load_from_cache(
            repo_id=repo_id, filename=filename, cache_dir=candidate_cache_dir
        )
        if isinstance(cached, str):
            return json.loads(Path(cached).read_text())
    return None


def _laya_config_filename(subfolder: str | None) -> str:
    """The path (relative to a repo or local directory) of a Laya checkpoint config.

    Args:
        subfolder:
            The requested `#subfolder`, if any.

    Returns:
        The relative path to `rl_agent_config.json`.
    """
    return (
        f"{subfolder}/{_LAYA_CONFIG_FILENAME}" if subfolder else _LAYA_CONFIG_FILENAME
    )


def _local_laya_config(
    directory: Path, subfolder: str | None
) -> dict[str, t.Any] | None:
    """Read a Laya checkpoint config from a local directory, if present.

    Args:
        directory:
            The local checkpoint directory (repo root or clone).
        subfolder:
            The requested `#subfolder`, if any.

    Returns:
        The parsed config, or None if no marker file is present.
    """
    path = directory / _laya_config_filename(subfolder=subfolder)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _remote_laya_config(
    repo_id: str, subfolder: str | None, token: str | None
) -> tuple[dict[str, t.Any] | None, bool]:
    """Fetch a Laya checkpoint config from the Hub, if that repo/subfolder has one.

    Args:
        repo_id:
            The Hub repo ID.
        subfolder:
            The requested `#subfolder`, if any.
        token:
            The Hugging Face Hub API token, if any.

    Returns:
        A tuple `(config, definitely_absent)`. `definitely_absent` is True only when
        the Hub was reachable and positively reported that the file doesn't exist;
        it is False when the Hub couldn't be reached at all (or the repo itself
        couldn't be found), since then we simply don't know.
    """
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=_laya_config_filename(subfolder=subfolder),
            token=token,
        )
    except EntryNotFoundError:
        return None, True
    except (RepositoryNotFoundError, OSError):
        return None, False
    return json.loads(Path(path).read_text()), True


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
    with safe_open(str(weights_path), framework="numpy") as f:
        return sum(math.prod(f.get_slice(key).get_shape()) for key in f.keys())


def _resolve_checkpoint_path(
    model_id: str, subfolder: str | None, cache_dir: str, token: str | None
) -> str:
    """Resolve the requested checkpoint through EuroEval's own configured cache.

    A local directory is returned unchanged. Otherwise, only the requested repo root
    (or `#subfolder`) is downloaded into `cache_dir`, so `--download-only`, evaluation,
    and `clear_model_cache` all agree on where the checkpoint lives, and unrelated
    variants bundled in the same repo aren't downloaded.

    Args:
        model_id:
            The Hub repo ID, or a local checkpoint directory.
        subfolder:
            The requested `#subfolder`, if any.
        cache_dir:
            EuroEval's configured model cache directory for this model.
        token:
            The Hugging Face Hub API token, if any.

    Returns:
        The local path to the downloaded (or already-local) checkpoint.
    """
    if Path(model_id).is_dir():
        return model_id

    # A requested subfolder is a self-contained variant checkpoint, so the whole
    # subfolder is downloaded and nothing else -- sibling variants bundled in the
    # same repo aren't. At the root, only the files a Laya checkpoint actually
    # ships with are downloaded, since the root of a repo bundling variants also
    # contains those variants' (much larger) subfolders.
    allow_patterns = (
        [f"{subfolder}/**"]
        if subfolder
        else [_LAYA_CONFIG_FILENAME, "model.safetensors", "tokenizer/**", "encoder/**"]
    )
    return snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        token=token,
    )


def is_laya_checkpoint(model_id: str, cache_dir: str | None = None) -> bool:
    """Detect, without a network call, whether a model ID is a Laya checkpoint.

    Used by other benchmark modules (namely the generic HF encoder) to defer to the
    zero-shot classifier module when they can't otherwise tell -- e.g. offline, where
    `siblings` isn't available.

    Args:
        model_id:
            The model ID, optionally with a `#subfolder`.
        cache_dir:
            EuroEval's configured model cache directory for this model -- the same
            one `_resolve_checkpoint_path` downloads into -- checked before falling
            back to the default Hub cache, so a checkpoint downloaded by EuroEval is
            found here too. Optional.

    Returns:
        Whether this looks like a Laya checkpoint, judging only by a local directory
        or the local Hub cache.
    """
    model_id_components = split_model_id(model_id=model_id)
    bare_model_id = model_id_components.model_id
    subfolder = model_id_components.param
    if Path(bare_model_id).is_dir():
        return (
            _local_laya_config(directory=Path(bare_model_id), subfolder=subfolder)
            is not None
        )
    return (
        _cached_laya_config(
            repo_id=bare_model_id, subfolder=subfolder, cache_dir=cache_dir
        )
        is not None
    )
