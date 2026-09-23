"""Adapter wrapping the Laya zero-shot "decision" model family."""

import logging
import typing as t
from pathlib import Path

from ..exceptions import InvalidModel, NeedsExtraInstalled
from ..logging_utils import log_once
from .base import ZeroShotClassifierAdapter

if t.TYPE_CHECKING:
    from ..data_models import BenchmarkConfig, ModelConfig

# `convaiinnovations/laya` bundles several checkpoints in one Hub repo, selected
# through the `model_id#param` syntax, each in its own subfolder (`None` means the
# repo root, the English checkpoint). Other `convaiinnovations/laya-*` repos (e.g.
# `laya-multilingual`, `laya-typed-decisions`) are standalone, single-checkpoint
# repos with the same files at their root and take no parameter.
BUNDLED_REPO_ID = "convaiinnovations/laya"
BUNDLED_PARAM_TO_SUBFOLDER: dict[str, str] = {
    "multilingual": "multilingual",
    "typed-decisions": "typed-decisions",
}

# The context length (in tokens) of each checkpoint. Keyed by (repo_id, param) for
# the bundled repo's variants, and by repo_id alone for standalone repos (where
# param is always None).
MAX_LENGTH_BY_CHECKPOINT: dict[tuple[str, str | None], int] = {
    (BUNDLED_REPO_ID, None): 512,
    (BUNDLED_REPO_ID, "multilingual"): 1024,
    (BUNDLED_REPO_ID, "typed-decisions"): 512,
    ("convaiinnovations/laya-multilingual", None): 1024,
    ("convaiinnovations/laya-typed-decisions", None): 512,
}
DEFAULT_MAX_LENGTH = 512


class LayaAdapter(ZeroShotClassifierAdapter):
    """Adapter wrapping Laya (https://pypi.org/project/laya/), a decision model.

    Laya is an encoder with trained decision heads, loaded through its own `laya`
    package rather than `transformers`. It answers typed questions (`choice`,
    `score`, `noul`) with calibrated per-label probabilities; this adapter only
    uses the `choice` question type, since that's what sequence classification
    and multiple-choice classification tasks reduce to.
    """

    name = "laya"
    # Variants of the bundled `convaiinnovations/laya` repo only; standalone repos
    # (`convaiinnovations/laya-multilingual`, `-typed-decisions`, a local checkpoint
    # directory) take no parameter, which `__init__` enforces.
    variants = list(BUNDLED_PARAM_TO_SUBFOLDER)
    max_length = DEFAULT_MAX_LENGTH

    def __init__(
        self, model_config: "ModelConfig", benchmark_config: "BenchmarkConfig"
    ) -> None:
        """Load the Laya agent for the requested checkpoint.

        Args:
            model_config:
                The model configuration.
            benchmark_config:
                The benchmark configuration.
        """
        import laya  # noqa: PLC0415

        self.model_config = model_config
        self.benchmark_config = benchmark_config

        model_id = model_config.model_id
        param = model_config.param
        subfolder = _resolve_subfolder(model_id=model_id, param=param)
        self.max_length = MAX_LENGTH_BY_CHECKPOINT.get(
            (model_id, param), DEFAULT_MAX_LENGTH
        )

        # `Router` is intentionally not used, since it would select a checkpoint on
        # its own; we always load exactly the checkpoint the model ID asked for.
        self.agent = laya.Agent(
            model_id, subfolder=subfolder, token=benchmark_config.api_key
        )

    def classify(
        self, texts: list[str], candidate_labels: list[str], instructions: str
    ) -> list[dict[str, float]]:
        """Classify each text against the candidate labels.

        Each text is turned into a single `choice` question, with the candidate
        labels as the question's criteria and the dataset's instructions as the
        question's instructions. Laya's `system_one` call already truncates the
        state and criteria to the checkpoint's `max_len`/`head_max_len`, so no
        separate truncation is done here.

        Args:
            texts:
                The texts to classify.
            candidate_labels:
                The candidate labels to classify each text into.
            instructions:
                The instructions describing the classification task.

        Returns:
            A list, with one dictionary per text, mapping each candidate label to
            its predicted probability.
        """
        question = {
            "type": "choice",
            "instructions": instructions,
            "criteria": {label: None for label in candidate_labels},
        }
        results = []
        for text in texts:
            output = self.agent.system_one(state=text, questions={"q": question})
            probabilities = output["answers"]["q"]["probabilities"]
            results.append(
                {label: float(probabilities[label]) for label in candidate_labels}
            )
        return results

    @classmethod
    def matches(
        cls, model_id: str, benchmark_config: "BenchmarkConfig"
    ) -> "bool | NeedsExtraInstalled":
        """Check whether the given model ID refers to a Laya checkpoint.

        Args:
            model_id:
                The model ID to check, without revision or parameter suffixes.
            benchmark_config:
                The benchmark configuration.

        Returns:
            Whether the model ID is a Laya checkpoint, or a `NeedsExtraInstalled`
            error if the `laya` package is not installed.
        """
        # Cheap, offline checks only, so every non-Laya model ID checked against
        # this adapter costs no Hub call:
        #   - any `convaiinnovations/laya*` Hub repo ID (the bundled repo and its
        #     standalone `-multilingual`/`-typed-decisions`/future siblings), or
        #   - a local directory that itself contains `rl_agent_config.json` (a
        #     custom checkpoint a user trained or downloaded themselves).
        is_known_hub_repo = model_id.startswith("convaiinnovations/laya")
        is_local_checkpoint_dir = (
            Path(model_id).is_dir()
            and (Path(model_id) / "rl_agent_config.json").is_file()
        )
        if not (is_known_hub_repo or is_local_checkpoint_dir):
            return False
        try:
            import laya  # noqa: F401,PLC0415
        except ImportError:
            return NeedsExtraInstalled(extra="laya")
        return True

    @classmethod
    def num_params(cls, model_id: str, param: str | None) -> int:
        """Get the number of parameters of the given Laya checkpoint.

        Reads only the safetensors header (a small range request), rather than
        downloading the full checkpoint.

        Args:
            model_id:
                The model ID, without revision or parameter suffixes.
            param:
                The parameter (variant) of the model, or None if no parameter was
                specified.

        Returns:
            The number of parameters in the model, or -1 if it could not be
            determined.
        """
        try:
            from huggingface_hub import parse_safetensors_file_metadata  # noqa: PLC0415

            if Path(model_id).is_dir():
                return _num_params_from_local_checkpoint(checkpoint_dir=Path(model_id))

            subfolder = _resolve_subfolder(model_id=model_id, param=param)
            filename = (
                f"{subfolder}/model.safetensors" if subfolder else ("model.safetensors")
            )
            metadata = parse_safetensors_file_metadata(
                repo_id=model_id, filename=filename
            )
            return sum(metadata.parameter_count.values())
        except Exception as error:
            log_once(
                f"Could not determine the number of parameters of the Laya "
                f"checkpoint {model_id!r} (param={param!r}): {error!r}.",
                level=logging.DEBUG,
            )
            return -1


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
            `convaiinnovations/laya` repo, which doesn't accept one.
    """
    if param is None:
        return None
    if model_id != BUNDLED_REPO_ID:
        raise InvalidModel(
            f"The model {model_id!r} does not accept a parameter (only the "
            f"bundled {BUNDLED_REPO_ID!r} repo does, via {BUNDLED_REPO_ID}#"
            f"{param!r})."
        )
    return BUNDLED_PARAM_TO_SUBFOLDER[param]
