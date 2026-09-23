"""Adapter wrapping the Laya zero-shot "decision" model family."""

import logging
import typing as t
from pathlib import Path

from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    NotASafetensorsRepoError,
    RepositoryNotFoundError,
)

from ..exceptions import InvalidBenchmark, InvalidModel, NeedsExtraInstalled
from ..logging_utils import log_once
from ..safetensors_utils import get_num_params_from_safetensors_metadata
from ..utils import get_hf_token
from .base import ZeroShotClassifierAdapter

if t.TYPE_CHECKING:
    from ..data_models import BenchmarkConfig, ModelConfig

# `convaiinnovations/laya` bundles several checkpoints in one Hub repo, selected
# through the `model_id#param` syntax, each in its own subfolder named after the
# checkpoint (the root/English checkpoint's name is ""). Other
# `convaiinnovations/laya-*` repos (e.g. `laya-multilingual`, `laya-typed-decisions`)
# are standalone, single-checkpoint repos with the same files at their root -- one
# per non-root checkpoint here -- and take no parameter.
BUNDLED_REPO_ID = "convaiinnovations/laya"

# Each checkpoint's context length (in tokens), keyed by checkpoint name. "" is the
# root/English checkpoint, also used as the fallback for unrecognised checkpoints
# (e.g. a local checkpoint directory). A non-root name doubles as both the
# `model_id#param` value for the bundled repo and the `laya-<name>` suffix of the
# corresponding standalone repo, since the subfolder equals the param name.
CHECKPOINTS: dict[str, int] = {"": 512, "multilingual": 1024, "typed-decisions": 512}
DEFAULT_MAX_LENGTH = CHECKPOINTS[""]


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
    variants = [checkpoint_name for checkpoint_name in CHECKPOINTS if checkpoint_name]
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

        Raises:
            InvalidModel:
                If a revision other than "main" is requested, since the `laya`
                package cannot load a specific revision, or if a parameter is
                given for a repo that doesn't accept one.
        """
        import laya  # noqa: PLC0415

        self.model_config = model_config
        self.benchmark_config = benchmark_config

        model_id = model_config.model_id
        param = model_config.param
        revision = model_config.revision
        # `laya.Agent` loads checkpoints via `huggingface_hub.snapshot_download`
        # without a `revision` argument, so it always fetches the repo's default
        # branch ("main"). There is no way to pin another revision, so we reject
        # any revision other than "main" up front, rather than silently ignoring
        # it.
        if revision not in ("main", ""):
            raise InvalidModel(
                f"The model {model_id!r} was requested at revision {revision!r}, "
                "but the `laya` package does not support loading a specific "
                "revision -- it always loads the repo's default branch ('main')."
            )
        subfolder = _resolve_subfolder(model_id=model_id, param=param)
        checkpoint_name = _checkpoint_name(model_id=model_id, param=param)

        token = get_hf_token(api_key=benchmark_config.api_key)

        # `Router` is intentionally not used, since it would select a checkpoint on
        # its own; we always load exactly the checkpoint the model ID asked for.
        self.agent = laya.Agent(model_id, subfolder=subfolder, token=token)

        if checkpoint_name in CHECKPOINTS:
            self.max_length = CHECKPOINTS[checkpoint_name]
        else:
            # Unknown standalone `laya-*` repo (or a local checkpoint directory):
            # the loaded agent's own config carries the context length it was
            # trained with, so prefer that over the hardcoded default when present.
            cfg = getattr(self.agent, "cfg", {})
            max_len = cfg.get("max_len") if isinstance(cfg, dict) else None
            if isinstance(max_len, int) and max_len > 0:
                self.max_length = max_len
            else:
                self.max_length = DEFAULT_MAX_LENGTH
                log_once(
                    f"Could not determine the context length of the Laya "
                    f"checkpoint {model_id!r} (param={param!r}) from its config; "
                    f"falling back to the default of {DEFAULT_MAX_LENGTH} tokens.",
                    level=logging.INFO,
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

        Raises:
            InvalidBenchmark:
                If Laya's response is missing a probability for one of the
                candidate labels.
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
            missing_labels = [
                label for label in candidate_labels if label not in probabilities
            ]
            if missing_labels:
                raise InvalidBenchmark(
                    "Laya did not return a probability for the candidate "
                    f"label(s) {missing_labels!r}. Expected labels: "
                    f"{candidate_labels!r}. Returned labels: "
                    f"{list(probabilities.keys())!r}."
                )
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
        is_known_hub_repo = model_id == BUNDLED_REPO_ID or model_id.startswith(
            f"{BUNDLED_REPO_ID}-"
        )
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
            if Path(model_id).is_dir():
                return _num_params_from_local_checkpoint(checkpoint_dir=Path(model_id))

            subfolder = _resolve_subfolder(model_id=model_id, param=param)
            filename = (
                f"{subfolder}/model.safetensors" if subfolder else "model.safetensors"
            )
            # `laya.Agent` only ever loads the "main" revision (see `__init__`), so
            # the same revision is used here to keep the parameter count consistent
            # with the checkpoint that will actually be loaded.
            num_params = get_num_params_from_safetensors_metadata(
                model_id=model_id,
                revision="main",
                api_key=get_hf_token(api_key=None),
                filename=filename,
            )
            if num_params is None:
                return -1
            return num_params
        except (
            EntryNotFoundError,
            GatedRepoError,
            HfHubHTTPError,
            NotASafetensorsRepoError,
            RepositoryNotFoundError,
            OSError,
        ) as error:
            log_once(
                f"Could not determine the number of parameters of the Laya "
                f"checkpoint {model_id!r} (param={param!r}): {error!r}.",
                level=logging.WARNING,
            )
            return -1


def _checkpoint_name(model_id: str, param: str | None) -> str:
    """Resolve the checkpoint name (a key into `CHECKPOINTS`) for a model/param pair.

    Args:
        model_id:
            The Hub repo ID, or a local checkpoint directory.
        param:
            The parameter (variant) requested through `model_id#param`.

    Returns:
        The checkpoint name: `param` for the bundled repo's variants, the
        `laya-<name>` suffix for a recognised standalone repo, or "" (the
        root/default checkpoint) otherwise.
    """
    if param is not None:
        return param
    prefix = f"{BUNDLED_REPO_ID}-"
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
        The subfolder to pass to `laya.Agent`, or None for the repo root. The
        subfolder is always equal to `param` itself, since checkpoint names are
        chosen to match the bundled repo's subfolder layout.

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
    return param
