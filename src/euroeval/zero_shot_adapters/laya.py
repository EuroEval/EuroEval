"""Adapter wrapping the Laya zero-shot "decision" model family."""

import typing as t

from ..exceptions import NeedsExtraInstalled
from .base import ZeroShotClassifierAdapter

if t.TYPE_CHECKING:
    from ..data_models import BenchmarkConfig, ModelConfig

# Laya bundles several checkpoints in one Hub repo, selected through the
# `model_id#param` syntax. `None` (no parameter) means the repo root.
PARAM_TO_SUBFOLDER: dict[str | None, str | None] = {
    None: None,
    "multilingual": "multilingual",
    "typed-decisions": "typed-decisions",
}

# The context length (in tokens) of each checkpoint, keyed by the same parameter.
PARAM_TO_MAX_LENGTH: dict[str | None, int] = {
    None: 512,
    "multilingual": 1024,
    "typed-decisions": 512,
}


class LayaAdapter(ZeroShotClassifierAdapter):
    """Adapter wrapping Laya (https://pypi.org/project/laya/), a decision model.

    Laya is an encoder with trained decision heads, loaded through its own `laya`
    package rather than `transformers`. It answers typed questions (`choice`,
    `score`, `noul`) with calibrated per-label probabilities; this adapter only
    uses the `choice` question type, since that's what sequence classification
    and multiple-choice classification tasks reduce to.
    """

    name = "laya"
    allowed_params = ["multilingual", "typed-decisions"]
    max_length = PARAM_TO_MAX_LENGTH[None]

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
        # Cheap, offline check: the only Laya repo currently published is
        # `convaiinnovations/laya`, which bundles all of its checkpoints as
        # subfolders selected through `allowed_params`. This avoids a Hub call
        # for every model ID that gets checked against this adapter.
        if model_id != "convaiinnovations/laya":
            return False
        try:
            import laya  # noqa: F401,PLC0415
        except ImportError:
            return NeedsExtraInstalled(extra="laya")
        return True

    @classmethod
    def num_params(cls, model_id: str, param: str | None) -> int:
        """Get the number of parameters of the given Laya checkpoint.

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
            from huggingface_hub import hf_hub_download  # noqa: PLC0415
            from safetensors import safe_open  # noqa: PLC0415

            subfolder = PARAM_TO_SUBFOLDER[param]
            filename = (
                f"{subfolder}/model.safetensors" if subfolder else ("model.safetensors")
            )
            weights_path = hf_hub_download(repo_id=model_id, filename=filename)
            num_params = 0
            with safe_open(weights_path, framework="numpy") as f:
                for key in f.keys():
                    shape = f.get_slice(key).get_shape()
                    n = 1
                    for dim in shape:
                        n *= dim
                    num_params += n
            return num_params
        except Exception:
            return -1

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

        param = model_config.param
        subfolder = PARAM_TO_SUBFOLDER[param]
        self.max_length = PARAM_TO_MAX_LENGTH[param]

        # `Router` is intentionally not used, since it would select a checkpoint on
        # its own; we always load exactly the checkpoint the model ID asked for.
        self.agent = laya.Agent(
            model_config.model_id, subfolder=subfolder, token=benchmark_config.api_key
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
