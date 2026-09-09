"""Fail-closed Hugging Face model safety and fit checks."""

import dataclasses
import json
import platform
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from .types import Gpu, Lease

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclasses.dataclass(frozen=True)
class SafetyReport:
    """Successful safety check details."""

    estimated_bytes: int
    available_bytes: int


class SafetyError(RuntimeError):
    """Raised when a model cannot be safely evaluated."""


def check_model_safety(
    lease: Lease,
    gpus: tuple[Gpu, ...],
    metadata: "ModelMetadata | None" = None,
    free_disk_bytes: int | None = None,
) -> SafetyReport:
    """Verify public, pinned, safetensors-only, non-code model metadata.

    Returns:
        Details of the successful fit check.

    Raises:
        SafetyError:
            If the revision, repository, weights, or memory estimate is unsafe.
    """
    if platform.machine() not in {"x86_64", "amd64", "aarch64", "arm64"}:
        raise SafetyError("unsupported worker architecture")
    if not _COMMIT_RE.fullmatch(lease.model_revision):
        raise SafetyError("broker supplied an unpinned model revision")
    info = metadata or HuggingFaceMetadata().fetch(
        model_id=lease.model_id, revision=lease.model_revision
    )
    if info.private or info.gated:
        raise SafetyError("model is private or gated")
    if info.auto_map:
        raise SafetyError("model declares auto_map and would require remote code")
    if any(path.lower().endswith(".py") for path in info.files):
        raise SafetyError("model repository contains Python files")
    if not info.safetensors:
        raise SafetyError("model has no safetensors weights")
    estimated = info.estimated_bytes
    if free_disk_bytes is not None and free_disk_bytes < estimated * 2:
        raise SafetyError("insufficient disk space for the model repository and cache")
    # Do not add unrelated GPUs: vLLM can only use a topology explicitly
    # configured by the backend. The broker currently leases one GPU.
    available = max((gpu.free_memory_bytes for gpu in gpus), default=0)
    available = int(available * 0.9)
    if estimated > available:
        raise SafetyError(
            f"model needs approximately {estimated} bytes but only {available} "
            "bytes are free on one supported GPU"
        )
    return SafetyReport(estimated_bytes=estimated, available_bytes=available)


class ModelMetadata:
    """Metadata required by the safety preflight."""

    def __init__(
        self,
        *,
        private: bool,
        gated: bool,
        auto_map: bool,
        files: tuple[str, ...],
        safetensors: bool,
        estimated_bytes: int,
    ) -> None:
        """Initialise metadata used by :func:`check_model_safety`."""
        self.private = private
        self.gated = gated
        self.auto_map = auto_map
        self.files = files
        self.safetensors = safetensors
        self.estimated_bytes = estimated_bytes


class HuggingFaceMetadata:
    """Fetch only public model metadata without consuming a token."""

    def fetch(self, model_id: str, revision: str) -> ModelMetadata:
        """Fetch metadata from the Hub and reject incomplete responses.

        Returns:
            Repository metadata.

        Raises:
            SafetyError:
                If the Hub cannot provide complete metadata.
        """
        try:
            info = HfApi(token=False).model_info(
                repo_id=model_id, revision=revision, files_metadata=True
            )
            config_path = Path(
                hf_hub_download(
                    repo_id=model_id,
                    filename="config.json",
                    revision=revision,
                    token=False,
                )
            )
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise SafetyError(
                f"could not verify Hugging Face metadata: {error}"
            ) from error
        siblings = getattr(info, "siblings", None)
        if not isinstance(siblings, list):
            raise SafetyError("Hugging Face response omitted repository files")
        files: list[str] = []
        estimated = 0
        for sibling in siblings:
            name = getattr(sibling, "rfilename", None)
            if not isinstance(name, str):
                continue
            files.append(name)
            size = getattr(sibling, "size", None)
            if name.endswith(".safetensors") and isinstance(size, int):
                estimated += size
        return ModelMetadata(
            private=bool(getattr(info, "private", False)),
            gated=bool(getattr(info, "gated", False)),
            auto_map=bool(config.get("auto_map")),
            files=tuple(files),
            safetensors=estimated > 0
            and any(name.endswith(".safetensors") for name in files),
            estimated_bytes=estimated,
        )
