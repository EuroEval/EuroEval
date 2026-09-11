"""Fail-closed Hugging Face model safety and fit checks."""

import dataclasses
import json
import platform
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig

from .types import Gpu, Lease

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SUPPORTED_HOST_ARCHITECTURES = {"x86_64", "amd64", "aarch64", "arm64"}
_MODEL_TYPES = frozenset({"encoder", "generative"})
_GENERATIVE_ARCHITECTURE_RE = re.compile(
    r"(?:For(?:CausalLM|ConditionalGeneration|Seq2SeqLM)|LMHeadModel)$"
)
_ENCODER_ARCHITECTURE_RE = re.compile(
    r"For(?:SequenceClassification|TokenClassification|QuestionAnswering|"
    r"MultipleChoice|MaskedLM|PreTraining)$"
)

_UNSAFE_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx", ".h5", ".msgpack")


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
        architectures: tuple[str, ...] = (),
        repository_bytes: int | None = None,
        model_type: str | None = None,
        is_encoder_decoder: bool | None = None,
    ) -> None:
        """Initialise metadata used by :func:`check_model_safety`."""
        self.private = private
        self.gated = gated
        self.auto_map = auto_map
        self.files = files
        self.safetensors = safetensors
        self.estimated_bytes = estimated_bytes
        self.architectures = architectures
        derived_type = derive_model_type(
            architectures=architectures, is_encoder_decoder=is_encoder_decoder
        )
        self.model_type = derived_type if model_type in {None, derived_type} else None
        self.repository_bytes = (
            estimated_bytes
            if repository_bytes is None and not architectures
            else repository_bytes
        )


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
        repository_bytes = 0
        for sibling in siblings:
            name = getattr(sibling, "rfilename", None)
            if not isinstance(name, str):
                continue
            files.append(name)
            size = getattr(sibling, "size", None)
            if not isinstance(size, int) or size < 0:
                raise SafetyError("Hugging Face response omitted file sizes")
            repository_bytes += size
            if name.lower().endswith(".safetensors"):
                estimated += size
        architectures = config.get("architectures")
        if not isinstance(architectures, list) or not all(
            isinstance(item, str) and item for item in architectures
        ):
            raise SafetyError("Hugging Face config omitted architectures")
        if not isinstance(config.get("model_type"), str) or not config["model_type"]:
            raise SafetyError("Hugging Face config omitted model_type")
        try:
            AutoConfig.from_pretrained(
                config_path.parent, local_files_only=True, trust_remote_code=False
            )
        except Exception as error:
            raise SafetyError(
                "model is not compatible with the installed Transformers stack"
            ) from error
        model_type = derive_model_type(
            architectures=tuple(architectures),
            is_encoder_decoder=config.get("is_encoder_decoder"),
        )
        if model_type is None:
            raise SafetyError(
                "Hugging Face config has no supported EuroEval capability"
            )
        return ModelMetadata(
            private=bool(getattr(info, "private", False)),
            gated=bool(getattr(info, "gated", False)),
            auto_map=bool(
                config.get("auto_map")
                or config.get("custom_code")
                or config.get("trust_remote_code")
            ),
            files=tuple(files),
            safetensors=estimated > 0
            and any(name.lower().endswith(".safetensors") for name in files),
            estimated_bytes=estimated,
            architectures=tuple(architectures),
            repository_bytes=repository_bytes,
            model_type=model_type,
            is_encoder_decoder=config.get("is_encoder_decoder"),
        )


class SafetyError(RuntimeError):
    """Raised when a model cannot be safely evaluated."""


@dataclasses.dataclass(frozen=True)
class SafetyReport:
    """Successful safety check details."""

    estimated_bytes: int
    available_bytes: int


def check_model_safety(
    lease: Lease,
    gpus: tuple[Gpu, ...],
    metadata: "ModelMetadata | None" = None,
    free_disk_bytes: int | None = None,
    gpu_memory_utilisation: float = 0.8,
    selected_gpu: Gpu | None = None,
) -> SafetyReport:
    """Verify public, pinned, safetensors-only, non-code model metadata.

    Returns:
        Details of the successful fit check.

    Raises:
        SafetyError:
            If the revision, repository, capability, or memory estimate is unsafe.
    """
    if platform.machine() not in _SUPPORTED_HOST_ARCHITECTURES:
        raise SafetyError("unsupported worker architecture")
    if not _COMMIT_RE.fullmatch(lease.model_revision):
        raise SafetyError("broker supplied an unpinned model revision")
    if not 0 < gpu_memory_utilisation <= 1:
        raise SafetyError("GPU memory utilisation must be between 0 and 1")
    info = metadata or HuggingFaceMetadata().fetch(
        model_id=lease.model_id, revision=lease.model_revision
    )
    if info.private or info.gated:
        raise SafetyError("model is private or gated")
    if info.auto_map:
        raise SafetyError("model declares auto_map and would require remote code")
    if lease.model_type not in _MODEL_TYPES:
        raise SafetyError("broker supplied an unsupported model type")
    if info.model_type is None:
        raise SafetyError("model capability could not be derived")
    if info.model_type != lease.model_type:
        raise SafetyError("model capability does not match its broker type")
    if any(path.lower().endswith(".py") for path in info.files):
        raise SafetyError("model repository contains Python files")
    if any(path.lower().endswith(_UNSAFE_SUFFIXES) for path in info.files):
        raise SafetyError("model repository contains unsafe weight artifacts")
    if not info.safetensors:
        raise SafetyError("model has no safetensors weights")
    if info.repository_bytes is None:
        raise SafetyError("model repository size could not be verified")
    if free_disk_bytes is not None and free_disk_bytes < info.repository_bytes:
        raise SafetyError("insufficient disk space for the model repository")
    if selected_gpu is None:
        if len(gpus) != 1:
            raise SafetyError("a selected GPU is required for multi-GPU hosts")
        selected_gpu = gpus[0]
    available = int(selected_gpu.free_memory_bytes * gpu_memory_utilisation)
    if not available or info.estimated_bytes * 1.35 > available:
        raise SafetyError(
            f"model needs approximately {info.estimated_bytes} bytes but only "
            f"{available} bytes are free on one supported GPU"
        )
    return SafetyReport(estimated_bytes=info.estimated_bytes, available_bytes=available)


def derive_model_type(
    *, architectures: tuple[str, ...], is_encoder_decoder: object = None
) -> str | None:
    """Derive EuroEval's broad capability from standard Transformers metadata.

    The task suffix is deliberately used instead of a maintained model-family list;
    new families supported by the pinned stack therefore do not need policy changes.
    Ambiguous or unknown task architectures remain unavailable to volunteers.

    Returns:
        The broad capability, or ``None`` for ambiguous metadata.
    """
    if len(architectures) != 1 or not architectures[0]:
        return None
    architecture = architectures[0]
    if is_encoder_decoder is True or _GENERATIVE_ARCHITECTURE_RE.search(architecture):
        return "generative"
    if _ENCODER_ARCHITECTURE_RE.search(architecture):
        return "encoder"
    return None
