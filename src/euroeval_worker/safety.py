"""Fail-closed Hugging Face model safety and fit checks."""

import collections.abc as c
import dataclasses
import importlib
import json
import platform
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
)

from euroeval.constants import GENERATIVE_PIPELINE_TAGS

from .types import Gpu, Lease

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SUPPORTED_HOST_ARCHITECTURES = {"x86_64", "amd64", "aarch64", "arm64"}
_MODEL_TYPES = frozenset({"encoder", "generative"})
_UNSAFE_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx", ".h5", ".msgpack")
_ENCODER_TASK_MAPPINGS: dict[str, object] = {
    "sequence_classification": MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    "token_classification": MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    "question_answering": MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    "multiple_choice_classification": MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    "fill_mask": MODEL_FOR_MASKED_LM_MAPPING,
}
_ENCODER_MAPPINGS = tuple(_ENCODER_TASK_MAPPINGS.values())
_ENCODER_PIPELINE_TAGS = frozenset(
    {
        "fill-mask",
        "text-classification",
        "token-classification",
        "question-answering",
        "feature-extraction",
        "sentence-similarity",
        "zero-shot-classification",
    }
)


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
        pipeline_tag: str | None = None,
        model_type: str | None = None,
        is_encoder_decoder: bool | None = None,
        backend_compatible: bool = False,
        config: object | None = None,
    ) -> None:
        """Initialise metadata used by :func:`check_model_safety`."""
        self.private = private
        self.gated = gated
        self.auto_map = auto_map
        self.files = files
        self.safetensors = safetensors
        self.estimated_bytes = estimated_bytes
        self.architectures = architectures
        self.pipeline_tag = pipeline_tag
        self.is_encoder_decoder = is_encoder_decoder
        derived_type = derive_model_type(
            pipeline_tag=pipeline_tag, is_encoder_decoder=is_encoder_decoder
        )
        self.model_type = derived_type if model_type in {None, derived_type} else None
        self.backend_compatible = backend_compatible
        self.config = config
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
        hub_model_id = getattr(info, "id", None)
        if (
            not isinstance(hub_model_id, str)
            or hub_model_id.casefold() != model_id.casefold()
        ):
            raise SafetyError("Hugging Face returned incomplete model identity")
        siblings = getattr(info, "siblings", None)
        if not isinstance(siblings, list):
            raise SafetyError("Hugging Face response omitted repository files")
        pipeline_tag = getattr(info, "pipeline_tag", None)
        if not isinstance(pipeline_tag, str) or not pipeline_tag.strip():
            raise SafetyError("Hugging Face response omitted pipeline_tag")
        files: list[str] = []
        estimated = 0
        repository_bytes = 0
        for sibling in siblings:
            name = getattr(sibling, "rfilename", None)
            if not isinstance(name, str) or not name:
                raise SafetyError("Hugging Face response omitted repository filenames")
            files.append(name)
            size = getattr(sibling, "size", None)
            if not isinstance(size, int) or size < 0:
                raise SafetyError("Hugging Face response omitted file sizes")
            repository_bytes += size
            if name.lower().endswith(".safetensors"):
                estimated += size
        if any(path.lower().endswith(".py") for path in files):
            raise SafetyError("model repository contains Python files")
        if any(path.lower().endswith(_UNSAFE_SUFFIXES) for path in files):
            raise SafetyError("model repository contains unsafe weight artifacts")
        if any(
            config.get(field)
            for field in ("auto_map", "custom_code", "trust_remote_code")
        ):
            raise SafetyError("model declares custom code and remote code is disabled")
        architectures = config.get("architectures")
        if (
            not isinstance(architectures, list)
            or not architectures
            or not all(isinstance(item, str) and item for item in architectures)
        ):
            raise SafetyError("Hugging Face config omitted architectures")
        config_model_type = config.get("model_type")
        if not isinstance(config_model_type, str) or not config_model_type.strip():
            raise SafetyError("Hugging Face config omitted model_type")
        is_encoder_decoder = config.get("is_encoder_decoder")
        if is_encoder_decoder is not None and not isinstance(is_encoder_decoder, bool):
            raise SafetyError("Hugging Face config has an invalid is_encoder_decoder")
        try:
            recognised_config = AutoConfig.from_pretrained(
                config_path.parent, local_files_only=True, trust_remote_code=False
            )
        except Exception as error:
            raise SafetyError(
                "model is not compatible with the installed Transformers stack"
            ) from error
        if getattr(recognised_config, "model_type", None) != config_model_type:
            raise SafetyError("Hugging Face config disagrees with AutoConfig")
        model_type = derive_model_type(
            pipeline_tag=pipeline_tag, is_encoder_decoder=is_encoder_decoder
        )
        if model_type is None:
            raise SafetyError("Hugging Face metadata has no unambiguous capability")
        if not _installed_backend_supports(
            model_type=model_type,
            config=recognised_config,
            architectures=tuple(architectures),
            pipeline_tag=pipeline_tag,
            is_encoder_decoder=is_encoder_decoder,
        ):
            raise SafetyError(
                "model is not compatible with the installed evaluation backend"
            )
        private = getattr(info, "private", None)
        gated = getattr(info, "gated", None)
        if not isinstance(private, bool) or not _hub_false_or_bool(gated):
            raise SafetyError("Hugging Face response omitted public access metadata")
        return ModelMetadata(
            private=private,
            gated=gated is True,
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
            pipeline_tag=pipeline_tag,
            model_type=model_type,
            is_encoder_decoder=is_encoder_decoder,
            backend_compatible=True,
            config=recognised_config,
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
    if info.model_type is None or info.pipeline_tag is None:
        raise SafetyError("model capability could not be derived")
    evidence = lease.model_metadata
    if evidence is None:
        raise SafetyError("broker omitted immutable model metadata")
    if (
        evidence.pipeline_tag != info.pipeline_tag
        or evidence.architectures != info.architectures
        or evidence.model_type != info.model_type
        or evidence.is_encoder_decoder != info.is_encoder_decoder
    ):
        raise SafetyError("broker model metadata does not match the Hub")
    if info.model_type != lease.model_type:
        raise SafetyError("model capability does not match its broker type")
    scope = lease.expected_scope
    if scope is None:
        raise SafetyError("broker omitted expected evaluation scope")
    if not _backend_compatible(info=info, task_groups=scope.task_groups):
        raise SafetyError("model backend compatibility was not verified")
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


def _backend_compatible(info: ModelMetadata, task_groups: tuple[str, ...]) -> bool:
    """Check a fetched model against the leased tasks.

    Returns:
        Whether the installed backend can execute every leased task.
    """
    if info.config is not None:
        return _installed_backend_supports(
            model_type=info.model_type or "",
            config=info.config,
            architectures=info.architectures,
            pipeline_tag=info.pipeline_tag,
            is_encoder_decoder=info.is_encoder_decoder,
            task_groups=task_groups,
        )
    if info.model_type == "encoder":
        return _encoder_task_mappings(task_groups) is not None
    return info.backend_compatible


def derive_model_type(
    *, pipeline_tag: str | None, is_encoder_decoder: object = None
) -> str | None:
    """Classify a model using EuroEval's Hub pipeline/backend contract.

    A non-generative Hub pipeline is sent through the Transformers encoder path;
    generative Hub pipelines are sent through vLLM. Architecture names are not
    classification evidence because they are implementation details, not a
    compatibility contract.

    Returns:
        The broad capability, or ``None`` for ambiguous metadata.
    """
    if not isinstance(pipeline_tag, str) or not pipeline_tag.strip():
        return None
    if is_encoder_decoder is not None and not isinstance(is_encoder_decoder, bool):
        return None
    generative = pipeline_tag in GENERATIVE_PIPELINE_TAGS
    if not generative and is_encoder_decoder is True:
        return None
    return "generative" if generative else "encoder"


def _hub_false_or_bool(value: object) -> bool:
    """Return whether a Hub access flag has a recognised representation."""
    return isinstance(value, bool) or value == "false"


def _installed_backend_supports(
    *,
    model_type: str,
    config: object,
    architectures: tuple[str, ...],
    pipeline_tag: str | None = None,
    is_encoder_decoder: bool | None = None,
    task_groups: tuple[str, ...] = (),
) -> bool:
    """Check actual task loading against the installed backend APIs.

    Returns:
        Whether the installed backend can load the model without remote code.
    """
    if model_type == "encoder":
        if pipeline_tag not in (None, *_ENCODER_PIPELINE_TAGS):
            return False
        if is_encoder_decoder is True:
            return False
        mappings = _encoder_task_mappings(task_groups)
        if mappings is None:
            return False
        try:
            return all(type(config) in mapping for mapping in mappings)
        except Exception:
            return False
    if model_type != "generative" or not architectures:
        return False
    if pipeline_tag == "text2text-generation" and is_encoder_decoder is False:
        return False
    if pipeline_tag == "text-generation" and is_encoder_decoder is True:
        return False
    try:
        registry_module = importlib.import_module("vllm.model_executor.models.registry")
        registry = getattr(registry_module, "ModelRegistry", None)
        get_supported = getattr(registry, "get_supported_archs", None)
        inspect_model = getattr(registry, "inspect_model_cls", None)
        if not callable(get_supported) or not callable(inspect_model):
            return False
        supported = get_supported()
        if not isinstance(supported, c.Collection):
            return False
        candidates = [
            architecture for architecture in architectures if architecture in supported
        ]
        if not candidates:
            return False
        inspection = inspect_model(candidates, _VllmInspectionConfig())
        model_info = inspection[0] if isinstance(inspection, tuple) else inspection
        return bool(
            getattr(model_info, "is_text_generation_model", False)
        ) and not bool(getattr(model_info, "is_pooling_model", False))
    except Exception:
        return False


def _encoder_task_mappings(task_groups: tuple[str, ...]) -> tuple[object, ...] | None:
    """Return the exact Transformers task mappings required by a scope."""
    if not task_groups:
        return _ENCODER_MAPPINGS
    mappings = tuple(_ENCODER_TASK_MAPPINGS.get(task) for task in task_groups)
    if any(mapping is None for mapping in mappings):
        return None
    return mappings


@dataclasses.dataclass(frozen=True)
class _VllmInspectionConfig:
    """Minimal safe configuration for vLLM registry inspection."""

    model_impl: str = "auto"
    convert_type: str = "none"
    runner_type: str = "generate"
    trust_remote_code: bool = False
