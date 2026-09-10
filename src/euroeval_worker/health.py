"""Credential-free GPU health checks for physical worker canaries."""

import collections.abc as c
import dataclasses
import importlib
import os
import typing as t

from .hardware import discover_gpus, select_gpu
from .types import Gpu


class _CudaProperties(t.Protocol):
    uuid: object


class _Cuda(t.Protocol):
    def current_device(self) -> int: ...

    def device_count(self) -> int: ...

    def get_device_properties(self, device: int) -> _CudaProperties: ...

    def is_available(self) -> bool: ...

    def set_device(self, device: int) -> None: ...

    def synchronize(self) -> None: ...


class _Tensor(t.Protocol):
    def __getitem__(self, key: tuple[int, int]) -> "_Tensor": ...

    def __matmul__(self, other: "_Tensor") -> "_Tensor": ...

    def item(self) -> float: ...


class _Torch(t.Protocol):
    __version__: str
    cuda: _Cuda
    float32: object

    def ones(self, size: tuple[int, int], *, device: str, dtype: object) -> _Tensor: ...


class _Vllm(t.Protocol):
    __version__: str


@dataclasses.dataclass(frozen=True)
class GpuHealthReport:
    """Safe metadata emitted after a successful GPU health check."""

    gpu_name: str
    gpu_index: int
    gpu_uuid: str
    torch_version: str
    vllm_version: str

    def summary(self) -> str:
        """Return concise, non-secret health metadata."""
        return (
            f"gpu={self.gpu_name!r} index={self.gpu_index} uuid={self.gpu_uuid} "
            f"torch={self.torch_version} vllm={self.vllm_version} operation=matmul"
        )


def run_gpu_health_check(
    discoverer: c.Callable[[], list[Gpu]] = discover_gpus,
    selector: c.Callable[[c.Iterable[Gpu]], Gpu] = select_gpu,
    importer: c.Callable[[str], object] = importlib.import_module,
) -> GpuHealthReport:
    """Discover, pin, import, and exercise one worker GPU.

    Args:
        discoverer (optional):
            NVIDIA hardware discovery function. Defaults to ``discover_gpus``.
        selector (optional):
            Deterministic GPU selector. Defaults to ``select_gpu``.
        importer (optional):
            Module importer, primarily useful for isolated tests. Defaults to
            :func:`importlib.import_module`.

    Returns:
        Safe metadata describing the successful check.
    """
    selected = selector(discoverer())
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = selected.uuid
    try:
        torch = _import_stack_module(importer=importer, name="torch")
        vllm = _import_stack_module(importer=importer, name="vllm")
        torch_module = t.cast(_Torch, torch)
        vllm_module = t.cast(_Vllm, vllm)
        _check_cuda(torch_module=torch_module, selected=selected)
        return GpuHealthReport(
            gpu_name=selected.name,
            gpu_index=selected.index,
            gpu_uuid=selected.uuid,
            torch_version=torch_module.__version__,
            vllm_version=getattr(vllm_module, "__version__", "unknown"),
        )
    finally:
        if original is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original


def _check_cuda(torch_module: _Torch, selected: Gpu) -> None:
    """Validate the pinned CUDA device and execute a small tensor operation.

    Raises:
        GpuHealthError:
            If CUDA is unavailable, not isolated, or produces a bad result.
    """
    cuda = torch_module.cuda
    if not cuda.is_available():
        raise GpuHealthError(
            "PyTorch cannot access CUDA; check --gpus all and NVIDIA Container Toolkit"
        )
    if cuda.device_count() != 1:
        raise GpuHealthError(
            "CUDA_VISIBLE_DEVICES did not isolate the selected GPU; check NVIDIA "
            "container runtime passthrough"
        )
    cuda.set_device(0)
    if cuda.current_device() != 0:
        raise GpuHealthError(
            "CUDA selected a non-zero logical device after GPU pinning"
        )
    properties = cuda.get_device_properties(0)
    device_uuid = getattr(properties, "uuid", None)
    if device_uuid is not None and _normalise_uuid(device_uuid) != selected.uuid:
        raise GpuHealthError(
            f"CUDA UUID {_normalise_uuid(device_uuid)} does not match nvidia-smi "
            f"UUID {selected.uuid}"
        )
    tensor = torch_module.ones((2, 2), device="cuda", dtype=torch_module.float32)
    result = tensor @ tensor
    cuda.synchronize()
    if result[0, 0].item() != 2.0:
        raise GpuHealthError("CUDA tensor operation returned an unexpected result")


class GpuHealthError(RuntimeError):
    """Raised when the image cannot execute a CUDA health check."""


def _normalise_uuid(value: object) -> str:
    """Convert CUDA's UUID representation to the nvidia-smi text form.

    Returns:
        The UUID as text.
    """
    if isinstance(value, bytes):
        return value.decode("ascii")
    return str(value)


def _import_stack_module(importer: c.Callable[[str], object], name: str) -> object:
    """Import one required runtime package with an actionable error.

    Returns:
        The imported module.

    Raises:
        GpuHealthError:
            If the package cannot be imported.
    """
    try:
        return importer(name)
    except Exception as error:  # noqa: BLE001 - import errors vary by backend
        raise GpuHealthError(
            f"could not import {name}; check the image's CUDA/vLLM installation: "
            f"{error}"
        ) from error
