"""NVIDIA, Apple Metal and host hardware discovery for worker claims."""

import collections.abc as c
import dataclasses
import functools
import importlib.metadata
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import typing as t
from pathlib import Path

from .types import Gpu, HardwareReport

logger = logging.getLogger(__name__)

METAL_GPU_UUID = "metal-0"


def _run(command: list[str]) -> str:
    """Run.

    Args:
        command:
            Argument vector passed to the subprocess.

    Returns:
        Standard output from the completed command.
    """
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout


def discover_hardware(
    runner: c.Callable[[list[str]], str] | None = None, *, state_dir: Path | None = None
) -> HardwareReport:
    """Discover host facts and fail clearly when no supported GPU is found.

    Args:
        runner (optional):
            Command runner used for hardware discovery.
        state_dir (optional):
            Worker state and model-cache directory. Defaults to the current working
            directory for backwards compatibility.

    Returns:
        A hardware report suitable for the broker claim.
    """
    run = runner or _run
    if _is_apple_silicon():
        gpu = discover_metal_gpu(runner=run)
        return HardwareReport(
            architecture=platform.machine(),
            ram_bytes=_darwin_ram_bytes(runner=run),
            free_disk_bytes=shutil.disk_usage(state_dir or Path.cwd()).free,
            driver_version=None,
            cuda_version=None,
            pytorch_version=_pytorch_version(),
            gpus=(gpu,),
            selected_gpu_index=gpu.index,
            selected_gpu_uuid=gpu.uuid,
        )
    gpus = discover_gpus(runner=run)
    selected = select_gpu(gpus)
    summary = run(["nvidia-smi"])
    driver = _match(summary, r"Driver Version\s*:\s*([^\s]+)")
    cuda = _match(summary, r"CUDA Version\s*:\s*([^\s]+)")
    return HardwareReport(
        architecture=platform.machine(),
        ram_bytes=_ram_bytes(),
        free_disk_bytes=shutil.disk_usage(state_dir or Path.cwd()).free,
        driver_version=driver,
        cuda_version=cuda,
        pytorch_version=_pytorch_version(),
        gpus=tuple(gpus),
        selected_gpu_index=selected.index,
        selected_gpu_uuid=selected.uuid,
    )


def _darwin_ram_bytes(runner: c.Callable[[list[str]], str]) -> int:
    """Read physical RAM on macOS, which does not expose ``SC_PHYS_PAGES``.

    Args:
        runner:
            Command runner used to execute ``sysctl``.

    Returns:
        Physical RAM in bytes, or ``0`` when it cannot be read.
    """
    try:
        return int(runner(["sysctl", "-n", "hw.memsize"]).strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def _is_apple_silicon() -> bool:
    """Is apple silicon.

    Returns:
        Whether the worker runs on macOS on an arm64 Apple Silicon chip.
    """
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _match(text: str, pattern: str) -> str | None:
    """Match.

    Args:
        text:
            Input text to search.
        pattern:
            Regular expression searched in the input text.

    Returns:
        The first capture group, or ``None`` when the pattern does not match.
    """
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _pytorch_version() -> str | None:
    """Pytorch version.

    Returns:
        Installed PyTorch version, or ``None`` when PyTorch is not installed.
    """
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def _ram_bytes() -> int:
    """Ram bytes.

    Returns:
        Physical RAM in bytes, or ``0`` when system page information is unavailable.
    """
    if hasattr(os, "sysconf"):
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    return 0


def discover_gpus(runner: c.Callable[[list[str]], str] | None = None) -> list[Gpu]:
    """Parse nvidia-smi CSV output, with an older-driver fallback.

    Args:
            runner (optional):
                Callable used to execute nvidia-smi queries.

    Returns:
            Discovered GPUs.

    Raises:
            NoGpuError:
                If nvidia-smi is unavailable or returns no GPUs.
    """
    run = runner or _run
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.free,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = run(query)
    except (OSError, subprocess.CalledProcessError):
        fallback = query.copy()
        fallback[1] = "--query-gpu=index,name,uuid,memory.free,memory.total"
        try:
            output = run(fallback)
        except (OSError, subprocess.CalledProcessError) as error:
            raise NoGpuError(
                "nvidia-smi found no NVIDIA GPU; install a driver or run this worker "
                "on a GPU host before claiming work"
            ) from error
    gpus = _parse_gpu_csv(output)
    gpus = [
        _discover_spark_memory(gpu) if _needs_spark_memory_fallback(gpu) else gpu
        for gpu in gpus
    ]
    if not gpus:
        raise NoGpuError(
            "nvidia-smi returned no GPUs; check the NVIDIA driver and container "
            "GPU access"
        )
    return gpus


class NoGpuError(RuntimeError):
    """Raised when no usable NVIDIA GPU is available."""


def _discover_spark_memory(gpu: Gpu) -> Gpu:
    """Discover spark memory.

    Args:
            gpu:
                Gpu.

    Returns:
            GPU report with verified memory, or zeros if verification fails.

    Raises:
    ValueError:
            If the input does not have the required structure or value.
            The input is malformed or cannot be converted to the required value.
    """
    script = (
        "import json, torch; "
        "p=torch.cuda.get_device_properties(0); "
        "u=getattr(p, 'uuid', None); "
        "f,t=torch.cuda.mem_get_info(0); "
        "print(json.dumps({'uuid': str(u) if u is not None else None, "
        "'free': f, 'total': t}))"
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu.uuid
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
            env=environment,
        )
        measurement = json.loads(result.stdout)
        free = int(measurement["free"])
        total = int(measurement["total"])
        reported_uuid = measurement["uuid"]
        host_available = _host_available_bytes()
        if (
            not isinstance(reported_uuid, str)
            or reported_uuid.lower().removeprefix("gpu-")
            != gpu.uuid.lower().removeprefix("gpu-")
            or free < 0
            or total <= 0
            or free > total
            or host_available is None
        ):
            raise ValueError("Unverified unified-memory measurement")
        usable_free = unified_free_bytes(
            device_free_bytes=free, host_available_bytes=host_available
        )
        return dataclasses.replace(
            gpu, free_memory_bytes=usable_free, total_memory_bytes=total
        )
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        logger.warning("Could not verify GB10 unified-memory availability")
        return dataclasses.replace(gpu, free_memory_bytes=0, total_memory_bytes=0)


def _host_available_bytes() -> int | None:
    """Host available bytes.

    Returns:
        Available host memory in bytes, or ``None`` when it cannot be read.
    """
    if sys.platform == "darwin":
        return _darwin_available_bytes(runner=_run)
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _darwin_available_bytes(runner: c.Callable[[list[str]], str]) -> int | None:
    """Estimate available memory on macOS from ``vm_stat``.

    Free plus inactive pages is an estimate. ``vm_stat``'s free count excludes
    speculative pages, which are also reclaimable, but reclaiming some inactive pages
    can need compression or swap.

    Args:
        runner:
            Command runner used to execute ``vm_stat``.

    Returns:
        Estimated available memory in bytes, or ``None`` when it cannot be read.
    """
    try:
        output = runner(["vm_stat"])
    except (OSError, subprocess.SubprocessError):
        return None
    page = _match(output, r"page size of (\d+) bytes")
    free = _match(output, r"Pages free:\s+(\d+)")
    inactive = _match(output, r"Pages inactive:\s+(\d+)")
    if page is None or free is None or inactive is None:
        return None
    return (int(free) + int(inactive)) * int(page)


def unified_free_bytes(device_free_bytes: int, host_available_bytes: int) -> int:
    """Cap a unified-memory device's free memory by what the host can spare.

    The CPU and GPU share the same physical memory on unified-memory devices, so the
    device's own free figure must not exceed available host memory minus 1 GiB of
    headroom for the operating system and the evaluation process.

    Args:
        device_free_bytes:
            Memory the device reports as usable.
        host_available_bytes:
            Memory the host reports as available.

    Returns:
        The usable inference memory in bytes, never negative.
    """
    return min(device_free_bytes, max(0, host_available_bytes - 1024**3))


def _needs_spark_memory_fallback(gpu: Gpu) -> bool:
    """Needs spark memory fallback.

    Args:
        gpu:
            GPU whose unified-memory measurement is being verified.

    Returns:
        Whether this GB10 GPU has missing free or total memory data.
    """
    return "gb10" in gpu.name.lower() and (
        gpu.free_memory_bytes == 0 or gpu.total_memory_bytes == 0
    )


def _parse_gpu_csv(output: str) -> list[Gpu]:
    """Parse gpu csv.

    Args:
        output:
            Raw CSV text returned by nvidia-smi.

    Returns:
        GPU reports parsed from valid nvidia-smi CSV rows.
    """
    gpus: list[Gpu] = []
    for row_index, line in enumerate(output.splitlines()):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) not in (4, 5, 6):
            continue
        try:
            if len(fields) in (5, 6) and fields[0].isdigit():
                index = int(fields[0])
                offset = 1
            else:
                index = row_index
                offset = 0
            expected = 5 if len(fields) - offset == 5 else 4
            if len(fields) - offset != expected:
                continue
            capability = (
                fields[offset + 4] if expected == 5 and fields[offset + 4] else None
            )
            gpus.append(
                Gpu(
                    name=fields[offset],
                    uuid=fields[offset + 1],
                    free_memory_bytes=_memory_value(fields[offset + 2]),
                    total_memory_bytes=_memory_value(fields[offset + 3]),
                    compute_capability=capability,
                    index=index,
                )
            )
        except ValueError:
            logger.warning("Ignoring malformed nvidia-smi row")
    return gpus


def _memory_value(value: str) -> int:
    """Memory value.

    Args:
        value:
            Input value to validate or convert.

    Returns:
        Memory amount converted from MiB to bytes; N/A values become ``0``.
    """
    if value.strip().lower() in {"n/a", "na", "[n/a]"}:
        return 0
    return int(float(value) * 1024**2)


def discover_metal_gpu(runner: c.Callable[[list[str]], str] | None = None) -> Gpu:
    """Report the Apple GPU and its unified-memory inference budget.

    The total is MLX's recommended working-set limit, which vllm-metal uses as the
    denominator for ``gpu_memory_utilization``. A subprocess also checks the
    prerequisites for evaluating on the GPU rather than the CPU: vLLM selects its Metal
    platform only when vllm-metal's ``metal`` plugin is registered and MLX can use
    Metal, and Transformers (encoder models) needs PyTorch's MPS backend. The plugin
    check reads entry-point metadata and does not load the plugin. The checks run in a
    subprocess so that MLX and PyTorch are never initialised in the worker. That
    subprocess runs once per worker process, because its results cannot change while
    the worker runs; available memory and the vLLM environment are checked every time.

    Args:
        runner (optional):
            Command runner used for ``sysctl`` and ``vm_stat``.

    Returns:
        A single GPU entry with identifier ``metal-0``.

    Raises:
        NoGpuError:
            If the Metal device, the GPU backends or a memory measurement cannot be
            verified.
    """
    run = runner or _run
    probe = _probe_metal()
    cap = probe.cap
    backends = {
        "metal": probe.metal,
        "vllm_metal_plugin": probe.vllm_metal_plugin,
        "mps": probe.mps,
    }
    if not backends["metal"]:
        raise NoGpuError(
            "MLX cannot use the Apple GPU, so evaluation would run on the CPU"
        )
    if not backends["vllm_metal_plugin"]:
        raise NoGpuError(
            "vllm-metal's platform plugin is not installed, so vLLM would run on the "
            "CPU; run `make install`, which uses Python 3.12"
        )
    # vLLM honours these before probing plugins (vllm/platforms/__init__.py and
    # vllm/plugins/__init__.py in the pinned version), and evaluation inherits them.
    if os.environ.get("VLLM_TARGET_DEVICE", "").lower() == "cpu":
        raise NoGpuError(
            "VLLM_TARGET_DEVICE is set to cpu, so vLLM would run on the CPU; unset it "
            "to use the Apple GPU"
        )
    if "VLLM_PLUGINS" in os.environ and "metal" not in os.environ["VLLM_PLUGINS"].split(
        ","
    ):
        raise NoGpuError(
            "VLLM_PLUGINS does not include metal, so vLLM would run on the CPU; unset "
            "it or add metal"
        )
    if not backends["mps"]:
        raise NoGpuError(
            "PyTorch cannot use the Apple GPU (MPS), so encoder models would run on "
            "the CPU"
        )
    available = _darwin_available_bytes(runner=run)
    if cap <= 0 or available is None:
        raise NoGpuError("could not verify Apple Silicon unified-memory availability")
    try:
        name = run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or "Apple GPU"
    except (OSError, subprocess.SubprocessError):
        name = "Apple GPU"
    return Gpu(
        name=name,
        uuid=METAL_GPU_UUID,
        free_memory_bytes=unified_free_bytes(
            device_free_bytes=cap, host_available_bytes=available
        ),
        total_memory_bytes=cap,
        compute_capability=None,
        index=0,
    )


class _MetalProbe(t.NamedTuple):
    """Static Apple GPU facts read by the probe subprocess."""

    cap: int
    metal: bool
    vllm_metal_plugin: bool
    mps: bool


@functools.cache
def _probe_metal() -> _MetalProbe:
    """Read MLX's working-set limit and the GPU backends, once per process.

    Only a successful probe is cached; a failed one is retried on the next call.

    Returns:
        The working-set limit and whether each GPU backend is available.

    Raises:
        NoGpuError:
            If the probe subprocess fails or prints unexpected output.
    """
    script = (
        "import importlib.metadata as m, json, mlx.core as mx, torch; "
        "plugins = m.entry_points(group='vllm.platform_plugins'); "
        "print(json.dumps({"
        "'cap': int(mx.device_info()['max_recommended_working_set_size']), "
        "'metal': bool(mx.metal.is_available()), "
        "'vllm_metal_plugin': any(p.name == 'metal' for p in plugins), "
        "'mps': bool(torch.backends.mps.is_available())}))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        probe = json.loads(result.stdout)
        return _MetalProbe(
            cap=int(probe["cap"]),
            metal=bool(probe["metal"]),
            vllm_metal_plugin=bool(probe["vllm_metal_plugin"]),
            mps=bool(probe["mps"]),
        )
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        raise NoGpuError(
            "could not probe the Apple GPU's Metal working-set limit and backends; "
            "run `make install` so that vllm-metal, MLX and PyTorch are installed"
        ) from error


def select_gpu(gpus: c.Iterable[Gpu]) -> Gpu:
    """Select the GPU with the most usable free memory deterministically.

    Args:
            gpus:
                Gpus.

    Returns:
            The selected GPU.

    Raises:
            NoGpuError:
                If no GPU has a usable memory report.
    """
    usable = [
        gpu for gpu in gpus if 0 < gpu.free_memory_bytes <= gpu.total_memory_bytes
    ]
    if not usable:
        raise NoGpuError("nvidia-smi returned no GPU with usable free memory")
    return min(usable, key=lambda gpu: (-gpu.free_memory_bytes, gpu.index, gpu.uuid))
