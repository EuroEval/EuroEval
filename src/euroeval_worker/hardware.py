"""NVIDIA and host hardware discovery for worker claims."""

import collections.abc as c
import dataclasses
import importlib.metadata
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .types import Gpu, HardwareReport

logger = logging.getLogger(__name__)


def _run(command: list[str]) -> str:
    """Run.

    Args:
        command:
            Command.

    Returns:
        The resulting value.
    """
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout


def discover_hardware(
    runner: c.Callable[[list[str]], str] | None = None, *, state_dir: Path | None = None
) -> HardwareReport:
    """Discover host facts and fail clearly when nvidia-smi finds no GPU.

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


def _match(text: str, pattern: str) -> str | None:
    """Match.

    Args:
        text:
            Text.
        pattern:
            Pattern.

    Returns:
        The resulting value.
    """
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _pytorch_version() -> str | None:
    """Pytorch version.

    Returns:
        The resulting value.
    """
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def _ram_bytes() -> int:
    """Ram bytes.

    Returns:
        The resulting value.
    """
    if hasattr(os, "sysconf"):
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    return 0


def discover_gpus(runner: c.Callable[[list[str]], str] | None = None) -> list[Gpu]:
    """Parse nvidia-smi CSV output, with an older-driver fallback.

    Returns:
        Discovered GPUs.

    Raises:
        NoGpuError:
            If nvidia-smi is unavailable or returns no GPUs.

    Args:
        runner (optional):
            Runner.
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
        The resulting value.

    Raises:
        An exception when the operation cannot be completed.
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
        usable_free = min(free, max(0, host_available - 1024**3))
        return dataclasses.replace(
            gpu, free_memory_bytes=usable_free, total_memory_bytes=total
        )
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        logger.warning("Could not verify GB10 unified-memory availability")
        return dataclasses.replace(gpu, free_memory_bytes=0, total_memory_bytes=0)


def _host_available_bytes() -> int | None:
    """Host available bytes.

    Returns:
        The resulting value.
    """
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _needs_spark_memory_fallback(gpu: Gpu) -> bool:
    """Needs spark memory fallback.

    Args:
        gpu:
            Gpu.

    Returns:
        The resulting value.
    """
    return "gb10" in gpu.name.lower() and (
        gpu.free_memory_bytes == 0 or gpu.total_memory_bytes == 0
    )


def _parse_gpu_csv(output: str) -> list[Gpu]:
    """Parse gpu csv.

    Args:
        output:
            Output.

    Returns:
        The resulting value.
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
            Value.

    Returns:
        The resulting value.
    """
    if value.strip().lower() in {"n/a", "na", "[n/a]"}:
        return 0
    return int(float(value) * 1024**2)


def select_gpu(gpus: c.Iterable[Gpu]) -> Gpu:
    """Select the GPU with the most usable free memory deterministically.

    Returns:
        The selected GPU.

    Raises:
        NoGpuError:
            If no GPU has a usable memory report.

    Args:
        gpus:
            Gpus.
    """
    usable = [
        gpu for gpu in gpus if 0 < gpu.free_memory_bytes <= gpu.total_memory_bytes
    ]
    if not usable:
        raise NoGpuError("nvidia-smi returned no GPU with usable free memory")
    return min(usable, key=lambda gpu: (-gpu.free_memory_bytes, gpu.index, gpu.uuid))
