"""NVIDIA and host hardware discovery for worker claims."""

import collections.abc as c
import logging
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

import torch

from .types import Gpu, HardwareReport

logger = logging.getLogger(__name__)


class NoGpuError(RuntimeError):
    """Raised when no usable NVIDIA GPU is available."""


def discover_hardware(
    runner: c.Callable[[list[str]], str] | None = None,
) -> HardwareReport:
    """Discover host facts and fail clearly when nvidia-smi finds no GPU.

    Returns:
        A hardware report suitable for the broker claim.
    """
    run = runner or _run
    gpus = discover_gpus(runner=run)
    summary = run(["nvidia-smi"])
    driver = _match(summary, r"Driver Version\s*:\s*([^\s]+)")
    cuda = _match(summary, r"CUDA Version\s*:\s*([^\s]+)")
    return HardwareReport(
        architecture=platform.machine(),
        ram_bytes=_ram_bytes(),
        free_disk_bytes=shutil.disk_usage(Path.cwd()).free,
        driver_version=driver,
        cuda_version=cuda,
        pytorch_version=torch.__version__,
        gpus=tuple(gpus),
    )


def discover_gpus(runner: c.Callable[[list[str]], str] | None = None) -> list[Gpu]:
    """Parse nvidia-smi CSV output, with an older-driver fallback.

    Returns:
        Discovered GPUs.

    Raises:
        NoGpuError:
            If nvidia-smi is unavailable or returns no GPUs.
    """
    run = runner or _run
    query = [
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.free,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = run(query)
    except (OSError, subprocess.CalledProcessError):
        fallback = query.copy()
        fallback[1] = "--query-gpu=name,uuid,memory.free,memory.total"
        try:
            output = run(fallback)
        except (OSError, subprocess.CalledProcessError) as error:
            raise NoGpuError(
                "nvidia-smi found no NVIDIA GPU; install a driver or run this worker "
                "on a GPU host before claiming work"
            ) from error
    gpus = _parse_gpu_csv(output)
    if not gpus:
        raise NoGpuError(
            "nvidia-smi returned no GPUs; check the NVIDIA driver and container "
            "GPU access"
        )
    return gpus


def _parse_gpu_csv(output: str) -> list[Gpu]:
    gpus: list[Gpu] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) not in (4, 5):
            continue
        try:
            capability = fields[4] if len(fields) == 5 and fields[4] else None
            gpus.append(
                Gpu(
                    name=fields[0],
                    uuid=fields[1],
                    free_memory_bytes=int(float(fields[2]) * 1024**2),
                    total_memory_bytes=int(float(fields[3]) * 1024**2),
                    compute_capability=capability,
                )
            )
        except ValueError:
            logger.warning("Ignoring malformed nvidia-smi row")
    return gpus


def _run(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout


def _match(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _ram_bytes() -> int:
    if hasattr(os, "sysconf"):
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    return 0
