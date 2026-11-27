"""GPU selection and model-fit calculations for worker hardware."""

import collections.abc as c
import math
import typing as t


def fits_gpu(
    model: c.Mapping[str, object],
    hardware: c.Mapping[str, object],
    selected: c.Mapping[str, object] | None = None,
) -> bool:
    """Determine whether model weights fit available GPU memory and disk.

    Args:
        model: Model metadata containing weight_bytes and repo_bytes.
        hardware: Worker hardware inventory, capacities, and optional selection.
        selected: Previously resolved GPU selection, if supplied.

    Returns:
        Whether disk and GPU memory requirements are satisfied.
    """
    disk = hardware.get("free_disk_bytes")
    repo_bytes = model.get("repo_bytes")
    utilisation = hardware.get("gpu_memory_utilisation", 1)
    if not _finite_number(disk) or not _finite_number(repo_bytes) or disk < repo_bytes:
        return False
    if not _finite_number(utilisation) or not 0 < utilisation <= 1:
        return False
    gpus = hardware.get("gpus")
    if not isinstance(gpus, list):
        return False

    has_selection = "selected_gpu_index" in hardware or "selected_gpu_uuid" in hardware
    gpu = (
        selected
        if selected is not None
        else (selected_gpu(hardware) if has_selection else None)
    )
    if has_selection and gpu is None:
        return False
    candidates = [gpu] if gpu is not None else gpus
    largest_free = max(
        (
            free
            for item in candidates
            if isinstance(item, c.Mapping)
            if _finite_number(free := item.get("free_memory_bytes")) and free > 0
        ),
        default=0,
    )
    weight_bytes = model.get("weight_bytes")
    return (
        _finite_number(weight_bytes)
        and weight_bytes * 1.35 <= largest_free * utilisation
    )


def _finite_number(value: object) -> bool:
    """Return whether a value is a finite number, excluding booleans."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def selected_gpu(hardware: c.Mapping[str, object]) -> dict[str, object] | None:
    """Resolve the selected GPU only when its index and UUID match.

    Args:
        hardware: Worker hardware data, including its GPU inventory and selection.

    Returns:
        The selected GPU mapping, or ``None`` when selection is absent or invalid.
    """
    gpus = hardware.get("gpus")
    index = hardware.get("selected_gpu_index")
    uuid = hardware.get("selected_gpu_uuid")
    if (
        not isinstance(gpus, list)
        or not _safe_index(index)
        or not isinstance(uuid, str)
        or not uuid
    ):
        return None

    items = [gpu for gpu in gpus if isinstance(gpu, dict)]
    has_indexes = any("index" in gpu for gpu in items)
    gpu = (
        next((item for item in items if item.get("index") == index), None)
        if has_indexes
        else (gpus[t.cast(int, index)] if index < len(gpus) else None)
    )
    if not isinstance(gpu, dict) or gpu.get("uuid") != uuid:
        return None
    return gpu


def _safe_index(value: object) -> bool:
    """Return whether a value is a nonnegative JavaScript-safe integer."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        and value.is_integer()
        if isinstance(value, float)
        else isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= 2**53 - 1
    )


def valid_hardware(value: object) -> bool:
    """Validate worker hardware before accepting an evaluation claim.

    Args:
        value:
            Untrusted hardware value to validate.

    Returns:
        Whether the value has valid hardware fields and a matching GPU selection.
    """
    if not isinstance(value, dict):
        return False

    utilisation = value.get("gpu_memory_utilisation")
    architecture = value.get("architecture")
    free_disk = value.get("free_disk_bytes")
    selected_index = value.get("selected_gpu_index")
    selected_uuid = value.get("selected_gpu_uuid")
    gpus = value.get("gpus")
    if (
        not _finite_number(utilisation)
        or not 0 < utilisation <= 1
        or not isinstance(architecture, str)
        or not architecture.strip()
        or not _safe_claim_integer(free_disk)
        or not _safe_claim_integer(selected_index)
        or not isinstance(selected_uuid, str)
        or not selected_uuid
        or not isinstance(gpus, list)
        or not gpus
    ):
        return False

    for gpu in gpus:
        if not isinstance(gpu, dict):
            return False
        free_memory = gpu.get("free_memory_bytes")
        total_memory = gpu.get("total_memory_bytes")
        if (
            not isinstance(gpu.get("name"), str)
            or not isinstance(gpu.get("uuid"), str)
            or not _safe_claim_integer(free_memory)
            or not _safe_claim_integer(total_memory, minimum=1)
            or t.cast(int | float, free_memory) > t.cast(int | float, total_memory)
            or ("index" in gpu and not _safe_claim_integer(gpu["index"]))
        ):
            return False

    return selected_gpu(value) is not None


def _safe_claim_integer(value: object, minimum: int = 0) -> bool:
    """Return whether a value is a safe integral number at least the minimum.

    Args:
        value:
            Value to check.
        minimum:
            Inclusive lower bound.

    Returns:
        Whether the value is a JavaScript-safe integer.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and minimum <= value <= 2**53 - 1
        and value.is_integer()
        if isinstance(value, float)
        else isinstance(value, int)
        and not isinstance(value, bool)
        and minimum <= value <= 2**53 - 1
    )
