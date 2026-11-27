"""Public behavior tests for GPU selection and fit decisions."""

import pytest

from euroeval_api.hardware import fits_gpu, selected_gpu


@pytest.mark.parametrize(
    ("disk", "utilisation", "free", "expected"),
    [
        (10, 1, 135, True),
        (9, 1, 1000, False),
        (10, 0, 1000, False),
        (10, 1.01, 1000, False),
        (10, 1, 134, False),
    ],
)
def test_fits_gpu_checks_disk_utilisation_and_vram_boundaries(
    disk: int, utilisation: float, free: int, expected: bool
) -> None:
    """Apply disk, utilisation, and 1.35x memory requirements."""
    model = {"weight_bytes": 100, "repo_bytes": 10}
    hardware = {
        "free_disk_bytes": disk,
        "gpu_memory_utilisation": utilisation,
        "gpus": [{"free_memory_bytes": free}],
    }

    assert fits_gpu(model, hardware) is expected


@pytest.mark.parametrize("index", [1, 1.0])
def test_selected_gpu_binds_explicit_index_and_uuid(index: int | float) -> None:
    """Resolve an explicitly indexed GPU only with its matching UUID."""
    chosen = {"index": 1, "uuid": "gpu-b", "free_memory_bytes": 200}
    hardware = {
        "gpus": [{"index": 0, "uuid": "gpu-a"}, chosen],
        "selected_gpu_index": index,
        "selected_gpu_uuid": "gpu-b",
    }

    assert selected_gpu(hardware) is chosen
    assert selected_gpu({**hardware, "selected_gpu_uuid": "stale"}) is None
    assert not fits_gpu(
        {"weight_bytes": 100, "repo_bytes": 0},
        {**hardware, "selected_gpu_uuid": "stale", "free_disk_bytes": 0},
    )


def test_unselected_gpu_uses_largest_available_memory() -> None:
    """Use the largest GPU when no GPU selection was requested."""
    hardware = {
        "free_disk_bytes": 1,
        "gpus": [{"free_memory_bytes": 20}, {"free_memory_bytes": 140}],
    }

    assert fits_gpu({"weight_bytes": 100, "repo_bytes": 1}, hardware)
