"""Tests for the credential-free worker GPU health check."""

import os
import types

import pytest

from euroeval_worker import broker, cli, health
from euroeval_worker.hardware import NoGpuError
from euroeval_worker.types import Gpu

GPU = Gpu(
    name="Test GPU",
    uuid="GPU-test",
    free_memory_bytes=2_000,
    total_memory_bytes=4_000,
    compute_capability="9.0",
    index=2,
)


def test_cli_health_check_does_not_construct_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The health mode completes without broker access or credentials.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    report = health.GpuHealthReport(
        gpu_name=GPU.name,
        gpu_index=GPU.index,
        gpu_uuid=GPU.uuid,
        torch_version="test-torch",
        vllm_version="test-vllm",
    )
    monkeypatch.setattr(health, "run_gpu_health_check", lambda: report)

    def forbidden_broker(*args: object, **kwargs: object) -> object:
        """Forbidden broker for this test.

        Args:
            args:
                Args supplied to the test.
            kwargs:
                Kwargs supplied to the test.

        Raises:
            AssertionError:
                If the simulated operation fails.
        """
        del args, kwargs
        raise AssertionError("health mode must not construct a broker")

    monkeypatch.setattr(broker, "BrokerClient", forbidden_broker)
    assert cli.main(["--gpu-health-check"]) == 0


def test_gpu_health_check_fails_when_cuda_is_unavailable() -> None:
    """Report an actionable error when PyTorch cannot access CUDA."""
    modules = {"torch": FakeTorch(available=False), "vllm": object()}

    def importer(name: str) -> object:
        """Importer for this test.

        Args:
            name:
                Name supplied to the test.

        Returns:
            The value returned by the test helper.
        """
        return modules[name]

    with pytest.raises(health.GpuHealthError, match="cannot access CUDA"):
        health.run_gpu_health_check(discoverer=lambda: [GPU], importer=importer)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


class FakeCuda:
    """CUDA API fake with configurable availability."""

    def __init__(self, available: bool) -> None:
        """Initialise the fake CUDA availability state.

        Args:
            available:
                Available supplied to the test.
        """
        self.available = available
        self.uuid = GPU.uuid
        self.set_device_calls: list[int] = []

    def current_device(self) -> int:
        """Return the selected logical device.

        Returns:
            The current logical CUDA device index.
        """
        return 0

    def device_count(self) -> int:
        """Return the number of visible devices.

        Returns:
            The number of visible logical CUDA devices.
        """
        return 1

    def get_device_properties(self, device: int) -> types.SimpleNamespace:
        """Return properties for the selected logical device.

        Args:
            device:
                Logical CUDA device whose properties are requested.

        Returns:
            Properties containing the fake device UUID.
        """
        assert device == 0
        return types.SimpleNamespace(uuid=self.uuid)

    def is_available(self) -> bool:
        """Report whether CUDA is available.

        Returns:
            Whether the fake CUDA runtime is available.
        """
        return self.available

    def set_device(self, device: int) -> None:
        """Record the selected logical device.

        Args:
            device:
                Device supplied to the test.
        """
        self.set_device_calls.append(device)

    def synchronize(self) -> None:
        """Synchronise the fake CUDA stream.

        Returns:
            None.
        """
        return None


class FakeTensor:
    """Small tensor fake supporting the health-check operation."""

    def __getitem__(self, key: tuple[int, int]) -> "FakeTensor":
        """Return the scalar view used by the assertion.

        Args:
            key:
                Tuple selecting the scalar tensor element.

        Returns:
            This fake tensor, representing the selected view.
        """
        del key
        return self

    def __matmul__(self, other: "FakeTensor") -> "FakeTensor":
        """Return the result of the fake matrix multiplication.

        Args:
            other:
                Tensor operand in the fake matrix multiplication.

        Returns:
            This fake tensor as the deterministic multiplication result.
        """
        del other
        return self

    def item(self) -> float:
        """Return the deterministic scalar result.

        Returns:
            The scalar value produced by the fake tensor.
        """
        return 2.0


class FakeTorch:
    """PyTorch API fake used by the health-check tests."""

    __version__ = "test-torch"
    float32 = object()

    def __init__(self, available: bool) -> None:
        """Initialise the fake torch module.

        Args:
            available:
                Available supplied to the test.
        """
        self.cuda = FakeCuda(available=available)

    def ones(self, size: tuple[int, int], *, device: str, dtype: object) -> FakeTensor:
        """Allocate a fake CUDA tensor.

        Args:
            size:
                Size supplied to the test.
            device:
                Device supplied to the test.
            dtype:
                Dtype supplied to the test.

        Returns:
            A fake tensor on the requested device.
        """
        assert size == (2, 2)
        assert device == "cuda"
        assert dtype is self.float32
        return FakeTensor()


def test_gpu_health_check_fails_without_gpu() -> None:
    """Do not import runtime packages when hardware discovery fails."""
    imported = False

    def importer(name: str) -> object:
        """Importer for this test.

        Args:
            name:
                Name supplied to the test.

        Returns:
            The value returned by the test helper.
        """
        nonlocal imported
        del name
        imported = True
        return object()

    def discoverer() -> list[Gpu]:
        """Simulate discovery failure because no GPU is available.

        Raises:
            NoGpuError:
                Always, to simulate a machine without an NVIDIA GPU.
        """
        raise NoGpuError("no NVIDIA GPU")

    with pytest.raises(NoGpuError, match="no NVIDIA GPU"):
        health.run_gpu_health_check(discoverer=discoverer, importer=importer)
    assert not imported


@pytest.mark.parametrize("cuda_uuid", ["GPU-test", "test"])
def test_gpu_health_check_imports_after_pinning_and_runs_operation(
    cuda_uuid: str,
) -> None:
    """Pin the selected UUID before importing torch and vLLM.

    Args:
        cuda_uuid:
            Cuda uuid supplied to the test.
    """
    imported: list[str] = []
    torch = FakeTorch(available=True)
    torch.cuda.uuid = cuda_uuid
    vllm = types.SimpleNamespace(__version__="test-vllm")
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "original"

    def importer(name: str) -> object:
        """Importer for this test.

        Args:
            name:
                Name supplied to the test.

        Returns:
            The value returned by the test helper.
        """
        assert os.environ["CUDA_VISIBLE_DEVICES"] == GPU.uuid
        imported.append(name)
        return {"torch": torch, "vllm": vllm}[name]

    try:
        report = health.run_gpu_health_check(
            importer=importer, discoverer=lambda: [GPU]
        )
    finally:
        if original is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original

    assert imported == ["torch", "vllm"]
    assert torch.cuda.set_device_calls == [0]
    assert report.gpu_uuid == GPU.uuid
    assert report.vllm_version == "test-vllm"
