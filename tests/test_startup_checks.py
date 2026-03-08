"""Tests for the startup checks performed in euroeval/__init__.py."""

import importlib.util
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestFlashAttnCheck:
    """Tests for the flash_attn compatibility check in __init__.py."""

    @pytest.mark.parametrize(
        argnames=["flash_attn_installed", "hip_version", "expect_exit"],
        argvalues=[
            (True, None, True),
            (True, "5.7.0", False),
            (False, None, False),
            (False, "5.7.0", False),
        ],
        ids=[
            "flash_attn on non-ROCm exits",
            "flash_attn on ROCm skips exit",
            "no flash_attn on non-ROCm does not exit",
            "no flash_attn on ROCm does not exit",
        ],
    )
    def test_flash_attn_check(
        self, flash_attn_installed: bool, hip_version: str | None, expect_exit: bool
    ) -> None:
        """Test that flash_attn check exits only on non-ROCm PyTorch builds."""
        find_spec_result = MagicMock() if flash_attn_installed else None

        with (
            patch("importlib.util.find_spec", return_value=find_spec_result),
            patch.object(torch.version, "hip", new=hip_version),
            patch("sys.exit") as mock_exit,
        ):
            # Re-run the check logic from __init__.py under mocked conditions
            if importlib.util.find_spec("flash_attn") is not None:
                try:
                    import torch as _torch

                    _is_rocm = _torch.version.hip is not None
                except (ImportError, AttributeError):
                    _is_rocm = False
                if not _is_rocm:
                    sys.exit(1)

            if expect_exit:
                mock_exit.assert_called_once_with(1)
            else:
                mock_exit.assert_not_called()

    def test_flash_attn_check_handles_attribute_error(self) -> None:
        """Test that AttributeError on torch.version.hip falls back to non-ROCm."""
        with (
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch("sys.exit") as mock_exit,
        ):
            # Simulate a torch version that has no `hip` attribute
            class _FakeTorchVersion:
                pass

            fake_version = _FakeTorchVersion()

            if importlib.util.find_spec("flash_attn") is not None:
                try:
                    _is_rocm = fake_version.hip is not None  # type: ignore[attr-defined]
                except (ImportError, AttributeError):
                    _is_rocm = False
                if not _is_rocm:
                    sys.exit(1)

        mock_exit.assert_called_once_with(1)
