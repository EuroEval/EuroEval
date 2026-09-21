"""Tests for NLTK output suppression."""

import io
from pathlib import Path
from unittest.mock import patch

from euroeval.nltk_utils import ensure_nltk_packages  # noqa: WPS433 (needed for test)


def test_ensure_nltk_packages_suppresses_output() -> None:
    """Test that ensure_nltk_packages suppresses download output."""
    captured_output = io.StringIO()

    # Create a temporary cache directory
    with patch("sys.stdout", captured_output), patch("sys.stderr", captured_output):
        test_cache_dir = Path("/tmp/test_ensure_nltk")
        test_cache_dir.mkdir(parents=True, exist_ok=True)

        # Call ensure_nltk_packages which should suppress output
        # Note: This will actually try to download packages, but with quiet=True
        # and no_terminal_output() wrapping, there should be no output
        ensure_nltk_packages(test_cache_dir, packages=["punkt_tab"])

    output = captured_output.getvalue()

    # Should not have any NLTK-related output
    assert "[nltk_data]" not in output, f"Found unsuppressed NLTK output: {output}"
    assert "Downloading" not in output.lower(), f"Found download messages: {output}"
