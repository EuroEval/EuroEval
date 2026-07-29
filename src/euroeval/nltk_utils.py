"""Utility functions for NLTK configuration and downloads."""

import os
from pathlib import Path

from .logging_utils import no_terminal_output

# Auto-configure NLTK cache on module import (top of call hierarchy)
# This ensures any code using NLTK after importing euroeval gets correct paths
_euroeval_cache_dir = Path(__file__).parent.parent / ".euroeval_cache"
setup_nltk_data_dir(_euroeval_cache_dir)


def download_nltk_packages(
    nltk_data_dir: Path, packages: list[str] | None = None
) -> None:
    """Download NLTK packages to a custom directory with suppressed output.

    Args:
        nltk_data_dir:
            The directory where NLTK data will be stored.
        packages:
            List of NLTK package names to download. If None, downloads the default
            packages: punkt_tab, wordnet, omw-1.4.
    """
    import nltk  # noqa: PLC0415 (import inside function intentional - defers NLTK init)

    if packages is None:
        packages = ["punkt_tab", "wordnet", "omw-1.4"]

    # Suppress all NLTK output during download
    with no_terminal_output():
        for package in packages:
            nltk.download(package, download_dir=str(nltk_data_dir), quiet=True)


def ensure_nltk_packages(
    cache_dir: str | Path, packages: list[str] | None = None
) -> Path:
    """Configure NLTK and download required packages.

    This is the main entry point - call this before using any NLTK functionality.

    Args:
        cache_dir:
            The cache directory where NLTK data will be stored.
        packages:
            List of NLTK package names to download. If None, downloads the default
            packages: punkt_tab, wordnet, omw-1.4.

    Returns:
        The path to the NLTK data directory.
    """
    nltk_data_dir = setup_nltk_data_dir(cache_dir)
    download_nltk_packages(nltk_data_dir, packages)
    return nltk_data_dir


def setup_nltk_data_dir(cache_dir: str | Path) -> Path:
    """Configure NLTK to use a custom data directory and return the path.

    This ensures NLTK data is stored inside .euroeval_cache and not in the
    home directory. Should be called early in the initialization chain.

    Args:
        cache_dir:
            The cache directory where NLTK data will be stored.

    Returns:
        The path to the NLTK data directory.
    """
    import nltk  # noqa: PLC0415 (import inside function intentional - defers NLTK init)

    nltk_data_dir = Path(cache_dir) / "nltk_data"
    nltk_data_dir.mkdir(parents=True, exist_ok=True)

    # Set NLTK_DATA before any NLTK operations
    os.environ["NLTK_DATA"] = str(nltk_data_dir)

    # Override NLTK's search path to use only our cache directory
    nltk.data.path = [str(nltk_data_dir)]

    return nltk_data_dir
