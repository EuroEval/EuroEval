"""Regression tests for Vercel Git source staging."""

from pathlib import Path

from src.scripts.check_vercel_source_staging import validate_source_staging

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_vercel_source_staging_keeps_build_inputs() -> None:
    """Ensure .vercelignore cannot hide files required by the Vite build."""
    errors = validate_source_staging(repo_root=REPO_ROOT)
    assert not errors, "\n".join(errors)
