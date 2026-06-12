"""Shared constants for the leaderboards package."""

from __future__ import annotations

# Tasks that don't contribute to the main rank score and are rendered
# alongside model metadata instead of inside the task-grouped columns.
ORTHOGONAL_TASKS: frozenset[str] = frozenset({"european-values"})
