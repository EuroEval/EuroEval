"""Shared visibility rules for generated leaderboards."""

import csv
from pathlib import Path

from .constants import MINIMUM_NUMBER_OF_RANKED_ENTRIES


def leaderboard_should_be_shown(simplified_csv_path: Path) -> bool:
    """Return whether a leaderboard has enough ranked entries to be shown.

    Args:
        simplified_csv_path:
            Path to the ranked-only simplified leaderboard CSV.

    Returns:
        Whether the leaderboard has at least the required number of ranked entries.
    """
    return (
        count_ranked_entries(simplified_csv_path=simplified_csv_path)
        >= MINIMUM_NUMBER_OF_RANKED_ENTRIES
    )


def count_ranked_entries(simplified_csv_path: Path) -> int:
    """Count ranked entries in a simplified leaderboard CSV.

    Args:
        simplified_csv_path:
            Path to the ranked-only simplified leaderboard CSV.

    Returns:
        The number of ranked entries, or zero when the CSV does not exist.
    """
    if not simplified_csv_path.exists():
        return 0
    with simplified_csv_path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))
