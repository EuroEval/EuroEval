"""Emit a JSON of task -> [primary, secondary] metric pretty names.

This is consumed by the frontend (`LeaderboardTable.vue`) to label the
raw "x ± e / y ± e" score cells on hover. Sourcing the metric names
from `euroeval.tasks` here keeps the frontend's tooltip in sync with
the library without a hand-maintained dict.

Re-run whenever a leaderboard task's metrics change. The output file is
checked in so the frontend build has no Python dependency at runtime.
"""

import json
import logging
from pathlib import Path

from leaderboards.paths import REPO_ROOT
from leaderboards.task_metadata import LEADERBOARD_TASKS, task_metric_pretty_names

logger = logging.getLogger(__name__)

OUTPUT_PATH: Path = REPO_ROOT / "src" / "frontend" / "generated" / "task-metrics.json"


def main() -> None:
    """Generate the task-metrics JSON file."""
    payload: dict[str, list[str]] = {}
    for task in LEADERBOARD_TASKS:
        primary, secondary = task_metric_pretty_names(task)
        payload[task] = [primary] + ([secondary] if secondary is not None else [])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open(mode="w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
