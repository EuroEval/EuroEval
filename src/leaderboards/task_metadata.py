"""Single source of truth for task metadata and per-language datasets.

The leaderboard pipeline used to maintain a parallel `task_config.yaml`
plus per-language `configs/<lang>.yaml` files that listed datasets. Both
duplicated information already declared in the `euroeval` library
(`euroeval.tasks` and `euroeval.dataset_configs`). This module derives
everything from those, so the per-language yamls only need to say which
languages a leaderboard covers.

A dataset is included in a leaderboard iff:
  - it is marked official (`DatasetConfig.unofficial is False`),
  - its task is one of `LEADERBOARD_TASKS`, and
  - at least one of its languages matches the leaderboard's languages.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections import OrderedDict
from functools import cache

from euroeval import dataset_configs as _ds_module
from euroeval.data_models import DatasetConfig
from euroeval.enums import TaskGroup
from euroeval.languages import get_all_languages
from euroeval.tasks import get_all_tasks

# Tasks displayed on every EuroEval leaderboard, in column order.
LEADERBOARD_TASKS: list[str] = [
    "sentiment-classification",
    "named-entity-recognition",
    "linguistic-acceptability",
    "reading-comprehension",
    "summarization",
    "knowledge",
    "common-sense-reasoning",
    "simplification",
    "european-values",
]

# The two leaderboard categories that every model is ranked within. The
# "generative" variant scores all tasks; "all_models" only scores NLU tasks so
# non-generative models can compete.
LEADERBOARD_CATEGORIES: tuple[str, str] = ("generative", "all_models")

# TaskGroup -> "nlu"/"nlg". The "all_models" leaderboard variant only
# scores NLU tasks so non-generative models can compete.
_NLU_GROUPS: frozenset[TaskGroup] = frozenset(
    {
        TaskGroup.SEQUENCE_CLASSIFICATION,
        TaskGroup.TOKEN_CLASSIFICATION,
        TaskGroup.QUESTION_ANSWERING,
    }
)


def task_category(task_name: str) -> str:
    """Return ``"nlu"`` or ``"nlg"`` for ``task_name``."""
    task = get_all_tasks()[task_name]
    return "nlu" if task.task_group in _NLU_GROUPS else "nlg"


def category_includes_task(category: str, task: str) -> bool:
    """Check whether a task is scored within a leaderboard category.

    Args:
        category:
            Leaderboard category name.
        task:
            Task slug.

    Returns:
        True if the task is scored within the category.
    """
    return category == "generative" or task_category(task) == "nlu"


def task_metric_names(task_name: str) -> tuple[str, str | None]:
    """Return ``(primary, secondary)`` metric slugs for a task.

    Secondary is ``None`` for single-metric tasks (e.g. ``european-values``).
    """
    metrics = get_all_tasks()[task_name].metrics
    primary = metrics[0].name
    secondary = metrics[1].name if len(metrics) > 1 else None
    return primary, secondary


def task_metric_pretty_names(task_name: str) -> tuple[str, str | None]:
    """Return ``(primary, secondary)`` human-readable metric names."""
    metrics = get_all_tasks()[task_name].metrics
    primary = metrics[0].pretty_name
    secondary = metrics[1].pretty_name if len(metrics) > 1 else None
    return primary, secondary


@cache
def _iter_all_dataset_configs() -> tuple[DatasetConfig, ...]:
    """Collect every ``DatasetConfig`` defined in ``euroeval.dataset_configs``.

    Cached because the leaderboard pipeline calls into this module once per
    language and the set is fixed per process.

    Returns:
        Every ``DatasetConfig`` found in the lib, in module-discovery order.
    """
    configs: list[DatasetConfig] = []
    for mod_info in pkgutil.iter_modules(_ds_module.__path__):
        mod = importlib.import_module(f"euroeval.dataset_configs.{mod_info.name}")
        for value in vars(mod).values():
            if isinstance(value, DatasetConfig):
                configs.append(value)
    return tuple(configs)


def language_name_to_codes(name: str) -> set[str]:
    """Resolve a leaderboard yaml language name (e.g. ``"danish"``) to codes.

    Returns:
        The set of language codes matching the given name.
    """
    target = name.strip().lower()
    return {
        lang.code
        for lang in get_all_languages().values()
        if lang.name.lower() == target
    }


def language_code_to_name(code: str) -> str:
    """Look up the canonical language name for a code (lower-cased).

    Returns:
        The lower-cased language name.
    """
    return get_all_languages()[code].name.lower()


def languages_with_official_datasets() -> list[str]:
    """List language names that have at least one official leaderboard dataset.

    Only single-token language names are returned, so dialect entries like
    ``norwegian bokmål``/``norwegian nynorsk``/``european portuguese`` don't
    produce duplicate leaderboards on top of their parent (``norwegian``,
    ``portuguese``). Names are lower-cased and sorted alphabetically.

    Returns:
        Sorted list of language names.
    """
    leaderboard_tasks = set(LEADERBOARD_TASKS)
    names: set[str] = set()
    languages = get_all_languages()
    for cfg in _iter_all_dataset_configs():
        if cfg.unofficial:
            continue
        if cfg.task.name not in leaderboard_tasks:
            continue
        for lang in cfg.languages:
            if lang.code not in languages:
                continue
            name = languages[lang.code].name.lower()
            if " " in name:
                continue
            names.add(name)
    return sorted(names)


def official_datasets_for_language(language_name: str) -> OrderedDict[str, list[str]]:
    """Return ``{task: [dataset_name, ...]}`` for a single-language leaderboard.

    Tasks appear in `LEADERBOARD_TASKS` order; tasks with no matching dataset
    are omitted. Dataset order within a task follows definition order in
    `euroeval.dataset_configs`.

    Returns:
        Ordered mapping from task name to list of dataset names.

    Raises:
        ValueError: If ``language_name`` doesn't match any known language.
    """
    codes = language_name_to_codes(language_name)
    if not codes:
        raise ValueError(f"Unknown leaderboard language: {language_name!r}")

    by_task: dict[str, list[str]] = {t: [] for t in LEADERBOARD_TASKS}
    for cfg in _iter_all_dataset_configs():
        if cfg.unofficial:
            continue
        if cfg.task.name not in by_task:
            continue
        if not any(lang.code in codes for lang in cfg.languages):
            continue
        if cfg.name not in by_task[cfg.task.name]:
            by_task[cfg.task.name].append(cfg.name)

    return OrderedDict(
        (task, datasets) for task, datasets in by_task.items() if datasets
    )
