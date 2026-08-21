"""Tests for the `leaderboards.task_metadata` module."""

from leaderboards.enums import LeaderboardCategory
from leaderboards.task_metadata import category_includes_task, task_category


def test_category_includes_task_all_models_only_scores_nlu() -> None:
    """all_models only scores tasks whose task group is an NLU group."""
    assert category_includes_task(
        category=LeaderboardCategory.ALL_MODELS, task="sentiment-classification"
    )
    assert not category_includes_task(
        category=LeaderboardCategory.ALL_MODELS, task="summarization"
    )


def test_category_includes_task_chat_includes_everything() -> None:
    """The chat category scores every task unconditionally, orthogonal or not."""
    assert category_includes_task(
        category=LeaderboardCategory.CHAT, task="multiple-choice-stereotype-bias"
    )
    assert category_includes_task(
        category=LeaderboardCategory.CHAT, task="european-values"
    )


def test_category_includes_task_generative_excludes_instruct_exclusive() -> None:
    """Generative drops instruct-exclusive tasks."""
    assert not category_includes_task(
        category=LeaderboardCategory.GENERATIVE, task="instruction-following"
    )


def test_category_includes_task_only_chat_shows_orthogonal_tasks() -> None:
    """Orthogonal tasks (e.g. european-values) only show on chat."""
    assert not category_includes_task(
        category=LeaderboardCategory.GENERATIVE, task="european-values"
    )
    assert not category_includes_task(
        category=LeaderboardCategory.ALL_MODELS, task="european-values"
    )


def test_task_category_european_values_is_exempt_from_instruct_exclusive() -> None:
    """european-values is restricted to instruction-tuned/reasoning models too.

    But its ``ORTHOGONAL_TASKS`` membership keeps it out of the
    "instruct_exclusive" classification here (that classification only
    drives the generative/all_models nlu/nlg split); which categories
    actually show orthogonal tasks is handled separately in
    ``category_includes_task``.
    """
    assert task_category("european-values") == "nlg"


def test_task_category_instruct_only_tasks_are_instruct_exclusive() -> None:
    """Tasks restricted to instruction-tuned/reasoning models are instruct-exclusive."""
    assert task_category("instruction-following") == "instruct_exclusive"
    assert task_category("tool-calling") == "instruct_exclusive"


def test_task_category_plain_nlg_task() -> None:
    """A task open to base models, outside the NLU task groups, is "nlg"."""
    assert task_category("summarization") == "nlg"


def test_task_category_plain_nlu_task() -> None:
    """A task open to base models, in an NLU task group, classifies as "nlu"."""
    assert task_category("sentiment-classification") == "nlu"


def test_task_category_stereotype_bias_is_instruct_exclusive() -> None:
    """A task restricted to instruction-tuned/reasoning models and not orthogonal.

    Such a task is classified as instruct-exclusive.
    """
    assert task_category("multiple-choice-stereotype-bias") == "instruct_exclusive"
