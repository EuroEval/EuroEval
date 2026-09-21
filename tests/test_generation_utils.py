"""Unit tests for the `generation_utils` module."""

import signal
from types import SimpleNamespace
from typing import Any

import pytest
from datasets import Dataset

from euroeval.generation_utils import _extract_token_classification_examples


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param(["o", "b-per", "B-PER", "i-per"], id="case-variant-labels"),
        pytest.param(["o", "b-per", "i-per"], id="sparse-entities"),
    ],
)
def test_token_classification_few_shot_terminates_with_sparse_entities(
    labels: list[str],
) -> None:
    """Few-shot extraction terminates when entity examples are scarce.

    This covers both ordinary sparse entities and case variants of the same label,
    which must not defeat the no-sample termination guard.
    """
    dataset = Dataset.from_dict(
        {
            "tokens": [["Alice"], ["the"], ["a"], ["and"], ["of"]],
            "labels": [["b-per"], ["o"], ["o"], ["o"], ["o"]],
        }
    )
    dataset_config: Any = SimpleNamespace(labels=labels)

    with _Timeout(seconds=30):
        result = _extract_token_classification_examples(
            shuffled_train=dataset, num_few_shots=5, dataset_config=dataset_config
        )

    assert len(result) == 1


class _Timeout:
    """Context manager that raises if the wrapped block runs too long.

    Used so an infinite loop surfaces as a test failure rather than a hang.
    """

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds

    def __enter__(self) -> "_Timeout":
        signal.signal(signal.SIGALRM, self._raise)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc: object) -> None:
        signal.alarm(0)

    @staticmethod
    def _raise(*_: object) -> None:
        raise TimeoutError("timed out — likely an infinite loop")
