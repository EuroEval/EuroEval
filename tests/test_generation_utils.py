"""Unit tests for the `generation_utils` module."""

import signal
from types import SimpleNamespace
from typing import Any

from datasets import Dataset

from euroeval.generation_utils import _extract_token_classification_examples


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


def test_token_classification_few_shot_terminates_with_sparse_entities() -> None:
    """Few-shot extraction must terminate when entity examples are scarce.

    Regression: when the training split has fewer entity-bearing examples than
    `num_few_shots` and at least one entity-free example remains, the label
    `it.cycle` kept matching nothing and `shuffled_train` never shrank, so the
    loop spun forever. Mirrors the guard already present in
    `_extract_classification_examples`.
    """
    dataset = Dataset.from_dict(
        {
            "tokens": [["Alice"], ["the"], ["a"], ["and"], ["of"]],
            "labels": [["b-per"], ["o"], ["o"], ["o"], ["o"]],
        }
    )
    # Typed as Any: the function only reads `.labels`, so a lightweight stand-in
    # avoids constructing a full DatasetConfig.
    dataset_config: Any = SimpleNamespace(labels=["o", "b-per", "i-per"])

    with _Timeout(seconds=30):
        result = _extract_token_classification_examples(
            shuffled_train=dataset, num_few_shots=5, dataset_config=dataset_config
        )

    # Only one entity-bearing example exists, so it must return at most that one
    # (and, crucially, return at all).
    assert len(result) <= 1
