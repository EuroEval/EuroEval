"""Tests for the `task_group_utils.sequence_classification` module."""

from euroeval.task_group_utils.sequence_classification import (
    get_closest_logprobs_labels,
)


class TestGetClosestLogprobsLabels:
    """Tests for `get_closest_logprobs_labels`."""

    def test_unsorted_scores_still_pick_the_highest_logprob_label(self) -> None:
        """The extracted label follows the highest logprob, even if unsorted."""
        candidate_labels = ["negative", "neutral", "positive"]

        # Deliberately unsorted: the highest logprob ("positive") is listed last.
        unsorted_scores = [[("negative", -5.0), ("neutral", -3.0), ("positive", -0.1)]]

        extracted = get_closest_logprobs_labels(
            generation_logprobs=[unsorted_scores],
            first_label_token_mapping=True,
            candidate_labels=[candidate_labels],
        )

        assert extracted == ["positive"]
