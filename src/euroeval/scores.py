"""Aggregation of raw scores into the mean and a confidence interval."""

import collections.abc as c
import logging
import typing as t
import warnings

import numpy as np

from .logging_utils import log

if t.TYPE_CHECKING:
    from .metrics import Metric
    from .types import ScoreDict


def log_scores(
    dataset_name: str,
    metrics: c.Sequence["Metric"],
    scores: c.Sequence[dict[str, float]],
    model_id: str,
    model_revision: str,
    model_param: str | None,
) -> "ScoreDict":
    """Log the scores.

    Args:
        dataset_name:
            Name of the dataset.
        metrics:
            List of metrics to log.
        scores:
            The scores that are to be logged. This is a list of dictionaries full of
            scores.
        model_id:
            The model ID of the model that was evaluated.
        model_revision:
            The revision of the model.
        model_param:
            The model parameter, if any.

    Returns:
        A dictionary with keys 'raw_scores' and 'total', with 'raw_scores' being
        identical to `scores` and 'total' being a dictionary with the aggregated scores
        (means and standard errors).
    """
    if model_revision and model_revision != "main":
        model_id += f"@{model_revision}"
    if model_param is not None:
        model_id += f"#{model_param}"

    total_dict: dict[str, float] = dict()
    all_log_strs: list[str] = [f"Finished benchmarking {model_id} on {dataset_name}."]
    for metric in metrics:
        test_score, test_se = aggregate_scores(scores=scores, metric=metric)
        test_score, test_score_str = metric.postprocessing_fn(test_score)
        test_se, test_se_str = metric.postprocessing_fn(test_se)
        total_dict[f"test_{metric.name}"] = test_score
        total_dict[f"test_{metric.name}_se"] = test_se
        log_str = (
            f"- {metric.pretty_name}: {test_score_str} ± {test_se_str}"
            if not np.isnan(test_se)
            else f"- {metric.pretty_name}: {test_score_str}"
        )
        all_log_strs.append(log_str)
    log("\n".join(all_log_strs), level=logging.INFO)

    return dict(raw=scores, total=total_dict)


def aggregate_scores(
    scores: c.Sequence[dict[str, float]], metric: "Metric"
) -> tuple[float, float]:
    """Helper function to compute the mean with confidence intervals.

    Args:
        scores:
            Dictionary with the names of the metrics as keys, of the form
            "<split>_<metric_name>", such as "val_f1", and values the metric values.
        metric:
            The metric, which is used to collect the correct metric from `scores`.

    Returns:
        A pair of floats, containing the score and the radius of its 95% confidence
        interval.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        test_scores = [
            dct[metric.name] if metric.name in dct else dct[f"test_{metric.name}"]
            for dct in scores
        ]
        test_score = np.mean(test_scores).item()

        if len(test_scores) > 1:
            sample_std = np.std(test_scores, ddof=1)
            test_se = (sample_std / np.sqrt(len(test_scores))).item()
        else:
            test_se = np.nan

        return (test_score, 1.96 * test_se)
