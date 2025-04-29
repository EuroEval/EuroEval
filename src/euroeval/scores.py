"""Aggregation of raw scores into the mean and a confidence interval."""

import logging
import typing as t
import warnings

import numpy as np

if t.TYPE_CHECKING:
    from .data_models import MetricConfig
    from .types import ScoreDict

logger = logging.getLogger("euroeval")


def log_scores(
    dataset_name: str,
    metric_configs: list["MetricConfig"],
    scores: list[dict[str, float]],
    model_id: str,
    model_revision: str,
) -> "ScoreDict":
    """Log the scores.

    Args:
        dataset_name:
            Name of the dataset.
        metric_configs:
            List of metrics to log.
        scores:
            The scores that are to be logged. This is a list of dictionaries full of
            scores.
        model_id:
            The model ID of the model that was evaluated.
        model_revision:
            The revision of the model.

    Returns:
        A dictionary with keys 'raw_scores' and 'total', with 'raw_scores' being
        identical to `scores` and 'total' being a dictionary with the aggregated scores
        (means and standard errors).
    """
    if model_revision and model_revision != "main":
        model_id += f"@{model_revision}"

    logger.info(f"Finished evaluation of {model_id} on {dataset_name}.")

    total_dict: dict[str, float] = dict()
    for metric_cfg in metric_configs:
        test_score, test_se = aggregate_scores(scores=scores, metric_config=metric_cfg)
        test_score, test_score_str = metric_cfg.postprocessing_fn(test_score)
        test_se, test_se_str = metric_cfg.postprocessing_fn(test_se)
        total_dict[f"test_{metric_cfg.name}"] = test_score
        total_dict[f"test_{metric_cfg.name}_se"] = test_se
        logger.info(f"{metric_cfg.pretty_name}: {test_score_str} ± {test_se_str}")

    return dict(raw=scores, total=total_dict)


def aggregate_scores(
    scores: list[dict[str, float]], metric_config: "MetricConfig"
) -> tuple[float, float]:
    """Helper function to compute the mean with confidence intervals.

    Args:
        scores:
            Dictionary with the names of the metrics as keys, of the form
            "<split>_<metric_name>", such as "val_f1", and values the metric values.
        metric_config:
            The configuration of the metric, which is used to collect the correct
            metric from `scores`.

    Returns:
        A pair of floats, containing the score and the radius of its 95% confidence
        interval.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        test_scores = [
            (
                dct[metric_config.name]
                if metric_config.name in dct
                else dct[f"test_{metric_config.name}"]
            )
            for dct in scores
        ]
        test_score = np.mean(test_scores).item()

        if len(test_scores) > 1:
            sample_std = np.std(test_scores, ddof=1)
            test_se = sample_std / np.sqrt(len(test_scores))
        else:
            test_se = np.nan

        return (test_score, 1.96 * test_se)
