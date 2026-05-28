"""Bootstrap-based confidence intervals and statistical tests for rank scores.

Resamples datasets with replacement (stratified by task), recomputes the full
hierarchy for each replicate, and uses the empirical distribution of overall
scores for both confidence intervals and pairwise hypothesis tests.

This approach respects the nested structure (dataset -> task -> language ->
overall) and the correlation between models that share datasets, addressing
the limitations of the analytical CI propagation and the CI-overlap heuristic.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from .task_metadata import ORTHOGONAL_TASKS, task_category

logger = logging.getLogger(__name__)

# Categories used throughout the pipeline.
_CATEGORIES = ("generative", "all_models")


def bootstrap_rank_scores(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    configs: dict[str, dict[str, list[str]]],
    n_bootstraps: int = 1000,
    seed: int | None = None,
    categories: tuple[str, ...] | None = None,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Compute bootstrap distributions of overall mean rank scores.

    For each bootstrap replicate, resample datasets with replacement (stratified
    by task), recompute the full hierarchy, and collect the overall score for
    every model.

    Args:
        model_results: The model results (same format as compute_ranks).
        configs: Per-language task -> dataset mappings.
        n_bootstraps: Number of bootstrap replicates.
        seed: Random seed for reproducibility.
        categories: Which categories to compute ("generative", "all_models").

    Returns:
        Nested dict: model_id -> category -> language -> np.ndarray of
        bootstrap scores (shape: (n_bootstraps,)).

    The key insight: since both models share the same resampled datasets, their
    bootstrap scores are correlated. The difference distribution correctly
    accounts for this correlation.
    """
    if categories is None:
        categories = _CATEGORIES

    rng = np.random.default_rng(seed)
    # Output: model -> category -> language -> [bootstrap_score_1, ...]
    bootstrap_scores: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    # Build dataset -> task mapping for stratified resampling
    dataset_to_task: dict[str, str] = {}
    for language, config in configs.items():
        for task, task_datasets_list in config.items():
            if task in ORTHOGONAL_TASKS:
                continue
            for ds in task_datasets_list:
                dataset_to_task[ds] = task

    # Group datasets by task for stratified resampling
    task_datasets: dict[str, list[str]] = defaultdict(list)
    for ds, task in dataset_to_task.items():
        task_datasets[task].append(ds)

    # For each bootstrap replicate, iterate over models
    for b in range(n_bootstraps):
        if b % 100 == 0:
            logger.info("Bootstrap replicate %d/%d ...", b, n_bootstraps)

        # Stratified resample: for each task, resample its datasets with replacement
        sampled_datasets: dict[str, list[str]] = {}
        for task, ds_list in task_datasets.items():
            n = len(ds_list)
            indices = rng.integers(0, n, size=n)
            sampled_datasets[task] = [ds_list[i] for i in indices]

        # Flatten to a dict: dataset -> task
        sampled_flat: dict[str, str] = {}
        for task, ds_list in sampled_datasets.items():
            for ds in ds_list:
                sampled_flat[ds] = task

        # For each model, compute the full hierarchy on sampled datasets
        for model_id, model_data in model_results.items():
            for category in categories:
                # Filter to datasets this model has data for
                model_ds_in_sample = {
                    ds: task
                    for ds, task in sampled_flat.items()
                    if ds in model_data
                    and _category_includes_task(category, task)
                }

                if not model_ds_in_sample:
                    continue

                # Compute per-dataset rank scores for this model's sampled datasets
                # (needs all models' data on each dataset for pooled_sd, best model)
                ds_scores = _compute_sampled_dataset_scores(
                    model_results, model_ds_in_sample, category, configs
                )

                if not ds_scores:
                    continue

                # Aggregate: dataset -> task -> language -> overall
                overall = _aggregate_hierarchy(ds_scores, configs, category)
                if overall is not None:
                    bootstrap_scores[model_id][category]["overall"].append(overall)

    # Convert lists to arrays
    result: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for mid, cats in bootstrap_scores.items():
        result[mid] = {}
        for cat, langs in cats.items():
            result[mid][cat] = {
                lang: np.array(scores) for lang, scores in langs.items()
            }

    return result


def _category_includes_task(category: str, task: str) -> bool:
    """Check whether a task belongs to a category.

    Returns:
        True if the task belongs to the category.
    """
    return category == "generative" or task_category(task) == "nlu"


def _compute_sampled_dataset_scores(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    model_ds_in_sample: dict[str, str],
    category: str,
    configs: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Compute per-dataset rank scores for sampled datasets, for all models.

    This mirrors compute_dataset_ranks but only processes sampled datasets.

    Returns:
        model_id -> category -> language -> dataset -> score.
    """
    # Group sampled datasets by (language, task)
    ds_by_lang_task: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for ds, task in model_ds_in_sample.items():
        for lang, config in configs.items():
            if task in config:
                ds_by_lang_task[lang][task].append(ds)
                break

    # For each (language, task, dataset), compute rank scores
    dataset_scores: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )

    for lang, task_map in ds_by_lang_task.items():
        for task, ds_list in task_map.items():
            if not _category_includes_task(category, task):
                continue
            for ds in ds_list:
                # Collect all models' scores on this dataset
                model_scores_on_ds: dict[str, tuple[float, float, list[float]]] = {}
                for mid, data in model_results.items():
                    if ds in data:
                        raw, mean_sc, se = data[ds][0]
                        if np.isfinite(mean_sc):
                            model_scores_on_ds[mid] = (mean_sc, se, raw)

                if not model_scores_on_ds:
                    continue

                # Find best model and compute pooled SD
                sorted_models = sorted(
                    model_scores_on_ds.items(), key=lambda x: x[1][0], reverse=True
                )
                best_mean = sorted_models[0][1][0]
                all_raw = [r for _, _, r in model_scores_on_ds.values() for r in r]
                pooled_sd = np.std(all_raw) if len(all_raw) > 1 else 1.0

                # Compute rank scores
                for mid, (mean_sc, se, _) in model_scores_on_ds.items():
                    diff = (best_mean - mean_sc) / pooled_sd if pooled_sd > 0 else 0.0
                    score = 1.0 + diff
                    dataset_scores[mid][category][lang][ds] = score

    return dataset_scores


def _aggregate_hierarchy(
    dataset_scores: dict[str, dict[str, dict[str, dict[str, float]]]],
    configs: dict[str, dict[str, list[str]]],
    category: str,
) -> float | None:
    """Aggregate per-dataset scores up to overall mean rank score.

    Hierarchy: dataset -> task -> language -> overall.

    Returns:
        The overall mean rank score, or None if no data.
    """
    # Group datasets by task, then by language
    lang_task_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for mid, cat_data in dataset_scores.items():
        if category not in cat_data:
            continue
        for lang, ds_scores in cat_data[category].items():
            for ds, score in ds_scores.items():
                # Find which task this dataset belongs to
                task = None
                for lang_config, config in configs.items():
                    for task_name, task_ds in config.items():
                        if ds in task_ds:
                            task = task_name
                            break
                    if task:
                        break
                if task and task not in ORTHOGONAL_TASKS:
                    lang_task_scores[lang][task].append(score)

    if not lang_task_scores:
        return None

    # Aggregate: task means -> language means -> overall
    lang_means = []
    for lang, task_scores in lang_task_scores.items():
        task_means = [np.mean(scores) for scores in task_scores.values()]
        if task_means:
            lang_means.append(np.mean(task_means))

    return float(np.mean(lang_means)) if lang_means else None


def bootstrap_confidence_intervals(
    bootstrap_scores: dict[str, dict[str, dict[str, np.ndarray]]],
    alpha: float = 0.05,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Compute percentile CIs from bootstrap distributions.

    Returns:
        model_id -> category -> language -> {"score", "ci_lower", "ci_upper"}.

    Uses the median as the point estimate and percentile CIs for the bounds.
    """
    result: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    for model_id, cats in bootstrap_scores.items():
        result[model_id] = {}
        for cat, langs in cats.items():
            result[model_id][cat] = {}
            for lang, samples in langs.items():
                if len(samples) == 0:
                    continue
                q_low = np.percentile(samples, 100 * alpha / 2)
                q_high = np.percentile(samples, 100 * (1 - alpha / 2))
                result[model_id][cat][lang] = {
                    "score": float(np.median(samples)),
                    "ci_lower": float(q_low),
                    "ci_upper": float(q_high),
                }

    return result


def bootstrap_test(
    bootstrap_scores: dict[str, dict[str, dict[str, np.ndarray]]],
    model_a: str,
    model_b: str,
    category: str,
    language: str,
    alternative: str = "less",
) -> float:
    """Compute a one-sided p-value: is model A significantly better than B?

    Uses the bootstrap distribution of score differences. Lower score = better.

    Args:
        bootstrap_scores: Output of bootstrap_rank_scores.
        model_a: Candidate for being better (lower score).
        model_b: Candidate for being worse (higher score).
        category: Which category to test.
        language: Which language to test.
        alternative: "less" (A < B), "greater" (A > B), "two-sided".

    Returns:
        p-value.
    """
    samples_a = bootstrap_scores[model_a][category][language]
    samples_b = bootstrap_scores[model_b][category][language]

    # Align by index (same bootstrap replicates)
    min_len = min(len(samples_a), len(samples_b))
    diffs = samples_a[:min_len] - samples_b[:min_len]

    if alternative == "less":
        # Is A significantly lower (better) than B?
        p_value = np.mean(diffs <= 0)
    elif alternative == "greater":
        p_value = np.mean(diffs >= 0)
    else:
        p_value = 2 * np.minimum(np.mean(diffs <= 0), np.mean(diffs >= 0))

    return float(p_value)
