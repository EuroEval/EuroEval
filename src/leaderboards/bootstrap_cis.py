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

    # Precompute: model -> set of datasets
    model_datasets: dict[str, set[str]] = {
        mid: set(data.keys()) for mid, data in model_results.items()
    }

    # Precompute: dataset -> set of models that have data
    dataset_models: dict[str, set[str]] = defaultdict(set)
    for mid, ds in model_datasets.items():
        for d in ds:
            dataset_models[d].add(mid)

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

    # Precompute category membership for each dataset
    dataset_to_category: dict[str, set[str]] = {}
    for ds, task in dataset_to_task.items():
        cats: set[str] = set()
        if task in ORTHOGONAL_TASKS:
            continue
        for category in categories:
            if _category_includes_task(category, task):
                cats.add(category)
        if cats:
            dataset_to_category[ds] = cats

    # Output: model -> category -> language -> [bootstrap_score_1, ...]
    bootstrap_scores: dict[str, dict[str, dict[str, list[float]]]] = {}

    # For each bootstrap replicate
    for b in range(n_bootstraps):
        if b % 100 == 0:
            logger.info("Bootstrap replicate %d/%d ...", b, n_bootstraps)

        # Stratified resample: for each task, resample its datasets with replacement
        sampled_datasets: dict[str, list[str]] = {}
        for task, ds_list in task_datasets.items():
            n = len(ds_list)
            indices = rng.integers(0, n, size=n)
            sampled_datasets[task] = [ds_list[i] for i in indices]

        # Flatten to a set of sampled datasets
        sampled_set: set[str] = set()
        for ds_list in sampled_datasets.values():
            sampled_set.update(ds_list)

        # Precompute pooled SD and best model for each sampled dataset ONCE
        # Only iterate over models that have data for each dataset
        dataset_stats: dict[str, tuple[float, float]] = {}
        for ds in sampled_set:
            models_on_ds = dataset_models.get(ds, set())
            if not models_on_ds:
                continue
            # Collect scores for models that have data
            scores: list[tuple[str, float, float, list[float]]] = []
            for mid in models_on_ds:
                if mid in model_results and ds in model_datasets[mid]:
                    raw, mean_sc, se = model_results[mid][ds][0]
                    if np.isfinite(mean_sc):
                        scores.append((mid, mean_sc, se, raw))
            if not scores:
                continue
            # Find best model and compute pooled SD
            scores.sort(key=lambda x: x[1], reverse=True)
            best_mean = scores[0][1]
            all_raw = [r for _, _, _, r in scores]
            pooled_sd = np.std(all_raw) if len(all_raw) > 1 else 1.0
            dataset_stats[ds] = (best_mean, pooled_sd)

        # Precompute rank scores for ALL models on ALL sampled datasets ONCE
        # This is the key optimization: O(n_datasets * n_models) instead of
        # O(n_models * n_datasets * n_models)
        all_rank_scores: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        for ds, (best_mean, pooled_sd) in dataset_stats.items():
            if pooled_sd <= 0:
                continue
            models_on_ds = dataset_models.get(ds, set())
            for mid in models_on_ds:
                if mid not in model_results or ds not in model_datasets[mid]:
                    continue
                raw, mean_sc, se = model_results[mid][ds][0]
                if not np.isfinite(mean_sc):
                    continue
                # Resample raw scores with replacement to create variation
                resampled_raw = rng.choice(raw, size=len(raw), replace=True)
                resampled_mean = float(np.mean(resampled_raw))
                diff = (best_mean - resampled_mean) / pooled_sd
                score = 1.0 + diff
                # Determine which categories this dataset belongs to
                for cat in dataset_to_category.get(ds, set()):
                    all_rank_scores.setdefault(mid, {}).setdefault(cat, {})[ds] = score

        # Now compute overall scores for each model by aggregating up the hierarchy
        for model_id, model_data in model_results.items():
            for category in categories:
                # Filter to datasets this model has data for
                model_ds_in_sample: dict[str, str] = {}
                for task, ds_list in sampled_datasets.items():
                    for ds in ds_list:
                        if (
                            ds in model_datasets[model_id]
                            and ds in dataset_stats
                            and category in dataset_to_category.get(ds, set())
                        ):
                            model_ds_in_sample[ds] = task

                if not model_ds_in_sample:
                    continue

                # Get precomputed rank scores for this model
                model_rank_scores = all_rank_scores.get(model_id, {}).get(category, {})
                if not model_rank_scores:
                    continue

                # Aggregate: dataset -> task -> language -> overall
                overall = _aggregate_hierarchy(
                    model_rank_scores, configs, category, model_ds_in_sample
                )
                if overall is not None:
                    bootstrap_scores.setdefault(model_id, {}).setdefault(
                        category, {}
                    ).setdefault("overall", []).append(overall)

    # Convert lists to arrays
    result: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for mid, cats_data in bootstrap_scores.items():
        result[mid] = {}
        for cat, langs in cats_data.items():
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


def _aggregate_hierarchy(
    dataset_scores: dict[str, dict[str, float]],
    configs: dict[str, dict[str, list[str]]],
    category: str,
    model_ds_in_sample: dict[str, str],
) -> float | None:
    """Aggregate per-dataset scores up to overall mean rank score.

    Hierarchy: dataset -> task -> language -> overall.

    Args:
        dataset_scores: model_id -> category -> dataset -> score (precomputed).
        configs: Per-language task -> dataset mappings.
        category: Category to aggregate.
        model_ds_in_sample: model_id -> dataset -> task mapping for this model.

    Returns:
        The overall mean rank score, or None if no data.
    """
    # Group datasets by task, then by language
    lang_task_scores: dict[str, dict[str, list[float]]] = {}

    for ds, score in dataset_scores.items():
        # Find which task this dataset belongs to
        task = None
        found_lang = None
        for lang_name, config in configs.items():
            for task_name, task_ds in config.items():
                if ds in task_ds:
                    task = task_name
                    found_lang = lang_name
                    break
            if task:
                break
        if task and task not in ORTHOGONAL_TASKS and found_lang:
            lang_task_scores.setdefault(found_lang, {}).setdefault(task, []).append(
                score
            )

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
    bootstrap_scores: dict[str, dict[str, dict[str, np.ndarray]]], alpha: float = 0.05
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Compute percentile CIs from bootstrap distributions.

    Returns:
        model_id -> category -> language -> {"score", "ci_lower", "ci_upper"}.

    Uses the median as the point estimate and percentile CIs for the bounds.
    """
    result: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    for model_id, cats_data in bootstrap_scores.items():
        result[model_id] = {}
        for cat, langs in cats_data.items():
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
