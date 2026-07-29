"""Bootstrap-based confidence intervals for rank scores.

Resamples datasets with replacement (stratified by task), recomputes the full
hierarchy for each replicate, and uses the empirical distribution of overall
scores for the confidence intervals.

This approach respects the nested structure (dataset -> task -> language ->
overall) and the correlation between models that share datasets, addressing
the limitations of the analytical CI propagation and the CI-overlap heuristic.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from euroeval.constants import ORTHOGONAL_TASKS

from .constants import LEADERBOARD_CATEGORIES
from .task_metadata import category_includes_task


def bootstrap_confidence_intervals(
    bootstrap_scores: dict[str, dict[str, dict[str, np.ndarray]]], alpha: float = 0.05
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Compute percentile CIs from bootstrap distributions.

    Uses the median as the point estimate and percentile CIs for the bounds.

    Args:
        bootstrap_scores:
            Output of ``bootstrap_rank_scores``.
        alpha (optional):
            Significance level used for the two-sided percentile interval.
            Defaults to 0.05.

    Returns:
        model_id -> category -> language -> {"score", "ci_lower", "ci_upper"}.
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


def bootstrap_rank_scores(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    configs: dict[str, dict[str, list[str]]],
    n_bootstraps: int,
    seed: int | None = None,
    categories: tuple[str, ...] | None = None,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Compute bootstrap distributions of overall mean rank scores.

    For each bootstrap replicate, resample datasets with replacement (stratified
    by task), recompute the full hierarchy, and collect the overall score for
    every model. Since models share the same resampled datasets within a
    replicate, their bootstrap scores stay correlated, so downstream difference
    distributions correctly account for that correlation.

    Args:
        model_results:
            The model results (same format as compute_ranks).
        configs:
            Per-language task -> dataset mappings.
        n_bootstraps:
            Number of bootstrap replicates.
        seed (optional):
            Random seed for reproducibility. Defaults to None.
        categories (optional):
            Which categories to compute. Defaults to LEADERBOARD_CATEGORIES.

    Returns:
        Nested dict: model_id -> category -> language -> np.ndarray of
        bootstrap scores (shape: (n_bootstraps,)).
    """
    if categories is None:
        categories = LEADERBOARD_CATEGORIES

    rng = np.random.default_rng(seed)

    model_datasets: dict[str, set[str]] = {
        mid: set(data.keys()) for mid, data in model_results.items()
    }

    dataset_models: dict[str, set[str]] = defaultdict(set)
    for mid, ds in model_datasets.items():
        for d in ds:
            dataset_models[d].add(mid)

    task_datasets, dataset_to_category = _build_dataset_mappings(
        configs=configs, categories=categories
    )

    bootstrap_scores: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for _ in range(n_bootstraps):
        iteration_scores = _bootstrap_single_iteration(
            task_datasets=task_datasets,
            dataset_models=dataset_models,
            model_datasets=model_datasets,
            model_results=model_results,
            dataset_to_category=dataset_to_category,
            configs=configs,
            categories=categories,
            rng=rng,
        )
        for model_id, model_data in iteration_scores.items():
            for category, cat_data in model_data.items():
                for key, score in cat_data.items():
                    bootstrap_scores[model_id][category][key].append(score)

    return _convert_bootstrap_scores_to_arrays(bootstrap_scores)


def _bootstrap_single_iteration(
    task_datasets: dict[str, list[str]],
    dataset_models: dict[str, set[str]],
    model_datasets: dict[str, set[str]],
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    dataset_to_category: dict[str, set[str]],
    configs: dict[str, dict[str, list[str]]],
    categories: tuple[str, ...],
    rng: np.random.Generator,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run a single bootstrap iteration and return aggregated scores.

    Args:
        task_datasets:
            Mapping of task to list of datasets.
        dataset_models:
            Mapping of dataset to set of models.
        model_datasets:
            Mapping of model to set of datasets.
        model_results:
            The model results.
        dataset_to_category:
            Mapping of dataset to categories.
        configs:
            Per-language task -> dataset mappings.
        categories:
            Categories to compute.
        rng:
            Random number generator.

    Returns:
        Nested dict: model_id -> category -> language -> score.
    """
    # Resample datasets with replacement (stratified by task)
    sampled_datasets: dict[str, list[str]] = {}
    for task, ds_list in task_datasets.items():
        n = len(ds_list)
        indices = rng.integers(0, n, size=n)
        sampled_datasets[task] = [ds_list[i] for i in indices]

    sampled_set: set[str] = set()
    for ds_list in sampled_datasets.values():
        sampled_set.update(ds_list)

    dataset_stats = _compute_dataset_stats(
        sampled_set=sampled_set,
        dataset_models=dataset_models,
        model_datasets=model_datasets,
        model_results=model_results,
    )

    all_rank_scores = _compute_rank_scores(
        dataset_stats=dataset_stats,
        dataset_models=dataset_models,
        model_datasets=model_datasets,
        model_results=model_results,
        dataset_to_category=dataset_to_category,
        rng=rng,
    )

    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    for model_id in model_results:
        model_agg = _aggregate_bootstrap_sample(
            model_id=model_id,
            categories=categories,
            all_rank_scores=all_rank_scores,
            configs=configs,
        )
        for category, cat_data in model_agg.items():
            for key, score in cat_data.items():
                if score is not None:
                    aggregated.setdefault(model_id, {}).setdefault(category, {})[
                        key
                    ] = score
    return aggregated


def _aggregate_bootstrap_sample(
    model_id: str,
    categories: tuple[str, ...],
    all_rank_scores: dict[str, dict[str, dict[str, float]]],
    configs: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, float | None]]:
    """Aggregate per-dataset scores for one model in one bootstrap sample.

    Returns:
        category -> {lang: score, "overall": overall or None}
    """
    result: dict[str, dict[str, float | None]] = {}
    for category in categories:
        model_rank_scores = all_rank_scores.get(model_id, {}).get(category, {})
        if not model_rank_scores:
            continue
        language_scores, overall = _aggregate_scores_to_categories(
            dataset_scores=model_rank_scores, configs=configs
        )
        result[category] = {**language_scores, "overall": overall}
    return result


def _aggregate_scores_to_categories(
    dataset_scores: dict[str, float], configs: dict[str, dict[str, list[str]]]
) -> tuple[dict[str, float], float | None]:
    """Aggregate per-dataset scores up to language and overall mean rank scores.

    Hierarchy: dataset -> task -> language -> overall.

    Args:
        dataset_scores:
            dataset -> score mapping for this model.
        configs:
            Per-language task -> dataset mappings.

    Returns:
        A tuple of (language_scores, overall_score). Language scores is a dict
        mapping language name to mean rank score. Overall score is the mean
        across languages. Returns ({}, None) if there is no data.
    """
    lang_task_scores: dict[str, dict[str, list[float]]] = {}

    for ds, score in dataset_scores.items():
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
        return {}, None

    language_scores: dict[str, float] = {}
    lang_means = []
    for lang, task_scores in lang_task_scores.items():
        task_means = [np.mean(scores) for scores in task_scores.values()]
        if task_means:
            lang_mean = float(np.mean(task_means))
            language_scores[lang] = lang_mean
            lang_means.append(lang_mean)

    overall = float(np.mean(lang_means)) if lang_means else None
    return language_scores, overall


def _compute_dataset_stats(
    sampled_set: set[str],
    dataset_models: dict[str, set[str]],
    model_datasets: dict[str, set[str]],
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
) -> dict[str, tuple[float, float]]:
    """Compute pooled SD and best model for each sampled dataset.

    Args:
        sampled_set:
            Set of sampled datasets.
        dataset_models:
            Mapping of dataset to set of models.
        model_datasets:
            Mapping of model to set of datasets.
        model_results:
            The model results.

    Returns:
        Dict of dataset -> (best_mean, pooled_sd).
    """
    dataset_stats: dict[str, tuple[float, float]] = {}
    for ds in sampled_set:
        models_on_ds = dataset_models.get(ds, set())
        if not models_on_ds:
            continue
        scores: list[tuple[str, float, list[float]]] = []
        for mid in models_on_ds:
            if mid in model_results and ds in model_datasets[mid]:
                raw, mean_sc, _ = model_results[mid][ds][0]
                if np.isfinite(mean_sc):
                    scores.append((mid, mean_sc, raw))
        if not scores:
            continue
        scores.sort(key=lambda x: x[1], reverse=True)
        best_mean = scores[0][1]
        all_raw = [score for _, _, r in scores for score in r]
        pooled_sd = np.std(all_raw) if len(all_raw) > 1 else 1.0
        dataset_stats[ds] = (best_mean, float(pooled_sd))
    return dataset_stats


def _compute_rank_scores(
    dataset_stats: dict[str, tuple[float, float]],
    dataset_models: dict[str, set[str]],
    model_datasets: dict[str, set[str]],
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    dataset_to_category: dict[str, set[str]],
    rng: np.random.Generator,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute rank scores for all models on all sampled datasets.

    Args:
        dataset_stats:
            Per-dataset statistics.
        dataset_models:
            Mapping of dataset to set of models.
        model_datasets:
            Mapping of model to set of datasets.
        model_results:
            The model results.
        dataset_to_category:
            Mapping of dataset to categories.
        rng:
            Random number generator.

    Returns:
        Nested dict: model_id -> category -> dataset -> score.
    """
    all_rank_scores: dict[str, dict[str, dict[str, float]]] = {}
    for ds, (best_mean, pooled_sd) in dataset_stats.items():
        if pooled_sd <= 0:
            continue
        models_on_ds = dataset_models.get(ds, set())
        for mid in models_on_ds:
            if mid not in model_results or ds not in model_datasets[mid]:
                continue
            raw, mean_sc, _ = model_results[mid][ds][0]
            if not np.isfinite(mean_sc):
                continue
            resampled_raw = rng.choice(raw, size=len(raw), replace=True)
            resampled_mean = float(np.mean(resampled_raw))
            diff = (best_mean - resampled_mean) / pooled_sd
            score = 1.0 + diff
            for cat in dataset_to_category.get(ds, set()):
                all_rank_scores.setdefault(mid, {}).setdefault(cat, {})[ds] = score
    return all_rank_scores


def _build_dataset_mappings(
    configs: dict[str, dict[str, list[str]]], categories: tuple[str, ...]
) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    """Build task->datasets and dataset->categories mappings for bootstrap.

    Args:
        configs:
            Per-language task -> dataset mappings.
        categories:
            Categories to compute.

    Returns:
        Tuple of (task_datasets dict, dataset_to_category dict).
    """
    dataset_to_task: dict[str, str] = {}
    for _language, config in configs.items():
        for task, task_datasets_list in config.items():
            if task in ORTHOGONAL_TASKS:
                continue
            for ds in task_datasets_list:
                dataset_to_task[ds] = task

    task_datasets: dict[str, list[str]] = defaultdict(list)
    for ds, task in dataset_to_task.items():
        task_datasets[task].append(ds)

    dataset_to_category: dict[str, set[str]] = {}
    for ds, task in dataset_to_task.items():
        cats: set[str] = set()
        for category in categories:
            if category_includes_task(category=category, task=task):
                cats.add(category)
        if cats:
            dataset_to_category[ds] = cats

    return task_datasets, dataset_to_category


def _convert_bootstrap_scores_to_arrays(
    bootstrap_scores: dict[str, dict[str, dict[str, list[float]]]],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Convert bootstrap score lists to numpy arrays.

    Args:
        bootstrap_scores:
            Nested dict with lists of bootstrap scores.

    Returns:
        Same structure but with np.ndarray values.
    """
    result: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for mid, cats_data in bootstrap_scores.items():
        result[mid] = {}
        for cat, langs in cats_data.items():
            result[mid][cat] = {
                lang: np.array(scores) for lang, scores in langs.items()
            }
    return result
