"""Functions related to computation of scores based on the model results."""

import collections.abc as c
import logging
import math
from collections import defaultdict

import numpy as np

from euroeval.constants import ORTHOGONAL_TASKS

from .bootstrap_cis import bootstrap_confidence_intervals, bootstrap_rank_scores
from .constants import LEADERBOARD_CATEGORIES, Z_SCORE_95
from .enums import LeaderboardCategory
from .task_metadata import category_includes_task

logger = logging.getLogger(__name__)


def compute_ranks(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    configs: dict[str, dict[str, list[str]]],
    n_bootstraps: int,
    seed: int | None = None,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Compute ranks via bootstrap confidence intervals.

    Dataset-level CIs are computed by resampling the raw iteration scores
    with replacement (n_bootstraps times), recomputing the rank score each
    time, and taking percentile CIs from the empirical distribution.

    Overall (language and aggregate) CIs are computed via the full
    non-parametric bootstrap in :func:`bootstrap_rank_scores`, which
    resamples datasets with replacement (stratified by task), recomputes the
    full hierarchy for each replicate, and returns the empirical distribution
    of overall scores as percentile confidence intervals.

    This replaces the older analytical CI propagation (which assumed normality
    of the mean and propagated variance through linear approximations) with a
    fully non-parametric approach that respects the nested structure and model
    correlations.

    Args:
        model_results:
            The model results.
        configs:
            The leaderboard configurations for each language.
        n_bootstraps:
            Number of bootstrap replicates for dataset-level CIs.
        seed:
            Random seed for reproducibility.

    Returns:
        The ranks of the models, per task category and per language.
        The dict structure is model_id -> category -> language/overall ->
        {"score", "ci_lower", "ci_upper"}.
    """
    logger.info("Computing ranks via bootstrap confidence intervals...")
    orthogonal_tasks = ORTHOGONAL_TASKS
    categories = LEADERBOARD_CATEGORIES

    # Step 1: Dataset-level ranks (bootstrap CIs).
    model_dataset_ranks = compute_dataset_ranks_bootstrap(
        model_results=model_results,
        configs=configs,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    # Step 2: Aggregate dataset -> task -> language -> overall.
    # Sort for deterministic iteration order.
    model_task_ranks = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for model_id in sorted(model_dataset_ranks.keys()):
        for category in sorted(categories):
            if category not in model_dataset_ranks[model_id]:
                continue
            for language in sorted(configs.keys()):
                config = configs[language]
                task_results = _aggregate_to_task_level(
                    model_id=model_id,
                    category=category,
                    language=language,
                    config=config,
                    model_dataset_ranks=model_dataset_ranks,
                    orthogonal_tasks=orthogonal_tasks,
                )
                if task_results:
                    for task, task_data in task_results.items():
                        model_task_ranks[model_id][category][language][task] = task_data

    # Step 3: Aggregate task -> language -> overall.
    # Sort for deterministic iteration order.
    final: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    for model_id in sorted(model_dataset_ranks.keys()):
        for category in sorted(categories):
            if category not in model_dataset_ranks[model_id]:
                continue

            lang_scores, overall_entries = _aggregate_to_language_level(
                model_id=model_id,
                category=category,
                configs=configs,
                model_task_ranks=model_task_ranks,
                orthogonal_tasks=orthogonal_tasks,
            )

            if overall_entries:
                mean_score = float(np.mean([e["score"] for e in overall_entries]))
                margin = _compute_margin(overall_entries)

                lang_scores["overall"] = {
                    "score": round(mean_score, 6),
                    "ci_lower": round(mean_score - margin, 6),
                    "ci_upper": round(mean_score + margin, 6),
                }
                final.setdefault(model_id, {})[category] = lang_scores

    logger.info("Finished computing ranks.")
    return final


def _aggregate_to_language_level(
    model_id: str,
    category: LeaderboardCategory,
    configs: dict[str, dict[str, list[str]]],
    model_task_ranks: dict,
    orthogonal_tasks: c.Container[str],
) -> tuple[dict[str, dict[str, float]], list[dict[str, float]]]:
    """Aggregate task ranks to language level.

    Args:
        model_id:
            The model ID.
        category:
            The category.
        configs:
            Per-language task -> dataset mappings.
        model_task_ranks:
            Task-level ranks.
        orthogonal_tasks:
            List of orthogonal task names.

    Returns:
        Tuple of (lang_scores dict, overall_entries list).
    """
    lang_scores: dict[str, dict[str, float]] = {}
    overall_entries: list[dict[str, float]] = []

    # Sort for deterministic iteration order.
    for language in sorted(configs.keys()):
        config = configs[language]
        task_entries = [
            model_task_ranks[model_id][category][language].get(task)
            for task in sorted(config.keys())
            if task not in orthogonal_tasks
            and category_includes_task(category=category, task=task)
        ]
        task_entries = [e for e in task_entries if e is not None]
        if not task_entries:
            continue

        mean_score = float(np.mean([e["score"] for e in task_entries]))
        margin = _compute_margin(task_entries)

        lang_scores[language] = {
            "score": round(mean_score, 6),
            "ci_lower": round(mean_score - margin, 6),
            "ci_upper": round(mean_score + margin, 6),
        }
        overall_entries.append(lang_scores[language])

    return lang_scores, overall_entries


def _compute_margin(entries: list[dict[str, float]]) -> float:
    """Compute confidence margin from a list of CI entries.

    Args:
        entries:
            List of dicts with "ci_upper" and "ci_lower" keys.

    Returns:
        The computed margin.
    """
    vars_ = [((e["ci_upper"] - e["ci_lower"]) / (2 * Z_SCORE_95)) ** 2 for e in entries]
    mean_var = np.sum(vars_) / (len(entries) ** 2)
    return Z_SCORE_95 * math.sqrt(mean_var)


def _aggregate_to_task_level(
    model_id: str,
    category: LeaderboardCategory,
    language: str,
    config: dict[str, list[str]],
    model_dataset_ranks: dict[str, dict[str, dict[str, dict[str, float]]]],
    orthogonal_tasks: c.Container[str],
) -> dict[str, dict[str, float]] | None:
    """Aggregate dataset ranks to task level for a single language.

    Args:
        model_id:
            The model ID.
        category:
            The category.
        language:
            The language.
        config:
            Task -> datasets mapping for this language.
        model_dataset_ranks:
            Dataset-level ranks.
        orthogonal_tasks:
            List of orthogonal task names to exclude.

    Returns:
        Dict of task -> {score, ci_lower, ci_upper} or None.
    """
    task_results: dict[str, dict[str, float]] = {}
    # Sort for deterministic iteration order.
    for task in sorted(config.keys()):
        task_datasets = config[task]
        if task in orthogonal_tasks:
            continue
        if not category_includes_task(category=category, task=task):
            continue

        entries = [
            model_dataset_ranks[model_id][category][ds]
            for ds in sorted(task_datasets)
            if ds in model_dataset_ranks[model_id][category]
        ]
        if not entries:
            continue

        mean_score = float(np.mean([e["score"] for e in entries]))
        margin = _compute_margin(entries)

        task_results[task] = {
            "score": round(mean_score, 6),
            "ci_lower": round(mean_score - margin, 6),
            "ci_upper": round(mean_score + margin, 6),
        }
    return task_results if task_results else None


def compute_dataset_ranks_bootstrap(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    configs: dict[str, dict[str, list[str]]],
    n_bootstraps: int,
    seed: int | None = None,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Compute per-dataset rank scores with bootstrap confidence intervals.

    For each model-dataset pair, resamples the raw iteration scores with
    replacement (n_bootstraps times), recomputes the rank score each time,
    and returns the empirical distribution's median and percentile CIs.

    The best model (highest mean score) is fixed from the observed data;
    only the candidate model's mean is resampled, keeping the normalisation
    stable across bootstrap replicates.

    Args:
        model_results: Model results grouped by model and dataset.
        configs: Per-language task -> dataset mappings.
        n_bootstraps: Number of bootstrap replicates.
        seed: Random seed for reproducibility.

    Returns:
        model_id -> category -> dataset -> {"score", "ci_lower", "ci_upper"}.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    # Sort for deterministic iteration order (RNG consumption must be
    # deterministic across runs with the same seed).
    for language in sorted(configs.keys()):
        config = configs[language]
        for category in sorted(LEADERBOARD_CATEGORIES):
            datasets = [
                ds
                for task in sorted(config.keys())
                for ds in sorted(config[task])
                if task not in ORTHOGONAL_TASKS
                and category_includes_task(category, task)
            ]
            for dataset in datasets:
                model_scores: dict[str, tuple[float, list[float]]] = {}
                for model_id in sorted(model_results.keys()):
                    results = model_results[model_id]
                    if dataset in results and results[dataset]:
                        raw, mean_sc, _ = results[dataset][0]
                        if np.isfinite(mean_sc):
                            model_scores[model_id] = (mean_sc, raw)

                if not model_scores:
                    continue

                dataset_results = _bootstrap_single_dataset_ranks(
                    model_scores=model_scores,
                    dataset=dataset,
                    category=category,
                    n_bootstraps=n_bootstraps,
                    rng=rng,
                )
                for mid, result_data in dataset_results.items():
                    out[mid][category][dataset] = result_data

    return out


def _bootstrap_single_dataset_ranks(
    model_scores: dict[str, tuple[float, list[float]]],
    dataset: str,
    category: LeaderboardCategory,
    n_bootstraps: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    """Compute bootstrap ranks for a single dataset.

    Args:
        model_scores:
            Dict of model_id -> (mean_score, raw_scores).
        dataset:
            Dataset name.
        category:
            Category name.
        n_bootstraps:
            Number of bootstrap replicates.
        rng:
            Random number generator.

    Returns:
        Dict of model_id -> {score, ci_lower, ci_upper}.
    """
    # Sort by mean score descending, so the best model is first
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1][0], reverse=True)
    mean_best = sorted_models[0][1][0]
    all_raw = [score for _, raw_scores in model_scores.values() for score in raw_scores]
    pooled_sd = np.std(all_raw) if len(all_raw) > 1 else 1.0
    if pooled_sd <= 0:
        pooled_sd = 1.0

    results: dict[str, dict[str, float]] = {}
    for mid, (_, raw) in model_scores.items():
        bootstrap_scores: list[float] = []
        for _ in range(n_bootstraps):
            resampled_raw = rng.choice(raw, size=len(raw), replace=True)
            resampled_mean = float(np.mean(resampled_raw))
            diff = float((mean_best - resampled_mean) / pooled_sd)
            bootstrap_scores.append(1.0 + diff)

        if not bootstrap_scores:
            continue

        score = float(np.median(bootstrap_scores))
        ci_lower = float(np.percentile(bootstrap_scores, 2.5))
        ci_upper = float(np.percentile(bootstrap_scores, 97.5))
        results[mid] = {
            "score": round(score, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
        }
    return results


def compute_ranks_bootstrap(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    configs: dict[str, dict[str, list[str]]],
    n_bootstraps: int,
    seed: int | None = None,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Compute bootstrap confidence intervals for overall mean rank scores.

    Resamples datasets with replacement (stratified by task), recomputes the
    full hierarchy for each replicate, and returns the empirical distribution
    of overall scores as percentile confidence intervals.

    This replaces the analytical CI propagation with a proper non-parametric
    approach that respects the nested structure and model correlations.

    Args:
        model_results: The model results (same format as compute_ranks).
        configs: Per-language task -> dataset mappings.
        n_bootstraps: Number of bootstrap replicates.
        seed: Random seed for reproducibility.

    Returns:
        model_id -> category -> language -> {"score", "ci_lower", "ci_upper"}
    """
    bootstrap_scores = bootstrap_rank_scores(
        model_results=model_results,
        configs=configs,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    return bootstrap_confidence_intervals(bootstrap_scores)


def compute_standard_ranks_bootstrap(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    configs: dict[str, dict[str, list[str]]],
    n_bootstraps: int,
    seed: int | None = None,
    alpha: float = 0.05,
) -> dict[str, dict[str, int]]:
    """Compute ordinal ranks with ties from paired bootstrap differences.

    Sorts models by median rank score (lower = better). A candidate starts a
    worse rank group only when the lower bound of the paired bootstrap CI for
    ``candidate - anchor`` is strictly above zero for every model in the current
    rank group.

    Args:
        model_results:
            The model results.
        configs:
            Per-language task -> dataset mappings.
        n_bootstraps:
            Number of bootstrap replicates.
        seed:
            Random seed for reproducibility.
        alpha (optional):
            Significance level for the paired bootstrap CI. Defaults to 0.05.

    Returns:
        model_id -> category -> int rank.
    """
    bootstrap_scores = bootstrap_rank_scores(
        model_results=model_results,
        configs=configs,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    return compute_standard_ranks_from_bootstrap_scores(
        bootstrap_scores=bootstrap_scores, alpha=alpha
    )


def compute_standard_ranks_from_bootstrap_scores(
    bootstrap_scores: dict[str, dict[str, dict[str, np.ndarray]]], alpha: float = 0.05
) -> dict[str, dict[str, int]]:
    """Compute ordinal ranks from pre-computed bootstrap score distributions.

    Uses a paired-bootstrap approach: for each pair of models, computes the
    distribution of differences (model_a - model_b) across bootstrap samples,
    then determines whether model_a is significantly worse than model_b by
    checking if the (1-alpha) percentile of the difference distribution is
    strictly positive.

    Models are grouped into rank groups: two models share the same rank if
    neither is significantly worse than the other. Ranks are assigned in
    order of median bootstrap score (lower = better).

    This method is more statistically principled than the CI-overlap heuristic
    because it directly uses the paired difference distribution.

    Args:
        bootstrap_scores: Pre-computed bootstrap distributions from
            ``bootstrap_rank_scores``. Structure: model_id -> category ->
            language -> np.ndarray of bootstrap scores.
        alpha (optional): Significance level. Defaults to 0.05.

    Returns:
        model_id -> category -> int rank.
    """
    ranks: dict[str, dict[str, int]] = {}

    categories = sorted(
        {
            category
            for model_scores in bootstrap_scores.values()
            for category in model_scores
        }
    )
    for category in categories:
        scored: list[tuple[float, str, np.ndarray]] = []
        for model_id in sorted(bootstrap_scores.keys()):
            if (
                category in bootstrap_scores[model_id]
                and "overall" in bootstrap_scores[model_id][category]
            ):
                samples = bootstrap_scores[model_id][category]["overall"]
                median_score = float(np.median(samples))
                scored.append((median_score, model_id, samples))

        if not scored:
            continue

        scored.sort(key=lambda x: x[0])

        n_models = len(scored)
        is_worse: list[list[bool]] = [[False] * n_models for _ in range(n_models)]

        for i in range(n_models):
            for j in range(i + 1, n_models):
                diff = scored[j][2] - scored[i][2]
                lower_bound = np.percentile(diff, 100 * alpha / 2)
                is_worse[j][i] = lower_bound > 0

        rank_groups: list[list[int]] = [[0]]

        for i in range(1, n_models):
            current_group = rank_groups[-1]
            group_anchor = current_group[0]

            if is_worse[i][group_anchor]:
                rank_groups.append([i])
            else:
                rank_groups[-1].append(i)

        # Flatten rank groups to model -> rank mapping.
        current_rank = 1
        for group in rank_groups:
            for idx in group:
                model_id = scored[idx][1]
                ranks.setdefault(model_id, {})[category] = current_rank
            current_rank += 1

    return ranks
