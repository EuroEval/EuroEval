"""Functions related to computation of scores based on the model results."""

import logging
import math
from collections import defaultdict

import numpy as np

from euroeval.constants import ORTHOGONAL_TASKS

from .bootstrap_cis import bootstrap_confidence_intervals, bootstrap_rank_scores
from .constants import LEADERBOARD_CATEGORIES, Z_SCORE_95
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
    model_task_ranks = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for model_id in model_dataset_ranks:
        for category in categories:
            if category not in model_dataset_ranks[model_id]:
                continue

            for language, config in configs.items():
                for task, task_datasets in config.items():
                    if task in orthogonal_tasks:
                        continue
                    if not category_includes_task(category=category, task=task):
                        continue

                    entries = [
                        model_dataset_ranks[model_id][category][ds]
                        for ds in task_datasets
                        if ds in model_dataset_ranks[model_id][category]
                    ]
                    if not entries:
                        continue

                    mean_score = float(np.mean([e["score"] for e in entries]))
                    vars_ = [
                        ((e["ci_upper"] - e["ci_lower"]) / (2 * Z_SCORE_95)) ** 2
                        for e in entries
                    ]
                    mean_var = np.sum(vars_) / (len(entries) ** 2)
                    margin = Z_SCORE_95 * math.sqrt(mean_var)

                    model_task_ranks[model_id][category][language][task] = {
                        "score": round(mean_score, 6),
                        "ci_lower": round(mean_score - margin, 6),
                        "ci_upper": round(mean_score + margin, 6),
                    }

    # Step 3: Aggregate task -> language -> overall.
    final: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    for model_id in model_dataset_ranks:
        for category in categories:
            if category not in model_dataset_ranks[model_id]:
                continue

            lang_scores: dict[str, dict[str, float]] = {}
            overall_entries: list[dict[str, float]] = []

            for language, config in configs.items():
                task_entries = [
                    model_task_ranks[model_id][category][language].get(task)
                    for task in config
                    if task not in orthogonal_tasks
                    and category_includes_task(category=category, task=task)
                ]
                task_entries = [e for e in task_entries if e is not None]
                if not task_entries:
                    continue

                mean_score = float(np.mean([e["score"] for e in task_entries]))
                vars_ = [
                    ((e["ci_upper"] - e["ci_lower"]) / (2 * Z_SCORE_95)) ** 2
                    for e in task_entries
                ]
                mean_var = np.sum(vars_) / (len(task_entries) ** 2)
                margin = Z_SCORE_95 * math.sqrt(mean_var)

                lang_scores[language] = {
                    "score": round(mean_score, 6),
                    "ci_lower": round(mean_score - margin, 6),
                    "ci_upper": round(mean_score + margin, 6),
                }
                overall_entries.append(lang_scores[language])

            if overall_entries:
                mean_score = float(np.mean([e["score"] for e in overall_entries]))
                vars_ = [
                    ((e["ci_upper"] - e["ci_lower"]) / (2 * Z_SCORE_95)) ** 2
                    for e in overall_entries
                ]
                mean_var = np.sum(vars_) / (len(overall_entries) ** 2)
                margin = Z_SCORE_95 * math.sqrt(mean_var)

                lang_scores["overall"] = {
                    "score": round(mean_score, 6),
                    "ci_lower": round(mean_score - margin, 6),
                    "ci_upper": round(mean_score + margin, 6),
                }
                final.setdefault(model_id, {})[category] = lang_scores

    logger.info("Finished computing ranks.")
    return final


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

    for _language, config in configs.items():
        for category in LEADERBOARD_CATEGORIES:
            datasets = [
                ds
                for task, task_ds in config.items()
                for ds in task_ds
                if task not in ORTHOGONAL_TASKS
                and category_includes_task(category, task)
            ]
            for dataset in datasets:
                model_scores: dict[str, tuple[float, list[float]]] = {}
                for model_id, results in model_results.items():
                    if dataset in results and results[dataset]:
                        raw, mean_sc, _ = results[dataset][0]
                        if np.isfinite(mean_sc):
                            model_scores[model_id] = (mean_sc, raw)

                if not model_scores:
                    continue

                # Sort by mean score descending, so the best model is first.
                sorted_models = sorted(
                    model_scores.items(), key=lambda x: x[1][0], reverse=True
                )
                mean_best = sorted_models[0][1][0]
                all_raw = [
                    score
                    for _, raw_scores in model_scores.values()
                    for score in raw_scores
                ]
                pooled_sd = np.std(all_raw) if len(all_raw) > 1 else 1.0
                if pooled_sd <= 0:
                    pooled_sd = 1.0

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
                    out[mid][category][dataset] = {
                        "score": round(score, 6),
                        "ci_lower": round(ci_lower, 6),
                        "ci_upper": round(ci_upper, 6),
                    }

    return out


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
    """Compute ordinal ranks (1, 2, 3…) with ties via bootstrap CIs.

    Sorts by overall mean rank score ascending (lower = better). Walks down
    the list; for each model, compares its CI against the current anchor's CI.
    If the candidate's lower CI bound is strictly above the anchor's upper CI
    bound (no overlap), it starts a new rank group. Otherwise it shares the
    anchor's rank.

    CIs are computed from the bootstrap distributions themselves (percentile
    method), so the Rank column is consistent with the statistical uncertainty
    captured by the bootstrap — rather than using a bootstrap hypothesis test
    which operates on raw distributions and can disagree with the displayed CIs.

    Args:
        model_results: The model results.
        configs: Per-language task -> dataset mappings.
        n_bootstraps: Number of bootstrap replicates.
        seed: Random seed for reproducibility.
        alpha: Significance level for the bootstrap CI.

    Returns:
        model_id -> category -> int rank.
    """
    # Step 1: Compute bootstrap distributions.
    bootstrap_scores = bootstrap_rank_scores(
        model_results=model_results,
        configs=configs,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    # Step 2: Compute CIs from bootstrap distributions and sort models.
    ranks: dict[str, dict[str, int]] = {}

    for category in LEADERBOARD_CATEGORIES:
        scored: list[tuple[float, str, float, float]] = []
        for model_id in model_results:
            if (
                model_id in bootstrap_scores
                and category in bootstrap_scores[model_id]
                and "overall" in bootstrap_scores[model_id][category]
            ):
                samples = bootstrap_scores[model_id][category]["overall"]
                mean_score = float(np.median(samples))
                ci_lower = float(np.percentile(samples, alpha * 50))
                ci_upper = float(np.percentile(samples, (1 - alpha) * 100))
                scored.append((mean_score, model_id, ci_lower, ci_upper))

        scored.sort(key=lambda x: x[0])

        if not scored:
            continue

        # Step 3: Walk down and assign ranks via CI overlap. Two models share a
        # rank iff their bootstrap CIs overlap; a strictly higher lower bound
        # starts a new rank group.
        current_rank = 1
        anchor_idx = 0
        anchor_id = scored[0][1]
        ranks.setdefault(scored[0][1], {})[category] = 1

        for i in range(1, len(scored)):
            candidate_id = scored[i][1]
            anchor_ci_upper = scored[anchor_idx][3]
            candidate_ci_lower = scored[i][2]

            if candidate_ci_lower > anchor_ci_upper:
                current_rank += 1
                anchor_idx = i
                anchor_id = candidate_id
                ranks.setdefault(candidate_id, {})[category] = current_rank
            else:
                ranks.setdefault(candidate_id, {})[category] = ranks[anchor_id][
                    category
                ]

    return ranks
