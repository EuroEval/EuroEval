"""Functions related to computation of scores based on the model results."""

import math
from collections import defaultdict

import numpy as np

from .bootstrap_cis import bootstrap_confidence_intervals, bootstrap_rank_scores
from .task_metadata import ORTHOGONAL_TASKS, task_category


def _category_includes_task(category: str, task: str) -> bool:
    return category == "generative" or task_category(task) == "nlu"


_CATEGORIES = ("generative", "all_models")


def compute_dataset_ranks(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    configs: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Compute per-dataset rank scores (step 1 of compute_ranks).

    Returns:
        model_id -> category -> dataset -> {"score", "ci_lower", "ci_upper"}.
    """
    out: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for _language, config in configs.items():
        for category in _CATEGORIES:
            datasets = [
                ds
                for task, task_ds in config.items()
                for ds in task_ds
                if task not in ORTHOGONAL_TASKS
                and _category_includes_task(category, task)
            ]
            for dataset in datasets:
                model_scores: dict[str, tuple[float, float, list[float]]] = {}
                for model_id, results in model_results.items():
                    if dataset in results and results[dataset]:
                        raw, mean_sc, se = results[dataset][0]
                        if np.isfinite(mean_sc):
                            model_scores[model_id] = (mean_sc, se, raw)

                if not model_scores:
                    continue

                # Sort by mean score descending — best model is first
                sorted_models = sorted(
                    model_scores.items(), key=lambda x: x[1][0], reverse=True
                )
                mean_best = sorted_models[0][1][0]
                all_raw = [r for _, _, r in model_scores.values() for r in r]
                pooled_sd = np.std(all_raw) if len(all_raw) > 1 else 1.0

                for mid, (mean_sc, se, _) in model_scores.items():
                    diff = (mean_best - mean_sc) / pooled_sd
                    var_diff = (se**2) / (pooled_sd**2) if pooled_sd > 0 else 0.0
                    score = 1.0 + diff
                    margin = 1.96 * math.sqrt(var_diff)
                    out[mid][category][dataset] = {
                        "score": round(score, 6),
                        "ci_lower": round(score - margin, 6),
                        "ci_upper": round(score + margin, 6),
                    }

    return out


def compute_ranks(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    configs: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Compute ranks via mean-of-normalised-differences with propagated CIs.

    Args:
        model_results:
            The model results.
        configs:
            The leaderboard configurations for each language.

    Returns:
        The ranks of the models, per task category and per language.
        The dict structure is model_id -> category -> language/overall ->
        {"score", "ci_lower", "ci_upper"}.
    """
    orthogonal_tasks = ORTHOGONAL_TASKS
    categories = _CATEGORIES

    def category_includes_task(category: str, task: str) -> bool:
        return _category_includes_task(category, task)

    # ── Step 1: Dataset-level ranks ──────────────────────────────────────
    model_dataset_ranks = compute_dataset_ranks(
        model_results=model_results, configs=configs
    )

    # ── Step 2: Aggregate dataset → task → language → overall ────────────
    model_task_ranks: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for model_id in model_dataset_ranks:
        for category in categories:
            if (
                model_id not in model_dataset_ranks
                or category not in model_dataset_ranks[model_id]
            ):
                continue

            for language, config in configs.items():
                for task, task_datasets in config.items():
                    if task in orthogonal_tasks:
                        continue
                    if not category_includes_task(category, task):
                        continue

                    entries = [
                        model_dataset_ranks[model_id][category][ds]
                        for ds in task_datasets
                        if ds in model_dataset_ranks[model_id][category]
                    ]
                    if not entries:
                        continue

                    mean_score = np.mean([e["score"] for e in entries]).item()
                    vars_ = [
                        ((e["ci_upper"] - e["ci_lower"]) / (2 * 1.96)) ** 2
                        for e in entries
                    ]
                    mean_var = np.sum(vars_) / (len(entries) ** 2)
                    margin = 1.96 * math.sqrt(mean_var)

                    model_task_ranks[model_id][category][language][task] = {
                        "score": round(mean_score, 6),
                        "ci_lower": round(mean_score - margin, 6),
                        "ci_upper": round(mean_score + margin, 6),
                    }

    # ── Step 3: Aggregate task → language → overall ──────────────────────
    final: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    for model_id in model_dataset_ranks:
        for category in categories:
            if (
                model_id not in model_dataset_ranks
                or category not in model_dataset_ranks[model_id]
            ):
                continue

            lang_scores: dict[str, dict[str, float]] = {}
            overall_entries: list[dict[str, float]] = []

            for language, config in configs.items():
                task_entries = [
                    model_task_ranks[model_id][category][language].get(task)
                    for task in config
                    if task not in orthogonal_tasks
                    and category_includes_task(category, task)
                ]
                task_entries = [e for e in task_entries if e is not None]
                if not task_entries:
                    continue

                mean_score = np.mean([e["score"] for e in task_entries]).item()
                vars_ = [
                    ((e["ci_upper"] - e["ci_lower"]) / (2 * 1.96)) ** 2
                    for e in task_entries
                ]
                mean_var = np.sum(vars_) / (len(task_entries) ** 2)
                margin = 1.96 * math.sqrt(mean_var)

                lang_scores[language] = {
                    "score": round(mean_score, 6),
                    "ci_lower": round(mean_score - margin, 6),
                    "ci_upper": round(mean_score + margin, 6),
                }
                overall_entries.append(lang_scores[language])

            if overall_entries:
                mean_score = np.mean([e["score"] for e in overall_entries]).item()
                vars_ = [
                    ((e["ci_upper"] - e["ci_lower"]) / (2 * 1.96)) ** 2
                    for e in overall_entries
                ]
                mean_var = np.sum(vars_) / (len(overall_entries) ** 2)
                margin = 1.96 * math.sqrt(mean_var)

                lang_scores["overall"] = {
                    "score": round(mean_score, 6),
                    "ci_lower": round(mean_score - margin, 6),
                    "ci_upper": round(mean_score + margin, 6),
                }
                final.setdefault(model_id, {})[category] = lang_scores

    return final


def _anchor_significantly_better(
    anchor_overall: dict[str, float], candidate_overall: dict[str, float]
) -> bool | None:
    """Test whether the anchor's mean rank score is significantly lower.

    Uses the propagated 95% confidence intervals that are shown in the
    leaderboard's Rank score column: the anchor is significantly better iff
    its upper CI lies strictly below the candidate's lower CI (i.e. the two
    intervals do not overlap). This keeps the tie-detection consistent with
    what the reader sees — two models share a rank iff their displayed
    "score ± margin" intervals overlap.

    Returns:
        True if the anchor is significantly better, False if not, or None
        when either side is missing the CI.
    """
    a_upper = anchor_overall.get("ci_upper", float("nan"))
    c_lower = candidate_overall.get("ci_lower", float("nan"))
    if math.isfinite(a_upper) and math.isfinite(c_lower):
        return a_upper < c_lower
    return None


def compute_standard_ranks(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    ranks: dict[str, dict[str, dict[str, dict[str, float]]]],
    category: str,
) -> dict[str, int]:
    """Compute ordinal ranks (1, 2, 3…) with ties via non-overlapping CIs.

    Sorts by overall mean rank score ascending (lower = better). Walks down
    the list; if the candidate's lower CI overlaps the anchor's upper CI it
    shares the anchor's rank, otherwise it starts a new tie group with rank
    one higher.

    Using the propagated 95% confidence intervals that are already shown in
    the Rank score column keeps the ordinal Rank consistent with what the
    reader sees: two models share a rank iff their displayed "score ±
    margin" intervals overlap.

    Args:
        model_results:
            The model results (used only to gate models on having any data).
        ranks:
            The output of `compute_ranks` — overall mean rank score and
            propagated CIs per model and category.
        category:
            Which leaderboard category ("generative" or "all_models") to
            rank within.

    Returns:
        model_id -> int rank.
    """
    # Collect (model_id, overall_score) for finite-scored models that also
    # appear in model_results — callers may pre-filter model_results to the
    # subset eligible for ranking (e.g. those holding every required dataset).
    scored: list[tuple[str, float]] = []
    for model_id, cats in ranks.items():
        if model_id not in model_results:
            continue
        if category in cats and "overall" in cats[category]:
            s = cats[category]["overall"]["score"]
            if math.isfinite(s):
                scored.append((model_id, s))

    scored.sort(key=lambda x: x[1])

    if not scored:
        return {}

    # Dense ranking: ties share a rank, and the next distinct group's rank
    # is just the previous group's rank + 1 (no gaps).
    ranks_out: dict[str, int] = {}
    ranks_out[scored[0][0]] = 1
    anchor_idx = 0
    current_rank = 1

    for i in range(1, len(scored)):
        mid_i = scored[i][0]
        anchor_id = scored[anchor_idx][0]

        verdict = _anchor_significantly_better(
            anchor_overall=ranks[anchor_id][category]["overall"],
            candidate_overall=ranks[mid_i][category]["overall"],
        )
        # Treat "no basis for comparison" as a new group, matching the prior
        # behaviour for non-overlapping evaluation sets.
        is_new_group = verdict is None or verdict
        if is_new_group:
            current_rank += 1
            ranks_out[mid_i] = current_rank
            anchor_idx = i
        else:
            ranks_out[mid_i] = ranks_out[anchor_id]

    return ranks_out


# ---------------------------------------------------------------------------
# Bootstrap-based rank computation
# ---------------------------------------------------------------------------


def compute_ranks_bootstrap(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    configs: dict[str, dict[str, list[str]]],
    n_bootstraps: int = 1000,
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
        n_bootstraps: Number of bootstrap replicates (default 1000).
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
    n_bootstraps: int = 1000,
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
    # Step 1: Compute bootstrap distributions
    bootstrap_scores = bootstrap_rank_scores(
        model_results=model_results,
        configs=configs,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    # Step 2: Compute CIs from bootstrap distributions and sort models
    ranks: dict[str, dict[str, int]] = {}
    categories = ["generative", "all_models"]

    for category in categories:
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

        # Step 3: Walk down and assign ranks via CI overlap
        current_rank = 1
        anchor_idx = 0

        for i in range(len(scored)):
            mid_i = scored[i][1]

            if i == 0:
                ranks.setdefault(mid_i, {})[category] = 1
                continue

            scored[anchor_idx][1]
            anchor_ci_upper = scored[anchor_idx][3]
            candidate_ci_lower = scored[i][2]

            # CI overlap: if candidate's lower CI > anchor's upper CI, no overlap
            if candidate_ci_lower > anchor_ci_upper:
                # No overlap → new rank group
                current_rank += 1
                anchor_idx = i

            ranks[mid_i] = ranks.setdefault(mid_i, {})
            ranks[mid_i][category] = current_rank

    return ranks


# ---------------------------------------------------------------------------
