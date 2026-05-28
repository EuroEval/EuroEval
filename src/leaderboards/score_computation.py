"""Functions related to computation of scores based on the model results."""

import math
from collections import defaultdict

import numpy as np

from .task_metadata import ORTHOGONAL_TASKS, task_category
from .utils import significantly_better


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
    categories = {"generative", "all_models"}

    def category_includes_task(category: str, task: str) -> bool:
        return category == "generative" or task_category(task) == "nlu"

    # ── Step 1: Dataset-level ranks ──────────────────────────────────────
    model_dataset_ranks: dict[str, dict[str, dict[str, dict[str, float]]]] = (
        defaultdict(lambda: defaultdict(dict))
    )

    for language, config in configs.items():
        for category in categories:
            datasets = [
                ds
                for task, task_ds in config.items()
                for ds in task_ds
                if task not in orthogonal_tasks
                and category_includes_task(category, task)
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
                    model_dataset_ranks[mid][category][dataset] = {
                        "score": round(score, 6),
                        "ci_lower": round(score - margin, 6),
                        "ci_upper": round(score + margin, 6),
                    }

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


def compute_standard_ranks(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    ranks: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> dict[str, int]:
    """Compute ordinal ranks (1, 2, 3…) with ties using Welch's t-test.

    Sorts by overall score ascending (better = lower). Walks from the top;
    if the current model is not significantly better than the anchor, it shares
    the anchor's rank. Otherwise it gets a new rank = position + 1.

    Uses significantly_better() to compare raw scores across common datasets.

    Args:
        model_results:
            The model results (for raw score comparison).
        ranks:
            The computed ranks from compute_ranks (for sorting).

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
        for cat in ("generative", "all_models"):
            if cat in cats and "overall" in cats[cat]:
                s = cats[cat]["overall"]["score"]
                if math.isfinite(s):
                    scored.append((model_id, s))
                    break

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

        # Gather comparable raw scores across common datasets
        scores_i, scores_anchor = [], []
        for ds in model_results[anchor_id]:
            if ds in model_results[mid_i]:
                for (raw_i, _, _), (raw_a, _, _) in zip(
                    model_results[mid_i][ds], model_results[anchor_id][ds]
                ):
                    if raw_i and raw_a:
                        scores_i.extend(raw_i)
                        scores_anchor.extend(raw_a)

        have_comparable = (
            bool(scores_i)
            and bool(scores_anchor)
            and len(scores_i) == len(scores_anchor)
        )
        # Promote to a new rank when the anchor is significantly better, or
        # when we have no comparable data (treat as a distinct group).
        is_new_group = not have_comparable or significantly_better(
            scores_anchor, scores_i
        )
        if is_new_group:
            current_rank += 1
            ranks_out[mid_i] = current_rank
            anchor_idx = i
        else:
            ranks_out[mid_i] = ranks_out[anchor_id]

    return ranks_out
