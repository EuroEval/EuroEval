"""Functions related to computation of scores based on the model results."""

import math
from collections import defaultdict

import numpy as np

from .utils import significantly_better


def compute_ranks(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    task_config: dict[str, dict[str, str]],
    configs: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute the ranks of the models.

    Args:
        model_results:
            The model results.
        task_config:
            The task configuration.
        configs:
            The leaderboard configurations for each language.

    Returns:
        The ranks of the models, per task category and per language. The dict structure
        is model_id -> category -> language/overall -> language_rank.
    """
    # These tasks are not involved in the ranking calculation, and should just be
    # presented with their raw scores instead.
    orthogonal_tasks = [
        task
        for task, task_dct in task_config.items()
        if task_dct.get("orthogonal", False)
    ]

    all_datasets = {
        language: [
            dataset
            for task, task_datasets in config.items()
            for dataset in task_datasets
            if task not in orthogonal_tasks
        ]
        for language, config in configs.items()
    }

    # Compute ranks for each dataset.
    # model_id -> dataset -> dataset_rank
    model_dataset_ranks: dict[str, dict[str, float]] = defaultdict(dict)
    for _, datasets in all_datasets.items():
        for dataset in datasets:
            dummy_scores: list[tuple[list[float], float, float]] = [
                ([], float("nan"), 0)
            ]
            model_dataset_scores = [
                (model_id, *scores.get(dataset, dummy_scores)[0])
                for model_id, scores in model_results.items()
            ]
            model_dataset_scores = sorted(
                [x for x in model_dataset_scores if not np.isnan(x[-2])],
                key=lambda x: x[-2],
                reverse=True,
            ) + [x for x in model_dataset_scores if np.isnan(x[-2])]
            stddev = np.std(
                [
                    score
                    for _, _, score, _ in model_dataset_scores
                    if not np.isnan(score)
                ]
            )

            rank_score = 1.0
            previous_scores: list[float] = list()
            for model_id, raw_scores, _, _ in model_dataset_scores:
                if raw_scores == []:
                    model_dataset_ranks[model_id][dataset] = math.inf
                    continue
                elif previous_scores == []:
                    previous_scores = raw_scores
                elif significantly_better(previous_scores, raw_scores):
                    difference = np.mean(previous_scores) - np.mean(raw_scores)
                    normalised_difference = difference / stddev
                    rank_score += normalised_difference.item()
                    previous_scores = raw_scores
                model_dataset_ranks[model_id][dataset] = rank_score

    # Aggregate dataset ranks into task ranks.
    # model_id -> language -> task -> task_rank
    model_task_ranks: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for model_id, dataset_ranks in model_dataset_ranks.items():
        for language, config in configs.items():
            for task, datasets in config.items():
                model_task_ranks[model_id][language][task] = np.mean(
                    [
                        dataset_ranks[dataset]
                        for dataset in datasets
                        if dataset in dataset_ranks
                    ]
                ).item()

    categories = {
        task_config[task]["category"] for config in configs.values() for task in config
    } | {"all"}

    # Aggregate task ranks into language ranks.
    # model_id -> category -> language/overall -> language_rank
    model_task_category_ranks: dict[str, dict[str, dict[str, float]]] = defaultdict(
        dict
    )
    for model_id, score_dict in model_task_ranks.items():
        for category in categories:
            language_scores = [
                np.mean(
                    [
                        score_dict[language][task]
                        for task in config
                        if (
                            task_config[task]["category"] == category
                            or category == "all"
                        )
                        and task not in orthogonal_tasks
                    ]
                ).item()
                for language, config in configs.items()
                if any(
                    task_config[task]["category"] == category or category == "all"
                    for task in config
                )
            ]
            model_rank_scores = dict(overall=np.mean(language_scores).item())
            if len(language_scores) > 1:
                model_rank_scores |= dict(zip(configs.keys(), language_scores))
            model_task_category_ranks[model_id][category] = model_rank_scores

    return model_task_category_ranks
