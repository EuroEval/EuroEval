"""Generate leaderboard CSV files from the EuroEval results."""

import datetime as dt
import json
import logging
import math
import re
import typing as t
from collections import defaultdict
from itertools import chain

import numpy as np
import pandas as pd

from .link_generation import generate_task_link
from .paths import OUTPUT_DIR
from .result_loading import load_processed_results
from .result_processing import extract_model_metadata, group_results_by_model
from .score_computation import compute_ranks, compute_standard_ranks
from .task_metadata import (
    ORTHOGONAL_TASKS,
    official_datasets_for_language,
    task_category,
)
from .utils import convert_to_float, drop_val_duplicates

logger = logging.getLogger(__name__)


def generate_leaderboard(
    leaderboard_name: str,
    language_names: list[str],
    categories: list[t.Literal["generative", "all_models"]],
    force: bool,
) -> None:
    """Generate leaderboard CSV files from the EuroEval results.

    Args:
        leaderboard_name:
            The slug used in output filenames (e.g. ``"danish"``,
            ``"mainland_scandinavian"``).
        language_names:
            The languages the leaderboard covers. Each name must resolve via
            ``euroeval.languages``; the official leaderboard datasets are
            derived from the lib for each.
        categories:
            The categories of leaderboards to generate. Should be a list containing
            "generative" and/or "all_models".
        force:
            Force the generation of the leaderboard, even if no updates are found.
    """
    leaderboard_title = leaderboard_name.replace("_", " ").title()

    logger.info(f"Generating {leaderboard_title} leaderboard...")

    # Derive per-language task→dataset configs from `euroeval`. The canonical
    # task/dataset/metric metadata lives in the library.
    configs: dict[str, dict[str, list[str]]] = {
        language: dict(official_datasets_for_language(language))
        for language in language_names
    }

    datasets = [
        dataset
        for config in configs.values()
        for task_datasets in config.values()
        for dataset in task_datasets
    ]

    # Load results and set them up for the leaderboard
    results = load_processed_results()
    results = [record for record in results if record["dataset"] in datasets]
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]] = (
        group_results_by_model(results=results)
    )
    model_results = drop_val_duplicates(model_results=model_results)
    ranks = compute_ranks(model_results=model_results, configs=configs)
    metadata_dict = extract_model_metadata(results=results)

    # Only include dataset columns in monolingual leaderboards
    include_dataset_columns = len(configs) == 1

    # Generate the leaderboard and store it to disk
    df_pairs = generate_dataframe(
        model_results=model_results,
        ranks=ranks,
        metadata_dict=metadata_dict,
        categories=categories,
        leaderboard_configs=configs,
        include_dataset_columns=include_dataset_columns,
    )

    for category, df_pair in zip(categories, df_pairs):
        df, df_simplified = df_pair

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        leaderboard_path = OUTPUT_DIR / f"{leaderboard_name}_{category}.csv"
        simplified_leaderboard_path = (
            OUTPUT_DIR / f"{leaderboard_name}_{category}_simplified.csv"
        )

        # Check if anything got updated
        new_records: list[str] = list()
        comparison_columns = [
            col
            for col in df.columns
            if col.lower() != "rank" or not include_dataset_columns
        ]
        if leaderboard_path.exists():
            old_df = pd.read_csv(leaderboard_path, header=0, skiprows=1)
            old_df.columns = [
                re.sub(r"<a href=.*?>(.*?)</a>", r"\1", col) for col in old_df.columns
            ]
            if any(col not in old_df.columns for col in comparison_columns):
                new_records = df.Model.tolist()
            else:
                for model_id in set(df.Model.tolist() + old_df.Model.tolist()):
                    old_df_is_missing_columns = any(
                        col not in old_df.columns for col in comparison_columns
                    )
                    if old_df_is_missing_columns:
                        new_records.append(model_id)
                        continue

                    model_is_new = (
                        model_id in df.Model.values
                        and model_id not in old_df.Model.values
                    )
                    model_is_removed = (
                        model_id in old_df.Model.values
                        and model_id not in df.Model.values
                    )
                    if model_is_new or model_is_removed:
                        new_records.append(model_id)
                        continue

                    old_model_results = (
                        old_df[comparison_columns]
                        .query("Model == @model_id")
                        .dropna()
                        .map(convert_to_float)
                    )
                    new_model_results = (
                        df[comparison_columns]
                        .query("Model == @model_id")
                        .dropna()
                        .map(convert_to_float)
                    )
                    model_has_new_results = not np.all(
                        old_model_results.values == new_model_results.values
                    )
                    if model_has_new_results:
                        new_records.append(model_id)
        else:
            new_records = df.Model.tolist()

        # Remove anchor tags from model names
        new_records = [
            re.sub(r"<a href=.*?>(.*?)</a>", r"\1", model) for model in new_records
        ]

        if new_records or force:
            top_header, second_header = create_leaderboard_headers(
                df=df, leaderboard_configs=configs
            )

            df.columns = top_header

            # Add second header as the first row
            df.loc[-1] = second_header
            df.index = df.index + 1
            df.sort_index(inplace=True)
            df = df.fillna("?")

            df.to_csv(leaderboard_path, index=False)
            df_simplified.to_csv(simplified_leaderboard_path, index=False)
            timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            notes = dict(annotate=dict(notes=f"Last updated: {timestamp} CET"))
            with leaderboard_path.with_suffix(".json").open(mode="w") as f:
                json.dump(notes, f, indent=2)
                f.write("\n")
            if not new_records and force:
                logger.info(
                    f"Updated the {category!r} category of the {leaderboard_title} "
                    "leaderboard with no changes."
                )
            elif include_dataset_columns:
                logger.info(
                    f"Updated the following {len(new_records):,} models in the "
                    f"{category!r} category of the {leaderboard_title} leaderboard: "
                    f"{', '.join(new_records)}"
                )
            else:
                logger.info(
                    f"Updated the {leaderboard_title} leaderboard with "
                    f"{len(new_records):,} new or modified models."
                )
        else:
            logger.info(
                f"No updates to the {category!r} category of the {leaderboard_title} "
                "leaderboard."
            )


def create_leaderboard_headers(
    df: pd.DataFrame | pd.Series, leaderboard_configs: dict[str, dict[str, list[str]]]
) -> tuple[list[str], list[str]]:
    """Create the leaderboard headers.

    The first header includes the task types (with links), and the second header
    contains the 'original' header but with html links to the datasets.

    Args:
        df:
            The dataframe.
        leaderboard_configs:
            The leaderboard configurations.

    Returns:
        The first and second header.
    """
    # Extract information from each dataset, and set up an anchor tag template which
    # will replace the dataset column name with a link
    all_datasets = []
    dataset_to_language = {}
    dataset_to_task_info = {}
    for language, tasks in leaderboard_configs.items():
        dataset_link_tag = (
            f"<a href='https://euroeval.com/datasets/{language}#"
            + "{anchor}'>{dataset}</a>"
        )

        language_datasets = list(chain.from_iterable(tasks.values()))
        all_datasets.extend(language_datasets)

        for dataset in language_datasets:
            dataset_to_language[dataset] = (language, dataset_link_tag)

        for task, datasets in tasks.items():
            for dataset in datasets:
                dataset_to_task_info[dataset] = (task, len(datasets))

    orthogonal_tasks = ORTHOGONAL_TASKS

    # Generate column headers
    top_header = []
    second_header = []
    processed_tasks_per_language: dict[str, set[str]] = {}
    seen_version_col = False
    for id_, col in enumerate(df.columns):
        # Case if the column is an orthogonal task
        if (task := col.replace(" ", "-").lower()) in orthogonal_tasks:
            top_header.append("")
            second_header.append(
                f'<a href="https://euroeval.com/tasks/{task}">{col}</a>'
            )

        # Replace dataset columns with task links in the first header, and dataset links
        # in the second header
        elif (leaderboard_col := col.replace("_", "-")) in all_datasets:
            language, dataset_link_tag = dataset_to_language[leaderboard_col]
            task, num_datasets = dataset_to_task_info[leaderboard_col]

            if language not in processed_tasks_per_language:
                processed_tasks_per_language[language] = set()

            if task in processed_tasks_per_language[language]:
                top_header.append("")
                second_header.append(
                    dataset_link_tag.format(anchor=leaderboard_col, dataset=col)
                )
                continue

            task_link = generate_task_link(id_, task)
            if num_datasets > 1:
                task_link = f"~~~{task_link}~~~"

            top_header.append(task_link)
            second_header.append(
                dataset_link_tag.format(anchor=leaderboard_col, dataset=col)
            )
            processed_tasks_per_language[language].add(task)

        # Special case if it's a dataset version column
        else:
            if "version" in col and not seen_version_col:
                top_header.append("<span style='visibility: hidden;'>hidden</span>")
                seen_version_col = True
            else:
                top_header.append("")

            second_header.append(col)

    # Add "Task Type" label to the top-left cell, and make cell (0, 1) invisible to
    # ensure proper alignment
    top_header[0] = (
        "<span style='font-size: 12px; font-weight: normal; opacity: 0.6;'>"
        "Task Type"
        "</span>"
    )
    top_header[1] = "<span style='visibility: hidden;'>dummy</span>"

    return top_header, second_header


def generate_dataframe(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    ranks: dict[str, dict[str, dict[str, dict[str, float]]]],
    metadata_dict: dict[str, dict],
    categories: list[t.Literal["generative", "all_models"]],
    leaderboard_configs: dict[str, dict[str, list[str]]],
    include_dataset_columns: bool,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate DataFrames from the model results.

    Args:
        model_results:
            The model results.
        ranks:
            The ranks of the models (from compute_ranks).
        metadata_dict:
            The metadata.
        categories:
            The categories of leaderboards to generate.
        leaderboard_configs:
            The leaderboard configurations.
        include_dataset_columns:
            Whether to include dataset columns in the DataFrame.

    Returns:
        A list of pairs (df, df_simplified), where df is the full leaderboard DataFrame
        and df_simplified is the simplified version.
    """
    if model_results == {}:
        logger.error("No model results found, skipping leaderboard generation.")
        return list()

    # Mapping from category to dataset names. The "generative" leaderboard
    # includes all tasks; the "all_models" leaderboard is restricted to NLU
    # tasks so non-generative models can be compared.
    def _include(category: str, task: str) -> bool:
        return category == "generative" or task_category(task) == "nlu"

    category_to_datasets = {
        category: [
            dataset
            for config in leaderboard_configs.values()
            for task, task_datasets in config.items()
            for dataset in task_datasets
            if _include(category, task)
        ]
        for category in categories
    }

    # Mapping from orthogonal dataset to orthogonal task
    category_to_orthogonal_datasets = {
        category: {
            dataset: task
            for config in leaderboard_configs.values()
            for task, task_datasets in config.items()
            for dataset in task_datasets
            if task in ORTHOGONAL_TASKS and _include(category, task)
        }
        for category in categories
    }

    dfs: list[tuple[pd.DataFrame, pd.DataFrame]] = list()
    for category in categories:
        # Standard (dense) ranks are computed per category over the models
        # eligible for display — i.e. those holding every non-orthogonal
        # dataset for this category. Ranking the full set and then hiding
        # ineligible rows would leave gaps in the 1, 2, 3, … sequence.
        required_datasets = [
            ds
            for ds in category_to_datasets[category]
            if ds not in category_to_orthogonal_datasets[category]
        ]
        eligible_model_results = {
            mid: r
            for mid, r in model_results.items()
            if all(ds in r for ds in required_datasets)
        }
        standard_ranks = compute_standard_ranks(
            model_results=eligible_model_results, ranks=ranks
        )

        data_dict: dict[str, list] = defaultdict(list)
        for model_id, results in model_results.items():
            has_all_datasets = model_id in eligible_model_results

            # Get the overall rank for the model (standard ordinal rank)
            rank = standard_ranks.get(model_id, math.nan)
            # Get the mean rank score with CI
            cat_ranks = ranks.get(model_id, {}).get(category, {})
            rank_data = cat_ranks.get("overall", {})
            rank_score = rank_data.get("score", float("nan"))
            rank_data.get("ci_lower", float("nan"))
            rank_ci_upper = rank_data.get("ci_upper", float("nan"))
            if math.isfinite(rank_score) and has_all_datasets:
                margin = (
                    (rank_ci_upper - rank_score)
                    if math.isfinite(rank_ci_upper)
                    else 0.0
                )
                mean_rank_score_str = f"{rank_score:.2f} \u00b1 {margin:.2f}"
            else:
                mean_rank_score_str = "-"
                rank = math.nan
            language_ranks = cat_ranks.copy()
            language_ranks.pop("overall", None)
            # Ensure all languages are present (even if missing for this model)
            for lang in leaderboard_configs:
                if lang not in language_ranks:
                    language_ranks[lang] = float("nan")

            # Extract score values from language_ranks dicts for CSV output
            language_ranks_scores = {
                lang: (entry["score"] if isinstance(entry, dict) else entry)
                for lang, entry in language_ranks.items()
            }

            # Get the default values for the dataset columns
            default_dataset_values = {
                ds: float("nan") for ds in category_to_datasets[category]
            } | {f"{ds}_version": "-" for ds in category_to_datasets[category]}
            default_orthogonal_values = {
                task: float("nan")
                for task in category_to_orthogonal_datasets[category].values()
            }

            # Get individual dataset scores for the model
            total_results = dict()
            orthogonal_scores = defaultdict(list)  # task -> list of scores
            for dataset in category_to_datasets[category]:
                if dataset in results:
                    scores = results[dataset]
                else:
                    scores = [(list(), float("nan"), 0)]
                main_score = scores[0][1]
                if not math.isnan(main_score):
                    score_str = " / ".join(
                        f"{total_score:,.2f} ± {std_err:,.2f}"
                        for _, total_score, std_err in scores
                    )
                    if dataset in category_to_orthogonal_datasets[category]:
                        orthogonal_task = category_to_orthogonal_datasets[category][
                            dataset
                        ]
                        orthogonal_scores[orthogonal_task].append(main_score)
                else:
                    score_str = "-"
                total_results[dataset] = score_str

            # Aggregate orthogonal scores by taking their mean
            orthogonal_task_scores = {
                task: np.mean(score_list).item()
                if len(score_list) > 0
                else float("nan")
                for task, score_list in orthogonal_scores.items()
            }

            # Filter metadata dict to only keep the dataset versions belonging to the
            # category
            metadata = {
                key: value
                for key, value in metadata_dict[model_id].items()
                if not key.endswith("_version")
                or key.replace("_version", "") in category_to_datasets[category]
            }

            # Add all the model values to the data dictionary
            model_values = (
                dict(model=model_id, rank=rank, mean_rank_score=mean_rank_score_str)
                | default_orthogonal_values
                | default_dataset_values
                | orthogonal_task_scores
                | metadata
                | language_ranks_scores
                | total_results
            )
            for key, value in model_values.items():
                if isinstance(value, float):
                    value = round(value, 2)
                data_dict[key].append(value)

            # Sanity check that all values have the same length
            assert len({len(values) for values in data_dict.values()}) == 1, (
                f"Length of data_dict values must be equal, but got "
                f"{dict([(key, len(values)) for key, values in data_dict.items()])}."
            )

        # Create dataframe and sort by rank (numeric, NaN sinks to the bottom)
        df = (
            pd.DataFrame(data_dict)
            .sort_values(by="rank", na_position="last")
            .reset_index(drop=True)
        )

        rank_cols = ["rank", "mean_rank_score"]
        if len(leaderboard_configs) > 1:
            rank_cols += list(leaderboard_configs.keys())

        # Format rank as a display string with a "-" sentinel for NaN.
        # The "rank" column is an integer (ordinal rank); "mean_rank_score"
        # is already formatted as "score \u00b1 margin".
        for col in rank_cols:
            if col == "mean_rank_score":
                # Already formatted, just replace NaN with "-"
                df[col] = [
                    v if isinstance(v, str) and v != "-" else "-" for v in df[col]
                ]
            else:
                df[col] = [
                    str(int(value)) if math.isfinite(value) else "-"
                    for value in df[col]
                ]

        # Replace dashes with underlines in all column names
        df.columns = df.columns.str.replace("-", "_")

        # Reorder columns
        orthogonal_cols = list(
            {
                orthogonal_task.replace("-", "_")
                for orthogonal_task in category_to_orthogonal_datasets[
                    category
                ].values()
            }
        )
        dataset_cols = [
            dataset.replace("-", "_")
            for dataset in category_to_datasets[category]
            if dataset not in category_to_orthogonal_datasets[category]
        ]
        cols = (
            ["rank", "model", "mean_rank_score"]
            + orthogonal_cols
            + [
                "generative_type",
                "open",
                "commercial",
                "merge",
                "trained_from_scratch",
                "parameters",
                "vocabulary_size",
                "context",
            ]
            + rank_cols[2:]
        )
        if include_dataset_columns:
            cols += dataset_cols
            cols += [f"{dataset}_version" for dataset in dataset_cols]
        df = df[cols]

        # If a model has only orthogonal values, we remove it from the leaderboard
        num_before = len(df)
        value_cols = [col for col in dataset_cols + rank_cols[1:] if col in df.columns]
        model_ids_with_dataset_values = df.query(
            " or ".join([f"({col} != '-')" for col in value_cols])
        ).model.tolist()
        model_ids_with_orthogonal_values = df[
            df[orthogonal_cols].notna().any(axis=1)
        ].model.tolist()
        model_ids_to_drop = set(model_ids_with_orthogonal_values) - set(
            model_ids_with_dataset_values
        )
        df = df[~df.model.isin(model_ids_to_drop)].reset_index(drop=True)
        num_after = len(df)
        if num_after < num_before:
            logger.info(
                f"Dropped {num_before - num_after:,} models from the {category!r} "
                "leaderboard that had only orthogonal scores but no dataset scores."
            )

        # Replace Boolean values by ✓ and ✗
        boolean_columns = ["commercial", "merge", "open"]
        for col in boolean_columns:
            df[col] = df[col].apply(lambda x: "✓" if x else "✗")

        # Convert trained_from_scratch values to symbols
        trained_mapping = {True: "✓", False: "✗"}
        df["trained_from_scratch"] = df["trained_from_scratch"].map(trained_mapping)

        # Orthogonal values only makes sense for instruction-tuned and reasoning models,
        # so we set the value to "N/A" for other model types
        for orthogonal_task in category_to_orthogonal_datasets[category].values():
            col_name = orthogonal_task.replace("-", "_")
            df[col_name] = df.apply(
                lambda row: (
                    row[col_name]
                    if row.generative_type in ["instruction_tuned", "reasoning"]
                    else "N/A"
                ),
                axis=1,
            )

        # Replace generative_type with emojis
        generative_type_emoji_mapping = {
            "base": "🧠",
            "instruction_tuned": "📝",
            "reasoning": "🤔",
        }
        df["generative_type"] = df.generative_type.map(
            lambda x: generative_type_emoji_mapping.get(x, "🔍")
        )

        # Create the simplified leaderboard
        df_simplified = df.copy()
        df_simplified = df[
            [
                "rank",
                "model",
                "mean_rank_score",
                "generative_type",
                "open",
                "commercial",
                "merge",
                "trained_from_scratch",
                "parameters",
                "vocabulary_size",
                "context",
            ]
        ]
        df_simplified = df_simplified.query(  # pyrefly: ignore[not-callable]
            "rank != '-'"
        )
        df_simplified = df_simplified.convert_dtypes()

        # Format headers for display
        renaming_dict = (
            {
                "model": "Model",
                "generative_type": "Type",
                "rank": "Rank",
                "parameters": "Parameters",
                "vocabulary_size": "Vocabulary",
                "context": "Context",
                "commercial": "Commercial",
                "merge": "Merge",
                "open": "Open",
                "trained_from_scratch": "Trained from scratch",
            }
            | {"mean_rank_score": "Mean rank score"}
            | {rank_col: rank_col.title() for rank_col in rank_cols[2:]}
            | {
                orthogonal_task.replace("-", "_"): orthogonal_task.replace(
                    "-", " "
                ).title()
                for orthogonal_task in category_to_orthogonal_datasets[
                    category
                ].values()
            }
        )
        df = df.rename(renaming_dict, axis="columns")

        assert isinstance(df, pd.DataFrame)
        dfs.append((df, df_simplified))

    return dfs
