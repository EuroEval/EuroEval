"""Generate leaderboard CSV files from the EuroEval results."""

import datetime as dt
import json
import logging
import math
import re
import typing as t
from collections import defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
from yaml import safe_load

from .link_generation import generate_task_link
from .result_loading import load_processed_results
from .result_processing import extract_model_metadata, group_results_by_model
from .score_computation import compute_ranks
from .utils import convert_to_float

logger = logging.getLogger(__name__)


def generate_leaderboard(
    leaderboard_config_path: Path,
    categories: list[t.Literal["all", "nlu"]],
    force: bool,
) -> None:
    """Generate leaderboard CSV files from the EuroEval results.

    Args:
        leaderboard_config_path:
            The path to the leaderboard configuration file.
        categories:
            The categories of leaderboards to generate. Should be a list containing
            "all" and/or "nlu".
        force:
            Force the generation of the leaderboard, even if no updates are found.
    """
    leaderboard_title = leaderboard_config_path.stem.replace("_", " ").title()

    logger.info(f"Generating {leaderboard_title} leaderboard...")

    # Load configs
    with Path("task_config.yaml").open(mode="r") as f:
        task_config: dict[str, dict[str, str]] = safe_load(stream=f)
    with leaderboard_config_path.open(mode="r") as f:
        config: dict[str, list[str]] = safe_load(stream=f)

    # If the config consists of multiple languages, we extract a dictionary with config
    # for each constituent language
    configs: dict[str, dict[str, list[str]]] = dict()
    if "languages" in config:
        for language in config["languages"]:
            with Path(f"leaderboard_configs/{language}.yaml").open(mode="r") as f:
                configs[language] = safe_load(stream=f)
    else:
        configs = {leaderboard_config_path.stem: config}

    del config

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
        group_results_by_model(results=results, task_config=task_config)
    )
    ranks = compute_ranks(
        model_results=model_results, task_config=task_config, configs=configs
    )
    metadata_dict = extract_model_metadata(results=results)

    # Only include dataset columns in monolingual leaderboards
    include_dataset_columns = len(configs) == 1

    # Generate the leaderboard and store it to disk
    df_pairs = generate_dataframe(
        model_results=model_results,
        ranks=ranks,
        metadata_dict=metadata_dict,
        categories=categories,
        task_config=task_config,
        leaderboard_configs=configs,
        include_dataset_columns=include_dataset_columns,
    )

    for category, df_pair in zip(categories, df_pairs):
        df, df_simplified = df_pair

        leaderboard_path = (
            Path("leaderboards") / f"{leaderboard_config_path.stem}_{category}.csv"
        )
        simplified_leaderboard_path = (
            Path("leaderboards")
            / f"{leaderboard_config_path.stem}_{category}_simplified.csv"
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
                df=df, leaderboard_configs=configs, task_config=task_config
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
    df: pd.DataFrame | pd.Series,
    leaderboard_configs: dict[str, dict[str, list[str]]],
    task_config: dict[str, dict[str, str]],
) -> tuple[list[str], list[str]]:
    """Create the leaderboard headers.

    The first header includes the task types (with links), and the second header
    contains the 'original' header but with html links to the datasets.

    Args:
        df:
            The dataframe.
        leaderboard_configs:
            The leaderboard configurations.
        task_config:
            The task configuration.

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

    # Get the set of orthogonal tasks
    orthogonal_tasks = {
        task
        for task, task_config in task_config.items()
        if task_config.get("orthogonal", False)
    }

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
    ranks: dict[str, dict[str, dict[str, float]]],
    metadata_dict: dict[str, dict],
    categories: list[t.Literal["all", "nlu"]],
    task_config: dict[str, dict[str, str]],
    leaderboard_configs: dict[str, dict[str, list[str]]],
    include_dataset_columns: bool,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate DataFrames from the model results.

    Args:
        model_results:
            The model results.
        ranks:
            The ranks of the models.
        metadata_dict:
            The metadata.
        categories:
            The categories of leaderboards to generate.
        task_config:
            The task configuration.
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

    # Mapping from category to dataset names
    category_to_datasets = {
        category: [
            dataset
            for config in leaderboard_configs.values()
            for task, task_datasets in config.items()
            for dataset in task_datasets
            if task_config[task]["category"] == category or category == "all"
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
            if task_config[task].get("orthogonal", False)
            and (task_config[task]["category"] == category or category == "all")
        }
        for category in categories
    }

    dfs: list[tuple[pd.DataFrame, pd.DataFrame]] = list()
    for category in categories:
        data_dict: dict[str, list] = defaultdict(list)
        for model_id, results in model_results.items():
            # Get the overall rank for the model
            rank = ranks[model_id][category]["overall"]
            if math.isfinite(rank):
                rank = round(rank, 2)
            language_ranks = ranks[model_id][category]
            language_ranks.pop("overall")

            # Get the default values for the dataset columns
            default_dataset_values = {
                ds: float("nan") for ds in category_to_datasets[category]
            } | {f"{ds}_version": "-@@0" for ds in category_to_datasets[category]}
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
                    score_str = (
                        " / ".join(
                            f"{total_score:,.2f} ± {std_err:,.2f}"
                            for _, total_score, std_err in scores
                        )
                        + f"@@{main_score:.2f}"
                    )
                    if dataset in category_to_orthogonal_datasets[category]:
                        orthogonal_task = category_to_orthogonal_datasets[category][
                            dataset
                        ]
                        orthogonal_scores[orthogonal_task].append(main_score)
                else:
                    score_str = "-@@-1"
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
                dict(model=model_id, rank=rank)
                | default_orthogonal_values
                | default_dataset_values
                | orthogonal_task_scores
                | metadata
                | language_ranks
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

        # Create dataframe and sort by rank
        df = (
            pd.DataFrame(data_dict)
            .sort_values(
                by="rank",
                key=lambda series: series.map(
                    lambda x: float(x.split("@@")[1]) if isinstance(x, str) else x
                ),
            )
            .reset_index(drop=True)
        )

        # Ensure that inf values appear at the bottom
        rank_cols = ["rank"]
        if len(leaderboard_configs) > 1:
            rank_cols += list(leaderboard_configs.keys())

        # Convert rank to string, where {shown value}@@{sort value} to ensures that NaN
        # values appear at the bottom.
        for col in rank_cols:
            df[col] = [
                f"{value:.2f}@@{value:.2f}"
                if math.isfinite(value)
                else "-@@100"  # just a large number
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
            ["model", "generative_type", "rank"]
            + orthogonal_cols
            + [
                "parameters",
                "vocabulary_size",
                "context",
                "commercial",
                "merge",
                "open",
                "trained_from_scratch",
            ]
            + rank_cols[1:]
        )
        if include_dataset_columns:
            cols += dataset_cols
            cols += [f"{dataset}_version" for dataset in dataset_cols]
        df = df[cols]

        # If a model has only orthogonal values, we remove it from the leaderboard
        num_before = len(df)
        value_cols = [col for col in dataset_cols + rank_cols[1:] if col in df.columns]
        model_ids_with_dataset_values = df.query(
            " or ".join(
                [f"({col} != '-@@-1' and {col} != '-@@100')" for col in value_cols]
            )
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
        boolean_columns = ["commercial", "merge"]
        for col in boolean_columns:
            df[col] = df[col].apply(lambda x: "✓" if x else "✗")

        # Convert open values to symbols
        open_mapping = {"open-source": "✓", "open-weight": "(✓)", "closed-source": "✗"}
        df["open"] = df["open"].map(open_mapping)

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
                "model",
                "generative_type",
                "rank",
                "parameters",
                "vocabulary_size",
                "context",
                "commercial",
                "merge",
                "open",
                "trained_from_scratch",
            ]
        ]
        df_simplified = df_simplified.map(
            lambda x: x.split("@@")[0] if isinstance(x, str) else x
        )
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
            | {rank_col: rank_col.title() for rank_col in rank_cols[1:]}
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
