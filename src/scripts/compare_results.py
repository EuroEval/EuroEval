"""Compare EuroEval evaluation outputs from one or more JSONL files.

Produces comparison tables with models as rows and datasets as columns, showing
scores with confidence bounds and marking statistically significant best results.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import typing as t
from collections import defaultdict
from pathlib import Path

import click
import numpy as np

from euroeval.data_models import BenchmarkResult
from leaderboards.bootstrap_cis import bootstrap_rank_scores, bootstrap_test
from leaderboards.core_models import API_MODEL_PATTERNS
from leaderboards.result_processing import group_results_by_model, record_is_valid
from leaderboards.score_computation import compute_ranks_bootstrap
from leaderboards.task_metadata import official_datasets_for_language

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s ⋅ %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

# Constants for validation (matching generate_leaderboards.py)
MINIMUM_VERSION: str = "15.0.0"
BANNED_VERSIONS: list[str] = ["9.3.0", "10.0.0"]
BANNED_MODEL_PATTERNS: list[re.Pattern] = [
    re.compile("^meta-llama/Llama-3.1-405B-Instruct$"),
    re.compile("^utter-project/EuroVLM-9B-Preview$"),
]


def load_jsonl_files(file_paths: list[Path]) -> list[dict[str, t.Any]]:
    """Load and parse multiple JSONL files.

    Args:
        file_paths:
            List of paths to JSONL files to load.

    Returns:
        List of parsed record dictionaries.

    Raises:
        FileNotFoundError:
            If any file path does not exist.
        ValueError:
            If any line contains invalid JSON.
    """
    records: list[dict[str, t.Any]] = []

    for file_path in file_paths:
        if not file_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {file_path}")

        logger.info(f"Loading {file_path}...")
        with file_path.open() as f:
            for line_num, line in enumerate(f, start=1):
                if not line.strip():
                    continue

                # Handle multiple JSON objects on the same line
                for record_str in re.split(pattern=r"(?<=})(?={)", string=line):
                    if not record_str.strip():
                        continue
                    try:
                        parsed = json.loads(record_str)
                        # Use BenchmarkResult.from_dict for parsing (handles EEE format)
                        converted = BenchmarkResult.from_dict(parsed)
                        records.append(converted.to_eee_dict())
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Invalid JSON in {file_path} at line {line_num}: {e}"
                        ) from e
                    except Exception as e:
                        logger.warning(
                            f"Skipping invalid record in {file_path} "
                            f"at line {line_num}: {e}"
                        )

    logger.info(f"Loaded {len(records):,} records from {len(file_paths)} file(s).")
    return records


def filter_records(
    records: list[dict[str, t.Any]],
    languages: list[str] | None = None,
    datasets: list[str] | None = None,
    models: list[str] | None = None,
) -> list[dict[str, t.Any]]:
    """Filter records by language, dataset, and model patterns.

    Args:
        records:
            List of records to filter.
        languages (optional):
            List of language codes to include. If None, include all.
        datasets (optional):
            List of dataset names to include. If None, include all.
        models (optional):
            List of model name patterns (regex) to include. If None, include all.

    Returns:
        Filtered list of records.
    """
    filtered = []

    for record in records:
        # Validate the record
        if not record_is_valid(
            record=record,
            min_version=MINIMUM_VERSION,
            banned_versions=BANNED_VERSIONS,
            banned_model_patterns=BANNED_MODEL_PATTERNS,
            api_model_patterns=API_MODEL_PATTERNS,
        ):
            continue

        # Filter by language
        if languages:
            record_languages = record.get("languages", [])
            if not any(lang in languages for lang in record_languages):
                continue

        # Filter by dataset
        if datasets:
            if record.get("dataset") not in datasets:
                continue

        # Filter by model pattern
        if models:
            model_name = record.get("model", "")
            if not any(re.search(pattern, model_name) for pattern in models):
                continue

        filtered.append(record)

    logger.info(f"Filtered to {len(filtered):,} records.")
    return filtered


def build_configs_for_languages(
    language_codes: list[str],
) -> dict[str, dict[str, list[str]]]:
    """Build task -> dataset configs for the specified languages.

    Args:
        language_codes:
            List of language codes (e.g., ["da", "sv"]).

    Returns:
        Nested dict: language_code -> task -> [dataset_names].
    """
    configs: dict[str, dict[str, list[str]]] = {}

    for lang_code in language_codes:
        try:
            # Get official datasets for this language
            datasets_by_task = official_datasets_for_language(lang_code)
            if datasets_by_task:
                # Convert OrderedDict to regular dict with lists
                configs[lang_code] = {
                    task: list(datasets) for task, datasets in datasets_by_task.items()
                }
        except Exception as e:
            logger.warning(f"Could not load config for language {lang_code}: {e}")

    return configs


def compute_dataset_scores(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
    n_bootstraps: int = 1000,
    seed: int | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute bootstrap confidence intervals for each dataset score.

    Args:
        model_results:
            Results grouped by model and dataset.
        n_bootstraps (optional):
            Number of bootstrap replicates. Defaults to 1000.
        seed (optional):
            Random seed for reproducibility.

    Returns:
        Nested dict: model_id -> dataset -> {"score", "ci_lower", "ci_upper"}.
    """
    # For dataset-level CIs, we need to compute bootstrap distributions
    # This is a simplified version - in practice we'd want a full bootstrap
    # For now, use the existing scores with analytical SE-based CIs
    result: dict[str, dict[str, dict[str, float]]] = {}

    for model_id, datasets in model_results.items():
        result[model_id] = {}
        for dataset, score_tuples in datasets.items():
            if not score_tuples:
                continue

            # Use the first (primary) metric
            raw_scores, total_score, se = score_tuples[0]

            # Use 95% CI based on standard error (approximate)
            # For proper bootstrap, we'd resample raw_scores
            rng = np.random.default_rng(seed)
            bootstrap_means = []
            for _ in range(n_bootstraps):
                resampled = rng.choice(raw_scores, size=len(raw_scores), replace=True)
                bootstrap_means.append(float(np.mean(resampled)))

            ci_lower = float(np.percentile(bootstrap_means, 2.5))
            ci_upper = float(np.percentile(bootstrap_means, 97.5))
            median_score = float(np.median(bootstrap_means))

            result[model_id][dataset] = {
                "score": median_score,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }

    return result


def format_score(score: float, ci_lower: float, ci_upper: float) -> str:
    """Format a score with confidence bounds.

    Args:
        score:
            Point estimate (median).
        ci_lower:
            Lower bound of confidence interval.
        ci_upper:
            Upper bound of confidence interval.

    Returns:
        Formatted string like "75.32 ± 2.14".
    """
    margin = (ci_upper - ci_lower) / 2
    return f"{score:.2f} ± {margin:.2f}"


def determine_best_results(
    model_datasets: dict[str, dict[str, dict[str, float]]],
    bootstrap_scores: dict[str, dict[str, dict[str, np.ndarray]]],
    category: str = "generative",
    language: str = "overall",
    alpha: float = 0.05,
) -> tuple[dict[str, dict[str, bool]], dict[str, dict[str, bool]]]:
    """Determine which results are best or not significantly different from best.

    Args:
        model_datasets:
            Model -> dataset -> score info dicts.
        bootstrap_scores:
            Bootstrap distributions from bootstrap_rank_scores.
        category (optional):
            Category to use for statistical tests. Defaults to "generative".
        language (optional):
            Language to use for statistical tests. Defaults to "overall".
        alpha (optional):
            Significance level. Defaults to 0.05.

    Returns:
        Tuple of (is_best, is_not_significantly_different) dicts.
    """
    is_best: dict[str, dict[str, bool]] = defaultdict(dict)
    is_not_sig_diff: dict[str, dict[str, bool]] = defaultdict(dict)

    # For each dataset, find the best score
    datasets = set()
    for model_data in model_datasets.values():
        datasets.update(model_data.keys())

    for dataset in datasets:
        # Get scores for all models on this dataset
        model_scores: list[tuple[str, float]] = []
        for model_id, model_data in model_datasets.items():
            if dataset in model_data:
                score = model_data[dataset]["score"]
                model_scores.append((model_id, score))

        if not model_scores:
            continue

        # Sort by score descending (higher is better)
        model_scores.sort(key=lambda x: x[1], reverse=True)
        best_score = model_scores[0][1]
        best_model = model_scores[0][0]

        # Mark the best model
        is_best[best_model][dataset] = True

        # Check if other models are not significantly different from best
        for model_id, score in model_scores[1:]:
            if score == best_score:
                is_best[model_id][dataset] = True
                continue

            # Use bootstrap test if available
            if model_id in bootstrap_scores and best_model in bootstrap_scores:
                try:
                    p_value = bootstrap_test(
                        bootstrap_scores=bootstrap_scores,
                        model_a=model_id,
                        model_b=best_model,
                        category=category,
                        language=language,
                        alternative="greater",  # Is model_a worse (higher score)?
                    )
                    # If p > alpha, not significantly different
                    if p_value > alpha:
                        is_not_sig_diff[model_id][dataset] = True
                except (KeyError, ValueError):
                    pass

    return dict(is_best), dict(is_not_sig_diff)


def format_markdown_table(
    model_datasets: dict[str, dict[str, dict[str, float]]],
    is_best: dict[str, dict[str, bool]],
    is_not_sig_diff: dict[str, dict[str, bool]],
    rank_scores: dict[str, dict[str, dict[str, dict[str, float]]]] | None = None,
) -> str:
    """Format results as a Markdown table.

    Args:
        model_datasets:
            Model -> dataset -> score info dicts.
        is_best:
            Model -> dataset -> is_best flags.
        is_not_sig_diff:
            Model -> dataset -> is_not_significantly_different flags.
        rank_scores (optional):
            Overall rank scores to append.

    Returns:
        Markdown-formatted table string.
    """
    # Collect all datasets
    all_datasets = set()
    for model_data in model_datasets.values():
        all_datasets.update(model_data.keys())

    # Sort datasets alphabetically
    sorted_datasets = sorted(all_datasets)

    # Sort models by overall rank score if available, else alphabetically
    models = list(model_datasets.keys())
    if rank_scores:

        def get_rank_score(model: str) -> float:
            try:
                return (
                    rank_scores.get(model, {})
                    .get("generative", {})
                    .get("overall", {})
                    .get("score", float("inf"))
                )
            except (KeyError, TypeError):
                return float("inf")

        models.sort(key=get_rank_score)
    else:
        models.sort()

    # Build table header
    lines = []
    header = "| Model | " + " | ".join(sorted_datasets)
    if rank_scores:
        header += " | Overall Rank |"
    lines.append(header)

    # Build separator
    separator = "|-" + "|-" * (len(sorted_datasets) + (1 if rank_scores else 0))
    lines.append(separator)

    # Build rows
    for model in models:
        model_data = model_datasets.get(model, {})
        row_cells = []

        # Model name
        model_display = model
        if len(model_display) > 40:
            model_display = model_display[:37] + "..."
        row_cells.append(model_display)

        # Dataset scores
        for dataset in sorted_datasets:
            if dataset in model_data:
                info = model_data[dataset]
                score_str = format_score(
                    info["score"], info["ci_lower"], info["ci_upper"]
                )

                # Add markdown formatting for best/not-sig-diff
                if is_best.get(model, {}).get(dataset, False):
                    score_str = f"**{score_str}**"
                elif is_not_sig_diff.get(model, {}).get(dataset, False):
                    score_str = f"**{score_str}**"

                row_cells.append(score_str)
            else:
                row_cells.append("-")

        # Overall rank score
        if rank_scores:
            rank_info = (
                rank_scores.get(model, {}).get("generative", {}).get("overall", {})
            )
            if rank_info:
                rank_str = format_score(
                    rank_info["score"], rank_info["ci_lower"], rank_info["ci_upper"]
                )
                row_cells.append(rank_str)
            else:
                row_cells.append("-")

        lines.append("| " + " | ".join(row_cells) + " |")

    return "\n".join(lines)


def format_csv_table(
    model_datasets: dict[str, dict[str, dict[str, float]]],
    rank_scores: dict[str, dict[str, dict[str, dict[str, float]]]] | None = None,
) -> str:
    """Format results as CSV.

    Args:
        model_datasets:
            Model -> dataset -> score info dicts.
        rank_scores (optional):
            Overall rank scores to append.

    Returns:
        CSV-formatted string.
    """
    # Collect all datasets
    all_datasets = set()
    for model_data in model_datasets.values():
        all_datasets.update(model_data.keys())

    sorted_datasets = sorted(all_datasets)
    models = sorted(model_datasets.keys())

    lines = []

    # Header
    header = "Model," + ",".join(sorted_datasets)
    if rank_scores:
        header += ",Overall Score,Overall CI Lower,Overall CI Upper"
    lines.append(header)

    # Rows
    for model in models:
        model_data = model_datasets.get(model, {})
        row_cells = [model]

        for dataset in sorted_datasets:
            if dataset in model_data:
                info = model_data[dataset]
                row_cells.append(f"{info['score']:.2f}")
            else:
                row_cells.append("")

        if rank_scores:
            rank_info = (
                rank_scores.get(model, {}).get("generative", {}).get("overall", {})
            )
            if rank_info:
                row_cells.extend(
                    [
                        f"{rank_info['score']:.2f}",
                        f"{rank_info['ci_lower']:.2f}",
                        f"{rank_info['ci_upper']:.2f}",
                    ]
                )
            else:
                row_cells.extend(["", "", ""])

        lines.append(",".join(row_cells))

    return "\n".join(lines)


def format_html_table(
    model_datasets: dict[str, dict[str, dict[str, float]]],
    is_best: dict[str, dict[str, bool]],
    is_not_sig_diff: dict[str, dict[str, bool]],
    rank_scores: dict[str, dict[str, dict[str, dict[str, float]]]] | None = None,
) -> str:
    """Format results as an HTML table.

    Args:
        model_datasets:
            Model -> dataset -> score info dicts.
        is_best:
            Model -> dataset -> is_best flags.
        is_not_sig_diff:
            Model -> dataset -> is_not_significantly_different flags.
        rank_scores (optional):
            Overall rank scores to append.

    Returns:
        HTML-formatted table string.
    """
    # Collect all datasets
    all_datasets = set()
    for model_data in model_datasets.values():
        all_datasets.update(model_data.keys())

    sorted_datasets = sorted(all_datasets)
    models = sorted(model_datasets.keys())

    lines = [
        '<table border="1" class="dataframe">',
        "  <thead>",
        "    <tr>",
        "      <th>Model</th>",
    ]

    for dataset in sorted_datasets:
        lines.append(f"      <th>{dataset}</th>")

    if rank_scores:
        lines.append("      <th>Overall Rank</th>")

    lines.extend(["    </tr>", "  </thead>", "  <tbody>"])

    for model in models:
        model_data = model_datasets.get(model, {})
        row_parts = ["    <tr>", f"      <td>{model}</td>"]

        for dataset in sorted_datasets:
            if dataset in model_data:
                info = model_data[dataset]
                score_str = format_score(
                    info["score"], info["ci_lower"], info["ci_upper"]
                )

                # Add styling for best/not-sig-diff
                if is_best.get(model, {}).get(dataset, False) or is_not_sig_diff.get(
                    model, {}
                ).get(dataset, False):
                    row_parts.append(
                        f'      <td style="font-weight: bold;">{score_str}</td>'
                    )
                else:
                    row_parts.append(f"      <td>{score_str}</td>")
            else:
                row_parts.append("      <td>-</td>")

        if rank_scores:
            rank_info = (
                rank_scores.get(model, {}).get("generative", {}).get("overall", {})
            )
            if rank_info:
                rank_str = format_score(
                    rank_info["score"], rank_info["ci_lower"], rank_info["ci_upper"]
                )
                row_parts.append(f"      <td>{rank_str}</td>")
            else:
                row_parts.append("      <td>-</td>")

        row_parts.append("    </tr>")
        lines.append("\n".join(row_parts))

    lines.extend(["  </tbody>", "</table>"])

    return "\n".join(lines)


@click.command()
@click.argument(
    "jsonl_files", nargs=-1, type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["markdown", "csv", "html"]),
    default="markdown",
    help="Output format. Defaults to markdown.",
)
@click.option(
    "--language",
    "-l",
    multiple=True,
    help="Filter by language code (e.g., da, sv). Can be specified multiple times.",
)
@click.option(
    "--dataset",
    "-d",
    multiple=True,
    help="Filter by dataset name. Can be specified multiple times.",
)
@click.option(
    "--model",
    "-m",
    multiple=True,
    help="Filter by model name pattern (regex). Can be specified multiple times.",
)
@click.option(
    "--n-bootstraps",
    type=int,
    default=1000,
    show_default=True,
    help="Number of bootstrap replicates for confidence intervals.",
)
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility.")
@click.option(
    "--category",
    "-c",
    type=click.Choice(["generative", "all_models"]),
    default="generative",
    help="Category for rank score computation.",
)
def main(
    jsonl_files: tuple[Path, ...],
    output: str,
    language: tuple[str, ...],
    dataset: tuple[str, ...],
    model: tuple[str, ...],
    n_bootstraps: int,
    seed: int | None,
    category: str,
) -> None:
    r"""Compare EuroEval evaluation outputs from JSONL files.

    Produces a comparison table with models as rows and datasets as columns.
    Each cell shows the score with confidence bounds (e.g., "75.32 ± 2.14").
    Best results are marked in bold, including results not significantly
    different from best (using bootstrap tests).

    \b
    Examples:
        uv run src/scripts/compare_results.py results.jsonl
        uv run src/scripts/compare_results.py file1.jsonl file2.jsonl -o csv
        uv run src/scripts/compare_results.py results.jsonl -l da -l sv
        uv run src/scripts/compare_results.py results.jsonl -d "da|dd" -m "Llama"
    """
    # Load JSONL files
    records = load_jsonl_files(list(jsonl_files))

    # Filter records
    filtered_records = filter_records(
        records=records,
        languages=list(language) if language else None,
        datasets=list(dataset) if dataset else None,
        models=list(model) if model else None,
    )

    if not filtered_records:
        logger.error("No valid records after filtering.")
        sys.exit(1)

    # Group results by model and dataset
    model_results = group_results_by_model(results=filtered_records)

    # Compute bootstrap confidence intervals for dataset scores
    logger.info("Computing dataset-level confidence intervals...")
    dataset_scores = compute_dataset_scores(
        model_results=model_results, n_bootstraps=n_bootstraps, seed=seed
    )

    # Compute overall rank scores using bootstrap methods
    logger.info("Computing overall rank scores...")

    # Build configs for the languages present in the data
    all_languages: set[str] = set()
    for record in filtered_records:
        all_languages.update(record.get("languages", []))

    configs = build_configs_for_languages(list(all_languages))

    if not configs:
        logger.warning(
            "Could not build task configs. Using default structure for rank scores."
        )
        # Create a minimal config with all datasets under a single task
        all_datasets = set()
        for model_data in model_results.values():
            all_datasets.update(model_data.keys())

        configs = {"overall": {"all_tasks": list(all_datasets)}}

    try:
        rank_scores = compute_ranks_bootstrap(
            model_results=model_results,
            configs=configs,
            n_bootstraps=n_bootstraps,
            seed=seed,
        )
    except Exception as e:
        logger.warning(
            f"Rank score computation failed: {e}. Using dataset scores only."
        )
        rank_scores = None

    # Compute bootstrap scores for statistical tests
    bootstrap_scores_dict: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    if rank_scores:
        try:
            bootstrap_scores_dict = bootstrap_rank_scores(
                model_results=model_results,
                configs=configs,
                n_bootstraps=n_bootstraps,
                seed=seed,
                categories=(category,),
            )
        except Exception as e:
            logger.warning(f"Bootstrap scores computation failed: {e}")

    # Determine best results
    logger.info("Determining statistically significant best results...")
    is_best, is_not_sig_diff = determine_best_results(
        model_datasets=dataset_scores,
        bootstrap_scores=bootstrap_scores_dict,
        category=category,
        language="overall",
    )

    # Format output
    logger.info(f"Formatting output as {output}...")
    if output == "markdown":
        result = format_markdown_table(
            model_datasets=dataset_scores,
            is_best=is_best,
            is_not_sig_diff=is_not_sig_diff,
            rank_scores=rank_scores,
        )
    elif output == "csv":
        result = format_csv_table(
            model_datasets=dataset_scores, rank_scores=rank_scores
        )
    elif output == "html":
        result = format_html_table(
            model_datasets=dataset_scores,
            is_best=is_best,
            is_not_sig_diff=is_not_sig_diff,
            rank_scores=rank_scores,
        )

    # Print to stdout
    print(result)


if __name__ == "__main__":
    main()
