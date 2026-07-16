"""Create a language spider/radial plot comparing models across languages.

This script loads evaluation results from local JSONL files and generates a
Plotly polar chart comparing selected models across languages.
Only rank score is plotted (lower is better, axis is reversed).
Output is a PNG file.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import typing as t
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import plotly.graph_objects as go
from kaleido.errors import KaleidoError

from euroeval.constants import ORTHOGONAL_TASKS
from euroeval.jsonl_io import parse_jsonl_lines
from euroeval.languages import get_all_languages
from leaderboards.constants import RESULTS_DIR
from leaderboards.record_fields import get_few_shot, get_task, get_total_scores
from leaderboards.records import get_dataset
from leaderboards.task_metadata import (
    language_name_to_codes,
    languages_with_official_datasets,
    official_datasets_for_language,
    task_metric_names,
)

# Type alias for evaluation records (recursive JSON structure)
Json = dict[str, "Json"] | list["Json"] | str | int | float | bool | None
JsonDict = dict[str, Json]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(
    models: tuple[str, ...],
    languages: tuple[str, ...],
    shots: str,
    max_score: float | None,
) -> int:
    """Create a language spider plot comparing models across languages.

    Loads evaluation results from local JSONL files and generates a Plotly
    polar chart comparing selected models across languages.
    Only rank score is plotted (lower is better).
    Output is a PNG file.

    Args:
        models:
            Model IDs to include.
        languages:
            Language names or codes to include. Empty means official languages.
        shots:
            Shot setting: "auto", "zero", or "few".
        max_score (optional):
            Override maximum rank score for the radial axis. If omitted,
            auto-computed from plotted rank scores (rounded up to nearest 0.5,
            minimum 2.5; rank score of 1 is perfect).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    model_list = list(models)
    language_list = list(languages) if languages else None

    try:
        resolved_languages = _resolve_languages(language_list)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if not language_list:
        logger.info(f"Using {len(resolved_languages)} official languages by default")

    logger.info(f"Loading results for {len(model_list)} model(s)")
    records = _load_results_for_models(model_list)

    if not records:
        click.echo("Error: No results found for specified models", err=True)
        sys.exit(1)

    logger.info(f"Loaded {len(records)} total records")

    try:
        filtered_records = _filter_by_shots(
            records, t.cast(t.Literal["auto", "zero", "few"], shots)
        )
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if not filtered_records:
        click.echo("Error: No records after shot filtering", err=True)
        sys.exit(1)

    logger.info(f"Using {len(filtered_records)} records after shot filtering")

    shot_value: bool | None = None
    if shots == "zero":
        shot_value = False
    elif shots == "few":
        shot_value = True

    model_scores_matrix = _build_score_matrix(
        records=filtered_records,
        models=model_list,
        languages=resolved_languages,
        shot_value=shot_value,
    )

    model_scores_matrix, used_languages = _compute_language_intersection(
        model_scores_matrix, resolved_languages
    )

    if not used_languages:
        missing_info = []
        for model_id, lang_scores in model_scores_matrix.items():
            missing = [lang for lang, score in lang_scores.items() if score is None]
            if missing:
                missing_info.append(f"{model_id}: {', '.join(missing[:5])}")
        click.echo(
            "Error: No languages have scores for all selected models. "
            "Missing scores:\n  " + "\n  ".join(missing_info),
            err=True,
        )
        sys.exit(1)

    if used_languages != resolved_languages:
        logger.info(
            f"Using {len(used_languages)} languages (intersection of all models): "
            f"{', '.join(used_languages)}"
        )

    try:
        max_score_val = _compute_max_score(model_scores_matrix, max_score)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    logger.info(f"Max score: {max_score_val:.2f}")

    fig = _create_spider_plot(
        model_scores=model_scores_matrix,
        languages=used_languages,
        max_score=max_score_val,
    )

    output_path = Path("language-spider-plot.png")
    try:
        fig.write_image(str(output_path))
    except KaleidoError as exc:
        click.echo(f"Error writing PNG: {exc}", err=True)
        return 1
    logger.info(f"Wrote spider plot to {output_path}")

    return 0


def _compute_language_intersection(
    model_scores: dict[str, dict[str, float | None]], languages: list[str]
) -> tuple[dict[str, dict[str, float | None]], list[str]]:
    """Compute the intersection of languages with scores for all models.

    Filters the score matrix to only include languages where all models
    have at least one valid score.

    Args:
        model_scores:
            Model x language score matrix.
        languages:
            Original language list.

    Returns:
        Tuple of (filtered score matrix, list of languages in intersection).
    """
    if not model_scores:
        return model_scores, languages

    languages_with_all_scores: set[str] = set(languages)

    for model_id, lang_scores in model_scores.items():
        models_valid_languages = {
            lang for lang, score in lang_scores.items() if score is not None
        }
        languages_with_all_scores &= models_valid_languages

    filtered_matrix: dict[str, dict[str, float | None]] = {}
    for model_id, lang_scores in model_scores.items():
        filtered_matrix[model_id] = {
            lang: lang_scores[lang] for lang in languages_with_all_scores
        }

    return filtered_matrix, sorted(languages_with_all_scores)


def _normalise_language_input(language_input: str) -> set[str]:
    """Normalise a language input (name or code) to a set of language codes.

    Handles both language names (e.g., "danish") and codes (e.g., "da").
    Uses the leaderboard language name → code mapping for names.

    Args:
        language_input:
            Language name or code.

    Returns:
        Set of language codes (may contain multiple codes for names like
        "norwegian" which map to both "nb" and "nn").

    Raises:
        ValueError:
            If the language cannot be resolved to any valid code.
    """
    lang_input = language_input.strip().lower()

    codes_from_name = language_name_to_codes(lang_input)
    if codes_from_name:
        return codes_from_name

    all_languages = get_all_languages()
    if lang_input in all_languages:
        return {lang_input}

    for code in all_languages:
        if code.lower() == lang_input:
            return {code}

    raise ValueError(
        f"Cannot resolve language {language_input!r} to any valid language code. "
        f"Use a language name (e.g., 'danish') or code (e.g., 'da')."
    )


def _resolve_languages(language_inputs: list[str] | None) -> list[str]:
    """Resolve language inputs to a list of language codes.

    Args:
        language_inputs (optional):
            List of language names or codes. If None, uses official languages.

    Returns:
        List of language codes.
    """
    if language_inputs:
        all_codes: set[str] = set()
        for lang_input in language_inputs:
            codes = _normalise_language_input(lang_input)
            all_codes.update(codes)
        return sorted(all_codes)
    else:
        lang_names = languages_with_official_datasets()
        all_codes: set[str] = set()
        for name in lang_names:
            codes = language_name_to_codes(name)
            all_codes.update(codes)
        return sorted(all_codes)


def _get_primary_metric_for_task(task: str) -> str:
    """Get the primary metric for a task.

    Args:
        task:
            Task name.

    Returns:
        Primary metric name.
    """
    primary, _ = task_metric_names(task)
    return primary


def _extract_languages_from_record(record: JsonDict) -> list[str]:
    """Extract language codes from an EEE-format record.

    Handles both modern EEE format (JSON-encoded string in eval_library) and
    legacy formats.

    Args:
        record:
            EEE-format result record.

    Returns:
        List of language codes.
    """
    eval_lib = record.get("eval_library", {})
    if not isinstance(eval_lib, dict):
        eval_lib = {}
    additional = eval_lib.get("additional_details", {})
    if not isinstance(additional, dict):
        additional = {}
    languages_json = additional.get("languages")

    if languages_json:
        try:
            languages = json.loads(languages_json)
            if isinstance(languages, list):
                return [str(lang) for lang in languages]
        except (json.JSONDecodeError, TypeError):
            pass

    legacy_languages_value = record.get("languages")
    if isinstance(legacy_languages_value, str):
        try:
            languages = json.loads(legacy_languages_value)
            if isinstance(languages, list):
                return [str(lang) for lang in languages]
        except (json.JSONDecodeError, TypeError):
            return [legacy_languages_value]
    elif isinstance(legacy_languages_value, list):
        return [str(lang) for lang in legacy_languages_value]

    return []


def _extract_task_from_record(record: JsonDict) -> str | None:
    """Extract task name from an EEE-format record.

    Args:
        record:
            EEE-format result record.

    Returns:
        Task name, or None if not found.
    """
    eval_lib = record.get("eval_library", {})
    if not isinstance(eval_lib, dict):
        return None
    additional = eval_lib.get("additional_details", {})
    if not isinstance(additional, dict):
        return None
    return additional.get("task")


def _extract_scores_from_record(record: JsonDict) -> dict[str, float]:
    """Extract all scores from an EEE-format record.

    Uses the robust get_total_scores helper which handles the
    evaluation_results structure.

    Args:
        record:
            EEE-format result record.

    Returns:
        Dict mapping metric names to scores. Empty dict if no scores found.
    """
    scores = get_total_scores(record)
    return scores if scores is not None else {}


def _get_model_identifier(record: JsonDict) -> str:
    """Extract model identifier from an EEE-format record.

    Args:
        record:
            EEE-format result record.

    Returns:
        Model name/ID.
    """
    model_info = record.get("model_info", {})
    if not isinstance(model_info, dict):
        return ""
    return model_info.get("name", "") or model_info.get("id", "")


def _load_results_for_models(model_ids: list[str]) -> list[JsonDict]:
    """Load all results for the specified model IDs from local JSONL files.

    Args:
        model_ids:
            List of model IDs to load results for.

    Returns:
        List of EEE-format result records.
    """
    records: list[JsonDict] = []
    model_files = sorted(RESULTS_DIR.glob("*.jsonl"))

    if not model_files:
        logger.error("No JSONL files found in %s", RESULTS_DIR)
        return records

    for jsonl_path in model_files:
        model_id_from_file = jsonl_path.stem.replace("_", "/", 1)
        if not any(
            model_id_from_file == m or model_id_from_file.endswith("/" + m)
            for m in model_ids
        ):
            continue

        try:
            lines = jsonl_path.read_text(encoding="utf-8").splitlines()
            file_records = parse_jsonl_lines(
                lines=lines, source=str(jsonl_path), strict=False
            )
            for rec in file_records:
                if isinstance(rec, dict):
                    records.append(t.cast(JsonDict, rec))
            logger.info(f"Loaded {len(file_records)} records from {jsonl_path.name}")
        except Exception as exc:
            logger.warning(f"Failed to parse {jsonl_path}: {exc}")

    return records


def _filter_by_shots(
    records: list[JsonDict], shots_setting: t.Literal["auto", "zero", "few"]
) -> list[JsonDict]:
    """Filter records by shot setting.

    Args:
        records:
            All result records.
        shots_setting:
            Shot setting: "auto", "zero", or "few".

    Returns:
        Filtered records.

    Raises:
        ValueError:
            If auto-detection is ambiguous or fails.
    """
    if shots_setting == "zero":
        return [r for r in records if get_few_shot(r) is False]
    elif shots_setting == "few":
        return [r for r in records if get_few_shot(r) is True]

    zero_records = [r for r in records if get_few_shot(r) is False]
    few_records = [r for r in records if get_few_shot(r) is True]

    zero_count = len(zero_records)
    few_count = len(few_records)

    if zero_count > 0 and few_count == 0:
        logger.info("Auto-detected: using zero-shot (few-shot records not found)")
        return zero_records
    elif few_count > 0 and zero_count == 0:
        logger.info("Auto-detected: using few-shot (zero-shot records not found)")
        return few_records
    elif zero_count > 0 and few_count > 0:
        raise ValueError(
            f"Auto-detection ambiguous: found {zero_count} zero-shot and "
            f"{few_count} few-shot records. Please specify --shots zero or --shots few."
        )
    else:
        raise ValueError(
            "Auto-detection failed: no records with known shot setting. "
            "Records may be missing few_shot metadata."
        )


def _build_score_matrix(
    records: list[JsonDict],
    models: list[str],
    languages: list[str],
    shot_value: bool | None,
) -> dict[str, dict[str, float | None]]:
    """Build a model x language mean rank score matrix from records.

    Computes per-language mean rank scores following the EuroEval methodology:
    1. Extract aggregated scores per (model, dataset, language)
    2. Compute dataset rank scores: 1 + (best_mean - model_mean) / pooled_std
    3. Aggregate hierarchy: dataset -> task -> language (unweighted means)

    Rank score of 1 is perfect (best on dataset). Lower is better.

    Args:
        records:
            Result records.
        models:
            Model IDs.
        languages:
            Language codes.
        shot_value:
            None for any, True for few-shot, False for zero-shot.

    Returns:
        Model x language mean rank score matrix (lower is better).
    """
    # Step 1: Collect aggregated scores per (model, dataset, language)
    # Structure: model -> dataset -> language -> list of scores
    model_dataset_lang_scores: dict[str, dict[str, dict[str, list[float]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    for record in records:
        model_name = _get_model_identifier(record)
        matching_model: str | None = None
        for model in models:
            if model_name == model or model_name.endswith("/" + model):
                matching_model = model
                break

        if matching_model is None:
            continue

        record_few_shot = get_few_shot(record)
        if shot_value is not None and record_few_shot is not None:
            if record_few_shot != shot_value:
                continue

        task_name = get_task(record)
        if not task_name or task_name in ORTHOGONAL_TASKS:
            continue

        record_languages = _extract_languages_from_record(record)
        scores = _extract_scores_from_record(record)
        dataset = get_dataset(record)

        if not dataset or not record_languages:
            continue

        # Get primary metric for this task
        primary_metric, secondary_metric = task_metric_names(task_name)
        metrics_to_try = [primary_metric]
        if secondary_metric:
            metrics_to_try.append(secondary_metric)

        score: float | None = None
        for metric in metrics_to_try:
            if metric is None:
                continue
            test_metric = f"test_{metric}"
            if test_metric in scores:
                score = scores[test_metric]
                break
            if metric in scores:
                score = scores[metric]
                break

        # Fallbacks for legacy records
        if score is None:
            for fallback in ["test_macro_f1", "test_accuracy", "test_rouge"]:
                if fallback in scores:
                    score = scores[fallback]
                    break
        if score is None:
            for fallback in ["macro_f1", "accuracy", "rouge"]:
                if fallback in scores:
                    score = scores[fallback]
                    break

        if score is None or not math.isfinite(score):
            continue

        for lang in record_languages:
            if lang in languages:
                model_dataset_lang_scores[matching_model][dataset][lang].append(score)

    if not model_dataset_lang_scores:
        return {model: {lang: None for lang in languages} for model in models}

    # Step 2: Compute mean score per (model, dataset, language)
    model_dataset_lang_means: dict[str, dict[str, dict[str, float]]] = {}
    for model, datasets in model_dataset_lang_scores.items():
        model_dataset_lang_means[model] = {}
        for dataset, lang_scores in datasets.items():
            model_dataset_lang_means[model][dataset] = {}
            for lang, scores_list in lang_scores.items():
                if scores_list:
                    model_dataset_lang_means[model][dataset][lang] = sum(
                        scores_list
                    ) / len(scores_list)

    # Step 3: Compute dataset-level statistics (best_mean, pooled_std)
    # Need all models' scores on each dataset to compute these
    dataset_all_scores: dict[str, list[float]] = defaultdict(list)
    dataset_model_means: dict[str, dict[str, float]] = defaultdict(dict)

    for model, dataset_lang_means in model_dataset_lang_means.items():
        for dataset, lang_means in dataset_lang_means.items():
            # Mean across languages for this model-dataset
            if lang_means:
                dataset_mean = sum(lang_means.values()) / len(lang_means)
                dataset_model_means[dataset][model] = dataset_mean
                # Collect all raw scores for pooled std
                for lang_scores in model_dataset_lang_scores[model][dataset].values():
                    dataset_all_scores[dataset].extend(lang_scores)

    dataset_stats: dict[str, tuple[float, float]] = {}
    for dataset, all_scores in dataset_all_scores.items():
        if not all_scores:
            continue
        model_means = dataset_model_means[dataset]
        if not model_means:
            continue
        best_mean = max(model_means.values())
        pooled_std = float(np.std(all_scores)) if len(all_scores) > 1 else 1.0
        if pooled_std <= 0:
            pooled_std = 1.0
        dataset_stats[dataset] = (best_mean, pooled_std)

    # Step 4: Compute rank score per (model, dataset, language)
    # rank_score = 1 + (best_mean - model_mean) / pooled_std
    model_dataset_lang_rank_scores: dict[str, dict[str, dict[str, float]]] = {}
    for model in models:
        model_dataset_lang_rank_scores[model] = defaultdict(dict)
        if model not in model_dataset_lang_means:
            continue
        for dataset, lang_means in model_dataset_lang_means[model].items():
            if dataset not in dataset_stats:
                continue
            best_mean, pooled_std = dataset_stats[dataset]
            for lang, model_mean in lang_means.items():
                rank_score = 1.0 + (best_mean - model_mean) / pooled_std
                model_dataset_lang_rank_scores[model][dataset][lang] = rank_score

    # Step 5: Aggregate rank scores: dataset -> task -> language
    # Use official dataset configs to map datasets to tasks and languages
    dataset_to_task: dict[str, str] = {}
    dataset_to_lang_name: dict[str, str] = {}

    # Official languages have configs; iterate all to build mappings
    for lang_name in languages_with_official_datasets():
        try:
            lang_configs = official_datasets_for_language(lang_name)
            for task_name, datasets_list in lang_configs.items():
                for ds in datasets_list:
                    if ds not in dataset_to_task:  # First match wins
                        dataset_to_task[ds] = task_name
                        dataset_to_lang_name[ds] = lang_name
        except Exception:
            continue

    # Group rank scores by (language, task)
    model_lang_task_scores: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for model in models:
        model_rank_scores = model_dataset_lang_rank_scores.get(model, {})
        for dataset, lang_rank_scores in model_rank_scores.items():
            task = dataset_to_task.get(dataset)
            lang_name = dataset_to_lang_name.get(dataset)
            if not task or task in ORTHOGONAL_TASKS or not lang_name:
                continue
            for lang_code, rank_score in lang_rank_scores.items():
                if lang_code in languages:
                    model_lang_task_scores[model][lang_code][task].append(rank_score)

    # Step 6: Aggregate to language mean rank scores (unweighted mean across tasks)
    matrix: dict[str, dict[str, float | None]] = {}
    for model in models:
        matrix[model] = {}
        lang_task_scores = model_lang_task_scores.get(model, {})

        for lang in languages:
            task_scores = lang_task_scores.get(lang, {})
            if not task_scores:
                matrix[model][lang] = None
                continue
            # Mean of each task's dataset scores, then mean across tasks
            task_means = [np.mean(scores) for scores in task_scores.values() if scores]
            if task_means:
                matrix[model][lang] = float(np.mean(task_means))
            else:
                matrix[model][lang] = None

    return matrix


def _compute_max_score(
    model_scores: dict[str, dict[str, float | None]], max_score_override: float | None
) -> float:
    """Compute or validate maximum rank score for radial axis.

    If max_score_override is None, automatically computes the maximum from
    all plotted rank scores and rounds up to the nearest 0.5, with a minimum
    of 2.5 (since rank score of 1 is perfect, typical values are 2.5–6).
    If provided, validates that it is finite, > 1, and >= all plotted scores.

    Args:
        model_scores:
            Nested dict of model -> language -> rank score (or None).
        max_score_override (optional):
            User-provided maximum rank score override. If omitted, auto-computed
            from the plotted rank scores.

    Returns:
        Maximum rank score value for the radial axis.

    Raises:
        ValueError:
            If override is invalid (NaN, inf, <= 1, or too small).
    """
    all_scores: list[float] = []
    for lang_scores in model_scores.values():
        for score in lang_scores.values():
            if score is not None and math.isfinite(score) and score > 0:
                all_scores.append(score)

    if not all_scores:
        return 2.5

    max_found = max(all_scores)

    if max_score_override is not None:
        if not math.isfinite(max_score_override):
            raise ValueError(
                f"max-score {max_score_override} is invalid (must be finite)."
            )
        if max_score_override <= 1:
            raise ValueError(
                f"max-score {max_score_override} is invalid (must be > 1, "
                f"since rank score of 1 is perfect)."
            )
        if max_score_override < max_found:
            raise ValueError(
                f"max-score {max_score_override} is too small; "
                f"found rank scores up to {max_found:.2f}"
            )
        return max_score_override

    rounded = math.ceil(max_found * 2) / 2
    return max(rounded, 2.5)


def _normalise_model_name(model_id: str) -> str:
    """Normalise model ID for display.

    Args:
        model_id:
            Full model ID (e.g., "alexandra/square-7b").

    Returns:
        Shortened display name (e.g., "square-7b").
    """
    if "/" in model_id:
        return model_id.split("/", 1)[1]
    return model_id


def _get_language_display_name(code: str) -> str:
    """Get display name for a language code.

    Args:
        code:
            Language code.

    Returns:
        Language name if known, otherwise the code itself.
    """
    all_languages = get_all_languages()
    if code in all_languages:
        return all_languages[code].name
    return code


def _create_spider_plot(
    model_scores: dict[str, dict[str, float | None]],
    languages: list[str],
    max_score: float,
) -> go.Figure:
    """Create a Plotly spider/radial plot.

    Creates a plot with rank score (lower is better), so the radial axis
    is always reversed.

    Args:
        model_scores:
            Nested dict of model -> language -> score (or None).
        languages:
            Language codes in order.
        max_score:
            Maximum score for radial axis.

    Returns:
        Plotly figure.
    """
    languages_display = [_get_language_display_name(lang) for lang in languages]

    colours = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    fig = go.Figure()

    for idx, (model_id, lang_scores) in enumerate(model_scores.items()):
        scores = [lang_scores.get(lang, 0) or 0 for lang in languages]
        display_name = _normalise_model_name(model_id)
        colour = colours[idx % len(colours)]

        fig.add_trace(
            go.Scatterpolar(
                r=scores,
                theta=languages_display,
                fill="toself",
                name=display_name,
                line=dict(color=colour, width=2),
                fillcolor=colour.replace(")", ", 0.2)").replace("rgb", "rgba"),
            )
        )

    radial_axis = dict(visible=True, range=[max_score, 0], tickformat=".0f")

    fig.update_layout(
        polar=dict(radialaxis=radial_axis),
        showlegend=True,
        title=dict(text="Language Performance Comparison", x=0.5, y=0.95),
        height=700,
        width=900,
    )

    return fig


@click.command()
@click.option(
    "--model",
    "-m",
    "models",
    multiple=True,
    required=True,
    metavar="MODEL",
    help="Model ID to include (can be repeated).",
)
@click.option(
    "--language",
    "-l",
    "languages",
    multiple=True,
    metavar="LANGUAGE",
    help="Language name or code to include (can be repeated). "
    "Defaults to official languages.",
)
@click.option(
    "--shots",
    type=click.Choice(["auto", "zero", "few"]),
    default="auto",
    show_default=True,
    help="Shot setting: zero-shot, few-shot, or auto-detect.",
)
@click.option(
    "--max-score",
    type=float,
    metavar="FLOAT",
    help=(
        "Optional override for maximum rank score on the radial axis. "
        "When omitted, auto-computed from plotted rank scores (rounded up "
        "to nearest 0.5, minimum 2.5; rank score of 1 is perfect)."
    ),
)
def cli(
    models: tuple[str, ...],
    languages: tuple[str, ...],
    shots: str,
    max_score: float | None,
) -> None:
    """Command-line entry point."""
    exit_code = main(
        models=models, languages=languages, shots=shots, max_score=max_score
    )
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    cli()
