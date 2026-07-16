"""Create a language spider/radial plot comparing models across languages.

This script loads evaluation results from local JSONL files and generates an
interactive Plotly polar chart comparing selected models across languages.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import typing as t
from pathlib import Path

import click
import plotly.graph_objects as go

from euroeval.jsonl_io import parse_jsonl_lines
from euroeval.languages import get_all_languages
from leaderboards.constants import RESULTS_DIR
from leaderboards.record_fields import get_few_shot, get_total_scores
from leaderboards.task_metadata import (
    language_name_to_codes,
    languages_with_official_datasets,
    task_metric_names,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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

    # Try as a language name first (using leaderboard mapping)
    codes_from_name = language_name_to_codes(lang_input)
    if codes_from_name:
        return codes_from_name

    # Try as a direct language code
    all_languages = get_all_languages()
    if lang_input in all_languages:
        return {lang_input}

    # Check if it's a case mismatch for a direct code
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
        # Use official language names and convert to codes
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


def _extract_languages_from_record(record: dict[str, t.Any]) -> list[str]:
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
    additional = eval_lib.get("additional_details", {})
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


def _extract_task_from_record(record: dict[str, t.Any]) -> str | None:
    """Extract task name from an EEE-format record.

    Args:
        record:
            EEE-format result record.

    Returns:
        Task name, or None if not found.
    """
    return record.get("eval_library", {}).get("additional_details", {}).get("task")


def _extract_scores_from_record(record: dict[str, t.Any]) -> dict[str, float]:
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


def _get_model_identifier(record: dict[str, t.Any]) -> str:
    """Extract model identifier from an EEE-format record.

    Args:
        record:
            EEE-format result record.

    Returns:
        Model name/ID.
    """
    model_info = record.get("model_info", {})
    return model_info.get("name", "") or model_info.get("id", "")


def _load_results_for_models(model_ids: list[str]) -> list[dict[str, t.Any]]:
    """Load all results for the specified model IDs from local JSONL files.

    Args:
        model_ids:
            List of model IDs to load results for.

    Returns:
        List of EEE-format result records.
    """
    records: list[dict[str, t.Any]] = []
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
            records.extend(file_records)
            logger.info("Loaded %d records from %s", len(file_records), jsonl_path.name)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", jsonl_path, exc)

    return records


def _filter_by_shots(
    records: list[dict[str, t.Any]], shots_setting: t.Literal["auto", "zero", "few"]
) -> list[dict[str, t.Any]]:
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
    records: list[dict[str, t.Any]],
    models: list[str],
    languages: list[str],
    metric: str,
    shot_value: bool | None,
) -> dict[str, dict[str, float | None]]:
    """Build a model x language score matrix from records.

    Aggregates scores properly: uses the first valid score found for each
    model-language combination (does not replace with None from later records).

    Args:
        records:
            Result records.
        models:
            Model IDs.
        languages:
            Language codes.
        metric:
            Metric name to extract.
        shot_value:
            None for any, True for few-shot, False for zero-shot.

    Returns:
        Model x language score matrix.
    """
    matrix: dict[str, dict[str, float | None]] = {}
    for model in models:
        matrix[model] = {lang: None for lang in languages}

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

        record_languages = _extract_languages_from_record(record)
        scores = _extract_scores_from_record(record)

        score: float | None = None
        if metric in scores:
            score = scores[metric]
        elif metric == "primary":
            task = _extract_task_from_record(record)
            if task:
                primary_metric = _get_primary_metric_for_task(task)
                if primary_metric in scores:
                    score = scores[primary_metric]
            if score is None:
                for fallback_metric in ["test_macro_f1", "test_accuracy", "test_rouge"]:
                    if fallback_metric in scores:
                        score = scores[fallback_metric]
                        break

        if score is None or not math.isfinite(score):
            continue

        for lang in record_languages:
            if lang in languages:
                existing = matrix[matching_model][lang]
                if existing is None:
                    matrix[matching_model][lang] = score

    return matrix


def _check_completeness(model_scores: dict[str, dict[str, float | None]]) -> bool:
    """Check if all model-language combinations have scores.

    Args:
        model_scores:
            Nested dict of model -> language -> score (or None).

    Returns:
        True if complete (no None values), False otherwise.
    """
    for model_id, lang_scores in model_scores.items():
        for lang, score in lang_scores.items():
            if score is None:
                return False
    return True


def _compute_max_score(
    model_scores: dict[str, dict[str, float | None]], max_score_override: float | None
) -> float:
    """Compute or validate maximum score for radial axis.

    Validates that max_score is finite, positive, and >= all plotted scores.

    Args:
        model_scores:
            Nested dict of model -> language -> score (or None).
        max_score_override (optional):
            User-provided max score override.

    Returns:
        Maximum score value.

    Raises:
        ValueError:
            If override is invalid (NaN, inf, non-positive, or too small).
    """
    all_scores: list[float] = []
    for lang_scores in model_scores.values():
        for score in lang_scores.values():
            if score is not None and math.isfinite(score) and score > 0:
                all_scores.append(score)

    if not all_scores:
        return 100.0

    max_found = max(all_scores)

    if max_score_override is not None:
        if not math.isfinite(max_score_override):
            raise ValueError(
                f"max-score {max_score_override} is invalid (must be finite)."
            )
        if max_score_override <= 0:
            raise ValueError(
                f"max-score {max_score_override} is invalid (must be positive)."
            )
        if max_score_override < max_found:
            raise ValueError(
                f"max-score {max_score_override} is too small; "
                f"found scores up to {max_found:.2f}"
            )
        return max_score_override

    return math.ceil(max_found / 10) * 10


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
    lower_is_better: bool = False,
) -> go.Figure:
    """Create a Plotly spider/radial plot.

    Args:
        model_scores:
            Nested dict of model -> language -> score (or None).
        languages:
            Language codes in order.
        max_score:
            Maximum score for radial axis.
        lower_is_better (optional):
            If True, reverse the radial axis so lower values are better.
            Defaults to False.

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

    radial_axis: dict[str, t.Any] = dict(
        visible=True,
        range=[0, max_score] if not lower_is_better else [max_score, 0],
        tickformat=".0f",
    )

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
    "--metric",
    default="primary",
    show_default=True,
    help="Metric to use for scores. 'primary' uses task-specific primary metric.",
)
@click.option(
    "--max-score",
    type=float,
    metavar="FLOAT",
    help="Override maximum score for radial axis.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=str),
    default="language_spider_plot.html",
    show_default=True,
    help="Output HTML file path.",
)
@click.option(
    "--lower-is-better",
    is_flag=True,
    default=False,
    show_default=True,
    help="Reverse radial axis so lower values are better (for rank-like metrics).",
)
@click.option(
    "--require-complete/--allow-incomplete",
    default=True,
    show_default=True,
    help="Whether to require all model-language scores to be present.",
)
def main(
    models: tuple[str, ...],
    languages: tuple[str, ...],
    shots: str,
    metric: str,
    max_score: float | None,
    output: str,
    lower_is_better: bool,
    require_complete: bool,
) -> int:
    """Create a language spider plot comparing models across languages.

    Loads evaluation results from local JSONL files and generates an
    interactive Plotly polar chart comparing selected models across languages.

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
        logger.info("Using %d official languages by default", len(resolved_languages))

    logger.info("Loading results for %d model(s)", len(model_list))
    records = _load_results_for_models(model_list)

    if not records:
        click.echo("Error: No results found for specified models", err=True)
        sys.exit(1)

    logger.info("Loaded %d total records", len(records))

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

    logger.info("Using %d records after shot filtering", len(filtered_records))

    shot_value: bool | None = None
    if shots == "zero":
        shot_value = False
    elif shots == "few":
        shot_value = True

    model_scores_matrix = _build_score_matrix(
        records=filtered_records,
        models=model_list,
        languages=resolved_languages,
        metric=metric,
        shot_value=shot_value,
    )

    if require_complete:
        for model_id, lang_scores in model_scores_matrix.items():
            missing = [lang for lang, score in lang_scores.items() if score is None]
            if missing:
                logger.warning(
                    "Model %s missing scores for %d language(s): %s",
                    model_id,
                    len(missing),
                    ", ".join(missing[:5]) + ("..." if len(missing) > 5 else ""),
                )

        completeness = _check_completeness(model_scores_matrix)
        if not completeness:
            click.echo(
                "Error: Incomplete score matrix. Some model-language combinations "
                "are missing scores. Use --allow-incomplete to proceed anyway.",
                err=True,
            )
            sys.exit(1)

    try:
        max_score_val = _compute_max_score(model_scores_matrix, max_score)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    logger.info("Max score: %.2f", max_score_val)

    fig = _create_spider_plot(
        model_scores=model_scores_matrix,
        languages=resolved_languages,
        max_score=max_score_val,
        lower_is_better=lower_is_better,
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True)
    logger.info("Wrote spider plot to %s", output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
