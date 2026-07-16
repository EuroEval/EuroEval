"""Create a language spider/radial plot comparing models across languages.

This script loads evaluation results from local JSONL files and generates an
interactive Plotly polar chart comparing selected models across languages.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import typing as t
from pathlib import Path

import plotly.graph_objects as go

from euroeval.jsonl_io import parse_jsonl_lines
from euroeval.languages import get_all_languages
from leaderboards.constants import RESULTS_DIR
from leaderboards.task_metadata import languages_with_official_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv (optional):
            Argument list. Defaults to sys.argv[1:].

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create a language spider plot comparing models across languages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "-m",
        action="append",
        required=True,
        dest="models",
        metavar="MODEL",
        help="Model ID to include (can be repeated).",
    )
    parser.add_argument(
        "--language",
        "-l",
        action="append",
        dest="languages",
        metavar="LANGUAGE",
        help="Language code to include (can be repeated). Defaults to official"
        " languages.",
    )
    parser.add_argument(
        "--shots",
        choices=["auto", "zero", "few"],
        default="auto",
        help="Shot setting: zero-shot, few-shot, or auto-detect.",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        dest="max_score",
        metavar="FLOAT",
        help="Override maximum score for radial axis.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("language_spider_plot.html"),
        help="Output HTML file path.",
    )
    parser.add_argument(
        "--metric",
        default="primary",
        choices=["primary", "test_macro_f1", "test_accuracy", "test_rouge"],
        help="Metric to use for scores.",
    )
    return parser.parse_args(argv)


def load_results_for_models(model_ids: list[str]) -> list[dict[str, t.Any]]:
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


def extract_languages_from_record(record: dict[str, t.Any]) -> list[str]:
    """Extract language codes from an EEE-format record.

    Handles both modern EEE format (JSON-encoded string in eval_library) and
    legacy formats.

    Args:
        record:
            EEE-format result record.

    Returns:
        List of language codes.
    """
    # Try EEE format first
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

    # Legacy format fallback
    languages_value = record.get("languages")
    if isinstance(languages_value, str):
        try:
            languages = json.loads(languages_value)
            if isinstance(languages, list):
                return [str(lang) for lang in languages]
        except (json.JSONDecodeError, TypeError):
            return [languages_value]
    elif isinstance(languages_value, list):
        return [str(lang) for lang in languages_value]

    return []


def extract_score_from_record(record: dict[str, t.Any], metric: str) -> float | None:
    """Extract the primary score from an EEE-format record.

    Args:
        record:
            EEE-format result record.
        metric:
            Metric name to extract.

    Returns:
        Score as float, or None if not found.
    """
    eval_results = record.get("evaluation_results", [])
    for eval_result in eval_results:
        eval_name = eval_result.get("evaluation_name", "")
        if eval_name == metric:
            score_details = eval_result.get("score_details", {})
            score = score_details.get("score")
            if score is not None:
                try:
                    return float(score)
                except (TypeError, ValueError):
                    pass

    return None


def is_few_shot(record: dict[str, t.Any]) -> bool | None:
    """Determine if a record is few-shot from EEE format.

    Args:
        record:
            EEE-format result record.

    Returns:
        True if few-shot, False if zero-shot, None if unknown.
    """
    eval_lib = record.get("eval_library", {})
    additional = eval_lib.get("additional_details", {})
    few_shot_str = additional.get("few_shot")

    if few_shot_str is None:
        return None

    if isinstance(few_shot_str, bool):
        return few_shot_str

    if few_shot_str.lower() == "true":
        return True
    if few_shot_str.lower() == "false":
        return False

    return None


def get_model_name_from_record(record: dict[str, t.Any]) -> str:
    """Extract model name from an EEE-format record.

    Args:
        record:
            EEE-format result record.

    Returns:
        Model name/ID.
    """
    model_info = record.get("model_info", {})
    return model_info.get("name", "") or model_info.get("id", "")


def check_completeness(
    model_scores: dict[str, dict[str, float | None]], shots_filter: bool | None
) -> bool:
    """Check if all model-language combinations have scores.

    Args:
        model_scores:
            Nested dict of model -> language -> score (or None).
        shots_filter:
            None for any, True for few-shot only, False for zero-shot only.

    Returns:
        True if complete (no None values), False otherwise.
    """
    for model_id, lang_scores in model_scores.items():
        for lang, score in lang_scores.items():
            if score is None:
                return False
    return True


def filter_by_shots(
    records: list[dict[str, t.Any]],
    shots_setting: t.Literal["auto", "zero", "few"],
    models: list[str],
    languages: list[str],
) -> list[dict[str, t.Any]] | None:
    """Filter records by shot setting, with auto-detection support.

    Args:
        records:
            All result records.
        shots_setting:
            Shot setting: "auto", "zero", or "few".
        models:
            Model IDs to consider.
        languages:
            Language codes to consider.

    Returns:
        Filtered records, or None if auto-detection is ambiguous.

    """
    if shots_setting == "zero":
        return [r for r in records if is_few_shot(r) is False]
    elif shots_setting == "few":
        return [r for r in records if is_few_shot(r) is True]

    # Auto-detection: check completeness for both zero and few
    zero_records = [r for r in records if is_few_shot(r) is False]
    few_records = [r for r in records if is_few_shot(r) is True]

    def build_score_matrix(
        recs: list[dict[str, t.Any]],
    ) -> dict[str, dict[str, float | None]]:
        """Build model x language score matrix.

        Args:
            recs:
                Records to build matrix from.

        Returns:
            Model x language score matrix with None for missing scores.
        """
        matrix: dict[str, dict[str, float | None]] = {}
        for model in models:
            matrix[model] = {lang: None for lang in languages}

        for record in recs:
            model_name = get_model_name_from_record(record)
            if not any(model_name == m or model_name.endswith("/" + m) for m in models):
                continue

            record_languages = extract_languages_from_record(record)
            for lang in record_languages:
                if lang in languages:
                    score = extract_score_from_record(record, metric="test_macro_f1")
                    if score is None:
                        score = extract_score_from_record(
                            record, metric="test_accuracy"
                        )
                    if score is None:
                        score = extract_score_from_record(record, metric="test_rouge")

                    for model in models:
                        if model_name == model or model_name.endswith("/" + model):
                            matrix[model][lang] = score

        return matrix

    zero_matrix = build_score_matrix(zero_records)
    few_matrix = build_score_matrix(few_records)

    zero_complete = check_completeness(zero_matrix, shots_filter=None)
    few_complete = check_completeness(few_matrix, shots_filter=None)

    if zero_complete and not few_complete:
        logger.info("Auto-detected: using zero-shot (few-shot incomplete)")
        return zero_records
    elif few_complete and not zero_complete:
        logger.info("Auto-detected: using few-shot (zero-shot incomplete)")
        return few_records
    elif zero_complete and few_complete:
        logger.error(
            "Auto-detection ambiguous: both zero-shot and few-shot are complete. "
            "Please specify --shots zero or --shots few explicitly."
        )
        return None
    else:
        logger.error(
            "Auto-detection failed: neither zero-shot nor few-shot is complete. "
            "Please specify --shots zero or --shots few explicitly."
        )
        return None


def compute_language_scores(
    records: list[dict[str, t.Any]],
    models: list[str],
    languages: list[str],
    metric: str,
) -> dict[str, dict[str, float]]:
    """Compute scores per model per language.

    Args:
        records:
            Filtered result records.
        models:
            Model IDs.
        languages:
            Language codes.
        metric:
            Metric to use.

    Returns:
        Nested dict of model -> language -> score.

    Raises:
        ValueError:
            If no scores found.
    """
    model_scores: dict[str, dict[str, float]] = {}
    for model in models:
        model_scores[model] = {}

    for record in records:
        model_name = get_model_name_from_record(record)
        matching_model = None
        for model in models:
            if model_name == model or model_name.endswith("/" + model):
                matching_model = model
                break

        if matching_model is None:
            continue

        record_languages = extract_languages_from_record(record)
        score = extract_score_from_record(record, metric)

        if score is None and metric == "primary":
            score = extract_score_from_record(record, "test_macro_f1")
        if score is None and metric == "primary":
            score = extract_score_from_record(record, "test_accuracy")
        if score is None and metric == "primary":
            score = extract_score_from_record(record, "test_rouge")

        if score is None:
            continue

        for lang in record_languages:
            if lang in languages:
                model_scores[matching_model][lang] = score

    # Validate we have scores
    if not any(scores for scores in model_scores.values()):
        raise ValueError("No scores found for the specified models and languages.")

    return model_scores


def compute_max_score(
    model_scores: dict[str, dict[str, float]], max_score_override: float | None
) -> float:
    """Compute or validate maximum score for radial axis.

    Args:
        model_scores:
            Nested dict of model -> language -> score.
        max_score_override (optional):
            User-provided max score override.

    Returns:
        Maximum score value.

    Raises:
        ValueError:
            If override is too small for the data.
    """
    all_scores = [
        score
        for lang_scores in model_scores.values()
        for score in lang_scores.values()
        if math.isfinite(score)
    ]

    if not all_scores:
        return 100.0

    auto_max = max(all_scores)

    if max_score_override is not None:
        if max_score_override < auto_max:
            raise ValueError(
                f"max-score {max_score_override} is too small; "
                f"found scores up to {auto_max:.2f}"
            )
        return max_score_override

    # Round up to nearest 10 for nice axis
    return math.ceil(auto_max / 10) * 10


def normalise_model_name(model_id: str) -> str:
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


def create_spider_plot(
    model_scores: dict[str, dict[str, float]], languages: list[str], max_score: float
) -> go.Figure:
    """Create a Plotly spider/radial plot.

    Args:
        model_scores:
            Nested dict of model -> language -> score.
        languages:
            Language codes in order.
        max_score:
            Maximum score for radial axis.

    Returns:
        Plotly figure.
    """
    languages_display = [
        get_all_languages().get(lang, {}).name or lang for lang in languages
    ]

    # Colour palette for models
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
        scores = [lang_scores.get(lang, 0) for lang in languages]
        display_name = normalise_model_name(model_id)
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

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max_score], tickformat=".0f")
        ),
        showlegend=True,
        title=dict(text="Language Performance Comparison", x=0.5, y=0.95),
        height=700,
        width=900,
    )

    return fig


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv (optional):
            Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args(argv)

    # Validate models provided
    if not args.models:
        logger.error("At least one model must be specified with --model/-m")
        return 1

    # Determine languages
    if args.languages:
        languages = args.languages
    else:
        languages = languages_with_official_datasets()
        logger.info("Using %d official languages by default", len(languages))

    # Load results
    logger.info("Loading results for %d model(s)", len(args.models))
    records = load_results_for_models(args.models)

    if not records:
        logger.error("No results found for specified models")
        return 1

    logger.info("Loaded %d total records", len(records))

    # Filter by shots
    if args.shots == "auto":
        filtered_records = filter_by_shots(records, args.shots, args.models, languages)
        if filtered_records is None:
            return 1
    else:
        filtered_records = filter_by_shots(records, args.shots, args.models, languages)
        if filtered_records is None:
            return 1

    if not filtered_records:
        logger.error("No records after shot filtering")
        return 1

    logger.info("Using %d records after shot filtering", len(filtered_records))

    # Compute scores
    try:
        model_scores = compute_language_scores(
            records=filtered_records,
            models=args.models,
            languages=languages,
            metric=args.metric,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    # Validate completeness
    for model_id, lang_scores in model_scores.items():
        missing = [lang for lang, score in lang_scores.items() if score is None]
        if missing:
            logger.warning(
                "Model %s missing scores for %d language(s): %s",
                model_id,
                len(missing),
                ", ".join(missing[:5]) + ("..." if len(missing) > 5 else ""),
            )

    # Compute max score
    try:
        max_score = compute_max_score(model_scores, args.max_score)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Max score: %.2f", max_score)

    # Create plot
    fig = create_spider_plot(model_scores, languages, max_score)

    # Write output
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True)
    logger.info("Wrote spider plot to %s", output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
