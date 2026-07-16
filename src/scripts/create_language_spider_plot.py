"""Create a language spider/radial plot comparing models across languages.

This script loads evaluation results from local JSONL files and generates a
Plotly polar chart comparing selected models across languages.
Only rank score is plotted (lower is better, axis is reversed).
Output is a PNG file.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import re
import subprocess
import sys
import tempfile
import typing as t
import webbrowser
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import plotly.graph_objects as go

from euroeval.constants import ORTHOGONAL_TASKS
from euroeval.jsonl_io import parse_jsonl_lines
from euroeval.languages import get_all_languages
from leaderboards.constants import RESULTS_DIR
from leaderboards.record_fields import (
    get_few_shot,
    get_raw_results,
    get_task,
    get_total_scores,
)
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

REPO_ROOT = Path(__file__).resolve().parents[2]
EUROEVAL_LOGO_PATH = REPO_ROOT / "gfx" / "euroeval.png"


# Silence logging by default; script should emit exactly one line on success
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(
    models: tuple[str, ...],
    languages: tuple[str, ...],
    shots: str,
    max_score: float | None,
    title: str | None,
    filename: str | None,
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
        title (optional):
            Plot title. If omitted, uses default title.
        filename (optional):
            Output PNG filename (without .png suffix if desired, it will be
            appended). If omitted and title is set, inferred from title using
            snake_case. If both omitted, uses "language-spider-plot.png".

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

    # Load all records as reference population (for rank score computation)
    all_records = _load_all_results()

    # Verify requested models have data
    requested_model_records = _load_results_for_models(model_list)
    if not requested_model_records:
        click.echo("Error: No results found for specified models", err=True)
        sys.exit(1)

    try:
        filtered_records = _filter_by_shots(
            requested_model_records, t.cast(t.Literal["auto", "zero", "few"], shots)
        )
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if not filtered_records:
        click.echo("Error: No records after shot filtering", err=True)
        sys.exit(1)

    shot_value = _resolve_shot_value(
        filtered_records=filtered_records,
        shots_setting=t.cast(t.Literal["auto", "zero", "few"], shots),
    )

    # Build score matrix using all records as reference population
    model_scores_matrix = _build_score_matrix(
        all_records=all_records,
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

    # Language intersection applied silently (no output on success)

    try:
        max_score_val = _compute_max_score(model_scores_matrix, max_score)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    fig = _create_spider_plot(
        model_scores=model_scores_matrix,
        languages=used_languages,
        max_score=max_score_val,
        title=title or _default_plot_title(shot_value=shot_value),
    )

    # Determine output filename
    output_filename = _determine_output_filename(title, filename)
    output_path = Path(output_filename).resolve()
    file_uri = output_path.as_uri()
    try:
        _write_image_silent(fig, str(output_path))
    except Exception as exc:
        click.echo(f"Error writing PNG: {exc}", err=True)
        return 1

    try:
        opened = webbrowser.open(file_uri)
    except Exception as exc:
        click.echo(f"Error opening PNG: {exc}", err=True)
        return 1
    if not opened:
        click.echo(f"Error opening PNG: {file_uri}", err=True)
        return 1

    click.echo(f"Finished. The output plot can now be found at {file_uri}")

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


def _extract_raw_scores_from_record(
    record: JsonDict, primary_metric: str, secondary_metric: str | None
) -> list[float]:
    """Extract raw per-iteration scores from an EEE-format record.

    Retrieves raw bootstrap/iteration scores for the primary metric (or
    secondary if primary is unavailable) from the record's raw_results.

    Args:
        record:
            EEE-format result record.
        primary_metric:
            Primary metric name (e.g. "mcc", "macro_f1").
        secondary_metric:
            Secondary metric name, or None if single-metric task.

    Returns:
        List of raw per-iteration scores. Empty list if no raw scores found.
    """
    raw_results = get_raw_results(record)
    if raw_results is None:
        return []

    raw_scores: list[float] = []
    metrics_to_try = [primary_metric]
    if secondary_metric:
        metrics_to_try.append(secondary_metric)

    for result_dict in raw_results:
        if not isinstance(result_dict, dict):
            continue
        score: float | None = None
        for metric in metrics_to_try:
            # Try with test_ prefix first, then bare metric name
            if f"test_{metric}" in result_dict:
                score = float(result_dict[f"test_{metric}"])
                break
            if metric in result_dict:
                score = float(result_dict[metric])
                break
        if score is not None and math.isfinite(score):
            raw_scores.append(score)

    # Scale normalised scores [0, 1] back to [0, 100] if needed
    if raw_scores and max(raw_scores) <= 1.0:
        raw_scores = [s * 100.0 for s in raw_scores]

    return raw_scores


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


def _load_all_results() -> list[JsonDict]:
    """Load all results from local JSONL files (reference population).

    Skips unknown.jsonl as it is not authoritative. Loads all other
    JSONL files in the results directory.

    Returns:
        List of EEE-format result records from all models.
    """
    records: list[JsonDict] = []
    model_files = sorted(RESULTS_DIR.glob("*.jsonl"))

    if not model_files:
        return records

    for jsonl_path in model_files:
        # Skip unknown.jsonl as it is not authoritative
        if jsonl_path.stem == "unknown":
            continue

        try:
            lines = jsonl_path.read_text(encoding="utf-8").splitlines()
            file_records = parse_jsonl_lines(
                lines=lines, source=str(jsonl_path), strict=False
            )
            for rec in file_records:
                if isinstance(rec, dict):
                    records.append(t.cast(JsonDict, rec))
        except Exception:
            # Silently skip unparseable files
            pass

    return records


def _load_results_for_models(model_ids: list[str]) -> list[JsonDict]:
    """Load results for the specified model IDs from local JSONL files.

    Used to verify requested models have data. For rank score computation,
    use _load_all_results() to get the full reference population.

    Args:
        model_ids:
            List of model IDs to load results for.

    Returns:
        List of EEE-format result records for the specified models.
    """
    all_records = _load_all_results()
    records: list[JsonDict] = []

    for record in all_records:
        model_name = _get_model_identifier(record)
        if any(model_name == m or model_name.endswith("/" + m) for m in model_ids):
            records.append(record)

    return records


def _resolve_shot_value(
    filtered_records: list[JsonDict], shots_setting: t.Literal["auto", "zero", "few"]
) -> bool:
    """Resolve the shot setting to a Boolean few-shot value.

    Args:
        filtered_records:
            Records remaining after shot filtering.
        shots_setting:
            User-requested shot setting.

    Returns:
        True for few-shot, False for zero-shot.
    """
    if shots_setting == "zero":
        return False
    if shots_setting == "few":
        return True
    return get_few_shot(filtered_records[0]) is True


def _default_plot_title(shot_value: bool) -> str:
    """Create the default plot title for the resolved shot setting.

    Args:
        shot_value:
            True for few-shot, False for zero-shot.

    Returns:
        Default plot title.
    """
    shot_label = "Few-shot" if shot_value else "Zero-shot"
    return f"{shot_label} EuroEval Results"


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
        return zero_records
    elif few_count > 0 and zero_count == 0:
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
    all_records: list[JsonDict],
    models: list[str],
    languages: list[str],
    shot_value: bool | None,
) -> dict[str, dict[str, float | None]]:
    """Build a model x language mean rank score matrix from records.

    Computes per-language mean rank scores following the EuroEval methodology:
    1. Extract raw per-iteration/bootstrap scores per (model, dataset, language)
    2. Compute mean_score(m,d) = mean of all raw scores for model m on dataset d
    3. Compute pooled_std(d) = std of ALL raw scores from all models on dataset d
       (uses all_records as reference population, not just requested models)
    4. Compute rank_score = 1 + (best_mean - model_mean) / pooled_std
    5. Aggregate hierarchy: dataset -> task -> language (unweighted means)

    Rank score of 1 is perfect (best on dataset). Lower is better.
    Reference population is all available records; output matrix contains
    only the requested models.

    Args:
        all_records:
            All result records (reference population for rank scores).
        models:
            Model IDs to include in output matrix.
        languages:
            Language codes to include in output matrix.
        shot_value:
            None for any, True for few-shot, False for zero-shot.

    Returns:
        Model x language mean rank score matrix (lower is better).
    """
    # Step 1: Collect raw per-iteration scores per (model, dataset, language)
    # from ALL records (reference population), not just requested models
    # Structure: model -> dataset -> language -> list of raw scores
    model_dataset_lang_raw_scores: dict[str, dict[str, dict[str, list[float]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    for record in all_records:
        model_name = _get_model_identifier(record)

        record_few_shot = get_few_shot(record)
        if shot_value is not None and record_few_shot is not None:
            if record_few_shot != shot_value:
                continue

        task_name = get_task(record)
        if not task_name or task_name in ORTHOGONAL_TASKS:
            continue

        record_languages = _extract_languages_from_record(record)
        dataset = get_dataset(record)

        if not dataset or not record_languages:
            continue

        # Get primary/secondary metric for this task
        primary_metric, secondary_metric = task_metric_names(task_name)

        # Extract raw per-iteration scores (not aggregated)
        raw_scores = _extract_raw_scores_from_record(
            record, primary_metric, secondary_metric
        )

        if not raw_scores:
            continue

        # Collect raw scores from ALL models (reference population)
        for lang in record_languages:
            if lang in languages:
                model_dataset_lang_raw_scores[model_name][dataset][lang].extend(
                    raw_scores
                )

    if not model_dataset_lang_raw_scores:
        return {model: {lang: None for lang in languages} for model in models}

    # Step 2: Compute mean score per (model, dataset) from raw scores
    # mean_score(m,d) = mean of ALL raw scores for model m on dataset d
    model_dataset_means: dict[str, dict[str, float]] = defaultdict(dict)
    for model, datasets in model_dataset_lang_raw_scores.items():
        for dataset, lang_raw_scores in datasets.items():
            all_raw_scores: list[float] = []
            for scores_list in lang_raw_scores.values():
                all_raw_scores.extend(scores_list)
            if all_raw_scores:
                model_dataset_means[model][dataset] = float(np.mean(all_raw_scores))

    # Step 3: Compute pooled_std per dataset from ALL raw scores across all models
    dataset_all_raw_scores: dict[str, list[float]] = defaultdict(list)
    for model, datasets in model_dataset_lang_raw_scores.items():
        for dataset, lang_raw_scores in datasets.items():
            for scores_list in lang_raw_scores.values():
                dataset_all_raw_scores[dataset].extend(scores_list)

    dataset_stats: dict[str, tuple[float, float]] = {}
    for dataset, all_scores in dataset_all_raw_scores.items():
        if not all_scores:
            continue
        # Check which models have this dataset
        models_with_dataset = {
            m for m, ds_means in model_dataset_means.items() if dataset in ds_means
        }
        if not models_with_dataset:
            continue
        best_mean = max(model_dataset_means[m][dataset] for m in models_with_dataset)
        pooled_std = float(np.std(all_scores)) if len(all_scores) > 1 else 1.0
        if pooled_std <= 0:
            pooled_std = 1.0
        dataset_stats[dataset] = (best_mean, pooled_std)

    # Step 4: Compute rank score per (model, dataset, language)
    # rank_score = 1 + (best_mean - model_mean) / pooled_std
    model_dataset_lang_rank_scores: dict[str, dict[str, dict[str, float]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    for model in models:
        if model not in model_dataset_lang_raw_scores:
            continue
        for dataset, lang_raw_scores in model_dataset_lang_raw_scores[model].items():
            if dataset not in dataset_stats:
                continue
            best_mean, pooled_std = dataset_stats[dataset]
            for lang, raw_scores_list in lang_raw_scores.items():
                if not raw_scores_list:
                    continue
                model_mean = float(np.mean(raw_scores_list))
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


def _normalise_model_name(model_id: str) -> str:
    """Normalise model identifier for display.

    Extracts the short model name from a full identifier like
    "author/model-name" -> "model-name".

    Args:
        model_id:
            Model identifier (may include author prefix).

    Returns:
        Normalised model name for display.
    """
    if "/" in model_id:
        return model_id.split("/")[-1]
    return model_id


def _to_snake_case(text: str) -> str:
    """Convert text to snake_case.

    Converts spaces and special characters to underscores, lowercases,
    and removes consecutive underscores.

    Args:
        text:
            Text to convert.

    Returns:
        Snake-case string.
    """
    # Replace spaces and hyphens with underscores
    result = text.replace(" ", "_").replace("-", "_")
    # Remove non-alphanumeric except underscores
    result = re.sub(r"[^a-z0-9_]", "", result.lower())
    # Remove consecutive underscores
    result = re.sub(r"_+", "_", result)
    # Remove leading/trailing underscores
    return result.strip("_")


def _determine_output_filename(title: str | None, filename: str | None) -> str:
    """Determine output PNG filename based on title and filename options.

    Priority:
    1. If filename is set, use it (append .png if missing).
    2. If title is set (but filename not), infer from title using snake_case.
    3. Otherwise, use default "language-spider-plot.png".

    Args:
        title:
            Plot title (optional).
        filename:
            Explicit filename (optional).

    Returns:
        PNG filename with .png extension.
    """
    if filename:
        # Ensure .png extension
        if not filename.lower().endswith(".png"):
            return filename + ".png"
        return filename

    if title:
        # Infer filename from title
        base_name = _to_snake_case(title)
        return base_name + ".png"

    # Default filename
    return "language-spider-plot.png"


def _hex_to_rgba(hex_colour: str, alpha: float = 0.2) -> str:
    """Convert hex colour to rgba string with specified alpha.

    Args:
        hex_colour:
            Hex colour string (e.g. "#1f77b4").
        alpha (optional):
            Alpha value (0.0 to 1.0). Defaults to 0.2.

    Returns:
        RGBA colour string (e.g. "rgba(31, 119, 180, 0.2)").
    """
    hex_val = hex_colour.lstrip("#")
    r = int(hex_val[0:2], 16)
    g = int(hex_val[2:4], 16)
    b = int(hex_val[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _create_spider_plot(
    model_scores: dict[str, dict[str, float | None]],
    languages: list[str],
    max_score: float,
    title: str | None = None,
) -> go.Figure:
    """Create a Plotly spider/radial plot.

    Creates a plot with rank score (lower is better), so the radial axis
    is always reversed. Radial axis minimum is 1 (perfect rank score),
    ticks show one decimal place.

    Args:
        model_scores:
            Nested dict of model -> language -> score (or None).
        languages:
            Language codes in order.
        max_score:
            Maximum score for radial axis.
        title (optional):
            Plot title. Uses default if omitted.

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

    closed_languages_display = [*languages_display, languages_display[0]]
    prepared_traces: list[tuple[str, list[float], str]] = []
    for idx, (model_id, lang_scores) in enumerate(model_scores.items()):
        scores = [lang_scores.get(lang, 0) or 0 for lang in languages]
        closed_scores = [*scores, scores[0]]
        display_name = _normalise_model_name(model_id)
        colour = colours[idx % len(colours)]
        prepared_traces.append((display_name, closed_scores, colour))

    for display_name, scores, colour in prepared_traces:
        fig.add_trace(
            go.Scatterpolar(
                r=scores,
                theta=closed_languages_display,
                fill="toself",
                name=display_name,
                line=dict(color=_hex_to_rgba(colour, alpha=0.0), width=0),
                fillcolor=_hex_to_rgba(colour, alpha=0.16),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for display_name, scores, colour in prepared_traces:
        fig.add_trace(
            go.Scatterpolar(
                r=scores,
                theta=closed_languages_display,
                fill="none",
                name=display_name,
                line=dict(color=colour, width=2.5),
            )
        )

    radial_axis = dict(visible=True, range=[max_score, 1], tickformat=".1f", dtick=0.5)

    plot_title = title if title else "Language Performance Comparison"

    fig.update_layout(
        polar=dict(radialaxis=radial_axis),
        showlegend=True,
        title=dict(text=plot_title, x=0.5, y=0.95),
        height=700,
        width=900,
    )
    fig.add_layout_image(
        dict(
            source=_load_logo_data_uri(),
            xref="paper",
            yref="paper",
            x=1.22,
            y=-0.14,
            sizex=0.36,
            sizey=0.36,
            xanchor="right",
            yanchor="bottom",
            sizing="contain",
            opacity=0.85,
            layer="above",
        )
    )

    return fig


def _load_logo_data_uri() -> str:
    """Load the EuroEval logo as a PNG data URI.

    Returns:
        Base64-encoded PNG data URI.
    """
    encoded_logo = base64.b64encode(EUROEVAL_LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded_logo}"


def _write_image_silent(fig: go.Figure, path: str) -> None:
    """Write Plotly figure to image without leaking Kaleido cleanup output.

    Kaleido/Chromium can write directly to stdout/stderr during interpreter
    shutdown, after normal Python redirection has ended. Running export in a
    child process lets this script capture all normal and shutdown output, then
    print only the final PNG URI on success.

    Args:
        fig:
            Plotly figure to write.
        path:
            Output file path.

    Raises:
        ClickException:
            If image export fails, includes captured diagnostic output.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        figure_path = Path(tmp_dir) / "figure.json"
        fig.write_json(figure_path)
        result = subprocess.run(
            [sys.executable, "-c", _IMAGE_EXPORT_CODE, str(figure_path), path],
            check=False,
            capture_output=True,
            text=True,
        )

    if result.returncode == 0:
        return

    diagnostic_parts: list[str] = []
    if result.stdout.strip():
        diagnostic_parts.append(f"stdout: {result.stdout.strip()}")
    if result.stderr.strip():
        diagnostic_parts.append(f"stderr: {result.stderr.strip()}")
    diagnostic = "\n".join(diagnostic_parts)
    message = "Failed to write PNG image"
    if diagnostic:
        message = f"{message}\n{diagnostic}"
    raise click.ClickException(message)


_IMAGE_EXPORT_CODE = """
import sys

import kaleido
import plotly.io as pio

figure = pio.read_json(sys.argv[1])
figure.write_image(sys.argv[2])
kaleido.stop_sync_server(silence_warnings=True)
"""


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
@click.option(
    "--title",
    type=str,
    metavar="TEXT",
    help="Plot title. If omitted, uses default title.",
)
@click.option(
    "--filename",
    type=str,
    metavar="PATH",
    help=(
        "Output PNG filename. If omitted and --title is set, "
        "inferred from title using snake_case. If both omitted, "
        "uses language-spider-plot.png. .png extension appended if missing."
    ),
)
def cli(
    models: tuple[str, ...],
    languages: tuple[str, ...],
    shots: str,
    max_score: float | None,
    title: str | None,
    filename: str | None,
) -> None:
    """Command-line entry point."""
    exit_code = main(
        models=models,
        languages=languages,
        shots=shots,
        max_score=max_score,
        title=title,
        filename=filename,
    )
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    cli()
