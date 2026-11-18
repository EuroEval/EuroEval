"""Create language spider plots from generated leaderboard scores.

The generated simplified leaderboard CSVs are the authoritative score source.
"""

from __future__ import annotations

import base64
import csv
import math
import re
import subprocess
import sys
import tempfile
import webbrowser
from html.parser import HTMLParser
from pathlib import Path

import click
import plotly.graph_objects as go

from euroeval.languages import get_all_languages
from leaderboards.constants import OUTPUT_DIR
from leaderboards.enums import LeaderboardCategory
from leaderboards.task_metadata import (
    language_name_to_codes,
    languages_with_official_datasets,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EUROEVAL_LOGO_PATH = REPO_ROOT / "gfx" / "euroeval.png"

_IMAGE_EXPORT_CODE = """
import sys

import kaleido
import plotly.io as pio

figure = pio.read_json(sys.argv[1])
figure.write_image(sys.argv[2], scale=3)
kaleido.stop_sync_server(silence_warnings=True)
"""


def main(
    models: tuple[str, ...],
    languages: tuple[str, ...],
    shots: str,
    max_score: float | None,
    title: str | None,
    filename: str | None,
    category: str = "generative",
) -> int:
    """Create a language spider plot using generated leaderboard scores.

    Args:
        models: Model IDs to include.
        languages: Language names or codes; empty selects official languages.
        shots: Shot setting: ``auto``, ``zero``, or ``few``.
        max_score: Optional maximum rank score for the radial axis.
        title: Optional plot title.
        filename: Optional output PNG filename.
        category (optional): Leaderboard category. Defaults to ``generative``.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        model_scores, shot_value, used_languages = load_language_scores(
            models=models,
            languages=tuple(_resolve_languages(list(languages) if languages else None)),
            shots=shots,
            category=category,
        )
        max_score_val = _compute_max_score(model_scores, max_score)
    except (ValueError, FileNotFoundError) as exc:
        click.echo(f"Error: {exc}", err=True)
        return 1

    fig = _create_spider_plot(
        model_scores=model_scores,
        languages=used_languages,
        max_score=max_score_val,
        title=title or _default_plot_title(shot_value=shot_value),
    )
    output_filename = _determine_output_filename(title, filename)
    output_path = Path(output_filename).resolve()
    file_uri = output_path.as_uri()
    try:
        _write_image_silent(fig, str(output_path))
        opened = webbrowser.open(file_uri)
    except Exception as exc:
        click.echo(f"Error writing or opening PNG: {exc}", err=True)
        return 1
    if not opened:
        click.echo(f"Error opening PNG: {file_uri}", err=True)
        return 1
    click.echo(f"Finished. The output plot can now be found at {file_uri}")
    return 0


def _compute_max_score(
    model_scores: dict[str, dict[str, float | None]], override: float | None
) -> float:
    """Choose and validate the radial axis maximum.

    Args:
        model_scores: Model-by-language score matrix.
        override: Optional user-provided maximum.

    Returns:
        Maximum score displayed on the radial axis.

    Raises:
        ValueError: If override is not positive or below a plotted score.
    """
    scores = [
        score
        for values in model_scores.values()
        for score in values.values()
        if score is not None
    ]
    maximum = max(scores, default=1.0)
    if override is not None:
        if override < maximum or override <= 1:
            raise ValueError(
                f"max-score must be greater than all plotted scores ({maximum})."
            )
        return override
    return max(2.5, math.ceil(maximum * 2) / 2)


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


def _load_logo_data_uri() -> str:
    """Load the EuroEval logo as a PNG data URI.

    Returns:
        Base64-encoded PNG data URI.
    """
    encoded_logo = base64.b64encode(EUROEVAL_LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded_logo}"


def _normalise_model_name(model_id: str) -> str:
    """Normalise whitespace and case for model identity matching.

    Args:
        model_id: Model identifier from user input or leaderboard display.

    Returns:
        Normalised identifier.
    """
    return " ".join(model_id.strip().casefold().split())


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


def load_language_scores(
    models: tuple[str, ...],
    languages: tuple[str, ...],
    shots: str = "auto",
    category: str = "generative",
    csv_dir: Path | None = None,
) -> tuple[dict[str, dict[str, float | None]], bool, list[str]]:
    """Load exact mean rank scores from generated per-language CSVs.

    Args:
        models: Model identifiers to select.
        languages: Canonical language codes to include.
        shots: ``auto``, ``zero``, or ``few``; auto rejects mixed shot modes.
        category (optional): Supported leaderboard category.
        csv_dir (optional): Directory containing generated simplified CSVs.

    Returns:
        Model-by-language score matrix, selected few-shot flag, and languages
        with scores for every requested model.

    Raises:
        ValueError: If category, shot selection, models, or scores are invalid.
        FileNotFoundError: If a requested language CSV is missing.
    """
    valid_categories = {item.value for item in LeaderboardCategory}
    if category not in valid_categories:
        choices = ", ".join(sorted(valid_categories))
        raise ValueError(f"Unsupported category {category!r}; choose from {choices}.")
    if shots not in {"auto", "zero", "few"}:
        raise ValueError("shots must be 'auto', 'zero', or 'few'.")
    if not models:
        raise ValueError("At least one model must be selected.")

    csv_dir = csv_dir or OUTPUT_DIR
    requested = {_normalise_model_name(model): model for model in models}
    found: dict[str, dict[str, dict[bool, float]]] = {model: {} for model in models}
    split_modes: set[str] = set()
    for language in languages:
        path = csv_dir / f"{language}_{category}_simplified.csv"
        if not path.is_file():
            language_slug = _language_csv_slug(language)
            path = csv_dir / f"{language_slug}_{category}_simplified.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing leaderboard CSV for {language} ({category}): {path}"
            )
        with path.open(encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames or not {"model", "mean_rank_score"}.issubset(
                reader.fieldnames
            ):
                raise ValueError(f"Invalid leaderboard CSV columns in {path}")
            for row in reader:
                parsed = _parse_leaderboard_model(row["model"] or "", category=category)
                if parsed is None:
                    continue
                model_id, few_shot, split = parsed
                selected_model = requested.get(_normalise_model_name(model_id))
                if selected_model is None or (
                    shots != "auto" and few_shot != (shots == "few")
                ):
                    continue
                score_text = (row["mean_rank_score"] or "").strip()
                score_match = re.fullmatch(r"(\d+(?:\.\d+)?)(?:\s*±.*)?", score_text)
                if score_match is None:
                    raise ValueError(
                        f"Invalid mean_rank_score {score_text!r} in {path} "
                        f"for {model_id}."
                    )
                score = float(score_match.group(1))
                language_scores = found[selected_model].setdefault(language, {})
                _record_language_score(
                    language_scores=language_scores,
                    few_shot=few_shot,
                    score=score,
                    model_id=model_id,
                    language=language,
                )
                split_modes.add(split)

    if len(split_modes) > 1:
        raise ValueError(
            "Leaderboard CSVs contain both validation and test scores for the "
            "selected models; select a consistent split before plotting."
        )

    shot_modes = {
        shot
        for language_scores in found.values()
        for scores in language_scores.values()
        for shot in scores
    }
    if shots == "auto":
        if len(shot_modes) > 1:
            raise ValueError(
                "Auto shot selection is ambiguous: generated CSVs contain both "
                "zero-shot and few-shot scores. Specify --shots zero or --shots few."
            )
        if not shot_modes:
            raise ValueError("No leaderboard scores found for the selected models.")
        few_shot = next(iter(shot_modes))
    else:
        few_shot = shots == "few"

    matrix: dict[str, dict[str, float | None]] = {}
    for model in models:
        matrix[model] = {
            language: found[model].get(language, {}).get(few_shot)
            for language in languages
        }
    common_languages = [
        language
        for language in languages
        if all(matrix[model][language] is not None for model in models)
    ]
    if not common_languages:
        missing = []
        for model in models:
            model_missing = [
                language for language in languages if matrix[model][language] is None
            ]
            if model_missing:
                missing.append(f"{model}: {', '.join(model_missing)}")
        raise ValueError(
            "No languages have scores for all selected models. Missing scores: "
            + "; ".join(missing)
        )
    matrix = {
        model: {language: scores[language] for language in common_languages}
        for model, scores in matrix.items()
    }
    return matrix, few_shot, common_languages


def _language_csv_slug(language: str) -> str:
    """Return the generated CSV filename slug for a language code.

    Args:
        language: Canonical language code.

    Returns:
        Lowercase language name with spaces represented by underscores.
    """
    name = _get_language_display_name(language).casefold()
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def _parse_leaderboard_model(
    value: str, category: str = "generative"
) -> tuple[str, bool, str] | None:
    """Return the model ID, shot mode, and split from a CSV cell.

    Args:
        value: Plain or HTML-anchored leaderboard model cell.
        category (optional): Leaderboard category. Chat rows are zero-shot.

    Returns:
        Model ID, few-shot flag, and split, or None when the cell is not recognised.
    """
    parser = _ModelHTMLParser()
    parser.feed(value)
    display = "".join(parser.text).strip() or value.strip()
    match = re.fullmatch(
        r"(?P<model>.+?)(?:\s+\((?P<shot>zero-shot|few-shot)"
        r"(?:,\s*(?P<shot_split>val|test))?\)|\s+\((?P<split>val|test)\))?",
        display,
    )
    if match is None:
        return None
    shot = match.group("shot")
    few_shot = shot == "few-shot" if shot else category != "chat"
    split = match.group("shot_split") or match.group("split") or "test"
    return match.group("model").strip(), few_shot, split


class _ModelHTMLParser(HTMLParser):
    """Extract visible model identifiers from leaderboard HTML links."""

    def __init__(self) -> None:
        super().__init__()
        self.text: list[str] = []

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def _record_language_score(
    language_scores: dict[bool, float],
    few_shot: bool,
    score: float,
    model_id: str,
    language: str,
) -> None:
    """Store a score unless it conflicts with an existing duplicate row.

    Args:
        language_scores: Scores already collected for this model and language.
        few_shot: Shot mode for the score.
        score: Parsed score to store.
        model_id: Model identifier shown in the source CSV.
        language: Language code for the source CSV.

    Raises:
        ValueError: If another row has a different score for the same identity.
    """
    previous_score = language_scores.get(few_shot)
    if previous_score is not None and previous_score != score:
        raise ValueError(f"Conflicting duplicate scores for {model_id} in {language}.")
    language_scores[few_shot] = score


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
    help="Language name or code to include; defaults to official languages.",
)
@click.option(
    "--shots",
    type=click.Choice(["auto", "zero", "few"]),
    default="auto",
    show_default=True,
    help="Shot setting: zero-shot, few-shot, or auto-detect.",
)
@click.option(
    "--category",
    type=click.Choice([item.value for item in LeaderboardCategory]),
    default="generative",
    show_default=True,
    help="Leaderboard category supplying scores.",
)
@click.option(
    "--max-score",
    type=float,
    metavar="FLOAT",
    help="Optional override for maximum radial rank score.",
)
@click.option("--title", type=str, metavar="TEXT", help="Plot title.")
@click.option("--filename", type=str, metavar="PATH", help="Output PNG filename.")
def cli(
    models: tuple[str, ...],
    languages: tuple[str, ...],
    shots: str,
    max_score: float | None,
    title: str | None,
    filename: str | None,
    category: str,
) -> None:
    """Generate a language spider plot from generated leaderboard scores.

    Args:
        models: Model IDs to compare.
        languages: Optional language names or codes.
        shots: Shot filter, either ``auto``, ``zero``, or ``few``.
        max_score: Optional upper bound for the radial rank-score axis.
        title: Optional plot title.
        filename: Optional output path.
        category: Leaderboard category used as the score source.
    """
    exit_code = main(
        models=models,
        languages=languages,
        shots=shots,
        max_score=max_score,
        title=title,
        filename=filename,
        category=category,
    )
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    cli()
