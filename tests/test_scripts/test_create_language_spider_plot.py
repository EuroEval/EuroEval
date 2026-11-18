"""Tests for language spider plots using generated leaderboard CSVs."""

import csv
from pathlib import Path

import click.testing
import pytest

import scripts.create_language_spider_plot as spider_plot


def test_auto_shots_rejects_mixed_shot_modes_across_languages(tmp_path: Path) -> None:
    """Auto selection is consistent across all requested language axes."""
    _write_csv(tmp_path, "fo", "generative", [("model-a (zero-shot, val)", "1.2")])
    _write_csv(tmp_path, "is", "generative", [("model-a (few-shot, val)", "1.3")])

    with pytest.raises(ValueError, match="ambiguous"):
        spider_plot.load_language_scores(
            models=("model-a",), languages=("fo", "is"), csv_dir=tmp_path
        )


def _write_csv(
    directory: Path, language: str, category: str, rows: list[tuple[str, str]]
) -> None:
    """Write a minimal generated leaderboard CSV.

    Args:
        directory: Destination directory.
        language: Language code used in the generated filename.
        category: Leaderboard category.
        rows: Model-cell and mean-rank-score pairs.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{language}_{category}_simplified.csv"
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["rank", "model", "mean_rank_score"]
        )
        writer.writeheader()
        for model, score in rows:
            writer.writerow({"rank": 1, "model": model, "mean_rank_score": score})


def test_category_selects_its_own_generated_csv(tmp_path: Path) -> None:
    """Scores come only from the explicitly selected category."""
    for category, score in (("generative", "1.4 ± 0.1"), ("chat", "2.2 ± 0.1")):
        _write_csv(tmp_path, "fo", category, [("model-a (zero-shot, val)", score)])

    scores, _, _ = spider_plot.load_language_scores(
        models=("model-a",), languages=("fo",), category="chat", csv_dir=tmp_path
    )

    assert scores["model-a"]["fo"] == 2.2


@pytest.mark.parametrize("annotation", ["", " (val)"])
def test_chat_bare_and_validation_rows_are_zero_shot(
    tmp_path: Path, annotation: str
) -> None:
    """Chat's unannotated generated rows are zero-shot by construction.

    Args:
        tmp_path:
            Temporary leaderboard directory.
        annotation:
            Split annotation emitted by the leaderboard generator.
    """
    _write_csv(tmp_path, "fo", "chat", [(f"model-a{annotation}", "1.3")])

    scores, few_shot, _ = spider_plot.load_language_scores(
        models=("model-a",),
        languages=("fo",),
        shots="zero",
        category="chat",
        csv_dir=tmp_path,
    )

    assert scores == {"model-a": {"fo": 1.3}}
    assert few_shot is False
    with pytest.raises(ValueError, match="No languages have scores"):
        spider_plot.load_language_scores(
            models=("model-a",),
            languages=("fo",),
            shots="few",
            category="chat",
            csv_dir=tmp_path,
        )


def test_cli_category_and_success_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI forwards category to main and preserves one-line successful output."""
    _write_csv(tmp_path, "fo", "chat", [("model-a (zero-shot, val)", "1.2 ± 0.1")])
    monkeypatch.setattr(spider_plot, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(spider_plot, "_write_image_silent", lambda fig, path: None)
    monkeypatch.setattr(spider_plot.webbrowser, "open", lambda _uri: True)

    result = click.testing.CliRunner().invoke(
        spider_plot.cli,
        ["--model", "model-a", "--language", "fo", "--category", "chat"],
    )

    assert result.exit_code == 0
    assert result.output.startswith(
        "Finished. The output plot can now be found at file://"
    )
    assert "Error" not in result.output


def test_conflicting_duplicate_rows_are_rejected(tmp_path: Path) -> None:
    """Conflicting duplicate model, language, and shot rows are not overwritten."""
    _write_csv(
        tmp_path,
        "fo",
        "generative",
        [("model-a (zero-shot, val)", "1.2"), ("model-a (zero-shot, val)", "1.3")],
    )

    with pytest.raises(ValueError, match="Conflicting duplicate scores"):
        spider_plot.load_language_scores(
            models=("model-a",), languages=("fo",), csv_dir=tmp_path
        )


@pytest.mark.parametrize(
    ("annotation", "expected_few_shot"),
    [
        ("", True),
        (" (val)", True),
        (" (zero-shot)", False),
        (" (zero-shot, val)", False),
    ],
)
def test_emitted_model_formats_preserve_identifier_suffixes(
    tmp_path: Path, annotation: str, expected_few_shot: bool
) -> None:
    """Generated annotations are parsed without consuming ID suffixes.

    Args:
        tmp_path:
            Temporary leaderboard directory.
        annotation:
            Shot and split annotation emitted by the generator.
        expected_few_shot:
            Expected shot mode for the generated row.
    """
    _write_csv(
        tmp_path, "fo", "generative", [(f"model#thinking@revision{annotation}", "1.4")]
    )

    scores, few_shot, languages = spider_plot.load_language_scores(
        models=("model#thinking@revision",), languages=("fo",), csv_dir=tmp_path
    )

    assert scores == {"model#thinking@revision": {"fo": 1.4}}
    assert few_shot is expected_few_shot
    assert languages == ["fo"]


def test_excluded_shot_rows_do_not_participate_in_split_validation(
    tmp_path: Path,
) -> None:
    """Split consistency applies only to rows selected by the shot filter."""
    _write_csv(
        tmp_path,
        "fo",
        "generative",
        [("model-a (few-shot, val)", "1.2"), ("model-a (zero-shot, test)", "1.3")],
    )

    scores, few_shot, _ = spider_plot.load_language_scores(
        models=("model-a",), languages=("fo",), shots="few", csv_dir=tmp_path
    )

    assert scores == {"model-a": {"fo": 1.2}}
    assert few_shot is True


def test_invalid_category_columns_and_scores_are_rejected(tmp_path: Path) -> None:
    """Unexpected category, schema, and score data fail clearly."""
    with pytest.raises(ValueError, match="Unsupported category"):
        spider_plot.load_language_scores(
            models=("model-a",), languages=("fo",), category="unknown", csv_dir=tmp_path
        )

    path = tmp_path / "fo_generative_simplified.csv"
    path.write_text("model,wrong_column\nmodel-a,1.2\n")
    with pytest.raises(ValueError, match="Invalid leaderboard CSV columns"):
        spider_plot.load_language_scores(
            models=("model-a",), languages=("fo",), csv_dir=tmp_path
        )

    _write_csv(tmp_path, "fo", "generative", [("model-a (zero-shot, val)", "NaN")])
    with pytest.raises(ValueError, match="Invalid mean_rank_score"):
        spider_plot.load_language_scores(
            models=("model-a",), languages=("fo",), csv_dir=tmp_path
        )


def test_matrix_and_main_axis_exclude_non_common_languages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only shared language scores contribute to plotted data and axis bounds."""
    _write_csv(
        tmp_path,
        "fo",
        "generative",
        [("model-a (zero-shot, val)", "1.2"), ("model-b (zero-shot, val)", "1.3")],
    )
    _write_csv(tmp_path, "is", "generative", [("model-a (zero-shot, val)", "100.0")])
    monkeypatch.setattr(spider_plot, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(spider_plot, "_resolve_languages", lambda languages: languages)
    monkeypatch.setattr(spider_plot, "_write_image_silent", lambda fig, path: None)
    monkeypatch.setattr(spider_plot.webbrowser, "open", lambda _uri: True)

    scores, _, languages = spider_plot.load_language_scores(
        models=("model-a", "model-b"), languages=("fo", "is"), csv_dir=tmp_path
    )
    exit_code = spider_plot.main(
        models=("model-a", "model-b"),
        languages=("fo", "is"),
        shots="auto",
        max_score=2.0,
        title=None,
        filename=str(tmp_path / "plot.png"),
    )

    assert scores == {"model-a": {"fo": 1.2}, "model-b": {"fo": 1.3}}
    assert languages == ["fo"]
    assert exit_code == 0


def test_missing_csv_and_missing_model_scores_are_informative(tmp_path: Path) -> None:
    """Missing source files and absent model rows do not fabricate scores."""
    with pytest.raises(FileNotFoundError, match="Missing leaderboard CSV"):
        spider_plot.load_language_scores(
            models=("model-a",), languages=("fo",), csv_dir=tmp_path
        )

    _write_csv(tmp_path, "fo", "generative", [("model-b (zero-shot, val)", "1.2")])
    with pytest.raises(ValueError, match="No leaderboard scores found"):
        spider_plot.load_language_scores(
            models=("model-a",), languages=("fo",), csv_dir=tmp_path
        )


def test_mixed_splits_are_rejected_instead_of_mixed_across_languages(
    tmp_path: Path,
) -> None:
    """Validation and test scores cannot be combined into one plot."""
    _write_csv(tmp_path, "fo", "generative", [("model-a (zero-shot, val)", "1.2")])
    _write_csv(tmp_path, "is", "generative", [("model-a (zero-shot, test)", "1.3")])

    with pytest.raises(ValueError, match="both validation and test"):
        spider_plot.load_language_scores(
            models=("model-a",), languages=("fo", "is"), csv_dir=tmp_path
        )


def test_scores_are_read_exactly_from_html_model_cells(tmp_path: Path) -> None:
    """HTML links and score uncertainty resolve to the exact displayed scores."""
    _write_csv(
        tmp_path,
        "fo",
        "generative",
        [
            (
                "<a href='https://example.com/gpt-6'>gpt-6-astra (zero-shot, val)</a>",
                "1.38 ± 0.05",
            ),
            (
                "<a href='https://example.com/gpt-6.2'>"
                "gpt-6.2-astra (zero-shot, val)</a>",
                "1.31 ± 0.05",
            ),
        ],
    )

    scores, few_shot, languages = spider_plot.load_language_scores(
        models=("gpt-6-astra", "gpt-6.2-astra"), languages=("fo",), csv_dir=tmp_path
    )

    assert scores == {"gpt-6-astra": {"fo": 1.38}, "gpt-6.2-astra": {"fo": 1.31}}
    assert few_shot is False
    assert languages == ["fo"]


def test_scores_do_not_require_or_recalculate_raw_task_results(tmp_path: Path) -> None:
    """The generated score remains authoritative regardless of raw task data."""
    _write_csv(
        tmp_path, "fo", "generative", [("model-a (zero-shot, val)", "1.38 ± 0.05")]
    )

    scores, _, _ = spider_plot.load_language_scores(
        models=("model-a",), languages=("fo",), csv_dir=tmp_path
    )

    assert scores["model-a"]["fo"] == 1.38


def test_shot_option_selects_only_matching_rows(tmp_path: Path) -> None:
    """Explicit shot selection ignores the other shot's CSV row."""
    _write_csv(
        tmp_path,
        "fo",
        "generative",
        [("model-a (zero-shot, val)", "1.2"), ("model-a (few-shot, val)", "1.8")],
    )

    scores, few_shot, _ = spider_plot.load_language_scores(
        models=("model-a",), languages=("fo",), shots="few", csv_dir=tmp_path
    )

    assert scores["model-a"]["fo"] == 1.8
    assert few_shot is True
