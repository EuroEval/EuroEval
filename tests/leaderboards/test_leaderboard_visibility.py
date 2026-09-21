"""Tests for shared leaderboard visibility rules."""

import csv
import json
from pathlib import Path

import pytest

from leaderboards.leaderboard_visibility import leaderboard_should_be_shown
from src.scripts import collect_evaluation_results, generate_leaderboards


def test_category_manifest_uses_visibility_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frontend manifest exposes the shared threshold decision."""
    output_dir = tmp_path / "csv"
    output_dir.mkdir()
    _write_simplified_leaderboard(
        path=output_dir / "hidden_chat_simplified.csv", num_entries=99
    )
    _write_simplified_leaderboard(
        path=output_dir / "shown_chat_simplified.csv", num_entries=100
    )
    monkeypatch.setattr(generate_leaderboards, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(generate_leaderboards, "REPO_ROOT", tmp_path)

    generate_leaderboards.generate_category_ranked()

    manifest_path = tmp_path / "src" / "frontend" / "generated" / "category-ranked.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["hidden"]["chat"] is False
    assert manifest["shown"]["chat"] is True


def _write_simplified_leaderboard(path: Path, num_entries: int) -> None:
    with path.open(mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "model", "mean_rank_score"])
        writer.writeheader()
        for rank in range(1, num_entries + 1):
            writer.writerow(
                {"rank": rank, "model": f"model-{rank}", "mean_rank_score": rank}
            )


@pytest.mark.parametrize(("num_entries", "expected"), [(99, False), (100, True)])
def test_leaderboard_should_be_shown_at_ranked_entry_threshold(
    tmp_path: Path, num_entries: int, expected: bool
) -> None:
    """A leaderboard is shown if and only if it has at least 100 ranked rows."""
    csv_path = tmp_path / "leaderboard_simplified.csv"
    _write_simplified_leaderboard(path=csv_path, num_entries=num_entries)

    assert leaderboard_should_be_shown(simplified_csv_path=csv_path) is expected


def test_missing_leaderboard_should_not_be_shown(tmp_path: Path) -> None:
    """A missing simplified CSV cannot represent a visible leaderboard."""
    assert not leaderboard_should_be_shown(
        simplified_csv_path=tmp_path / "missing_simplified.csv"
    )


def test_verification_prunes_all_files_for_hidden_leaderboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publication pruning uses the shared rule and removes the complete file set."""
    output_dir = tmp_path / "src" / "frontend" / "csv"
    output_dir.mkdir(parents=True)
    hidden_stem = output_dir / "hidden_chat"
    shown_csv = output_dir / "shown_chat_simplified.csv"
    _write_simplified_leaderboard(
        path=hidden_stem.with_name("hidden_chat_simplified.csv"), num_entries=99
    )
    hidden_stem.with_suffix(".csv").write_text("invalid\n", encoding="utf-8")
    hidden_stem.with_suffix(".json").write_text("{}\n", encoding="utf-8")
    _write_simplified_leaderboard(path=shown_csv, num_entries=100)
    monkeypatch.setattr(collect_evaluation_results, "REPO_ROOT", tmp_path)

    assert collect_evaluation_results.verify_leaderboards()
    assert not hidden_stem.with_suffix(".csv").exists()
    assert not hidden_stem.with_name("hidden_chat_simplified.csv").exists()
    assert not hidden_stem.with_suffix(".json").exists()
    assert shown_csv.exists()
