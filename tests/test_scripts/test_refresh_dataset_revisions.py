"""Tests for refreshing built-in dataset revisions."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.scripts import refresh_dataset_revisions as refresh

SHA = "a" * 40
NEW_SHA = "b" * 40


def test_inserts_revision_and_preserves_source_subset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing revisions are inserted without changing source subsets."""
    run_main(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        source='DatasetConfig(source="org/data::validation", name="data")\n',
        datasets={"org/data": SHA},
    )
    assert (tmp_path / "config.py").read_text() == (
        f'DatasetConfig(source="org/data::validation", name="data", revision="{SHA}")\n'
    )


class FakeHfApi:
    """Offline Hugging Face API replacement."""

    def __init__(self, datasets: dict[str, str] | None = None) -> None:
        self.datasets = datasets or {}
        self.lookups: list[str] = []

    def dataset_info(self, repo: str, **kwargs: object) -> SimpleNamespace:
        self.lookups.append(repo)
        if repo not in self.datasets:
            raise ValueError("missing")
        return SimpleNamespace(sha=self.datasets[repo])

    def list_datasets(self, **kwargs: object) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(id=repo, sha=sha)
            for repo, sha in self.datasets.items()
            if repo.startswith("EuroEval/")
        ]


def run_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: str,
    datasets: dict[str, str],
) -> FakeHfApi:
    """Run the public script interface with offline Hub data."""
    api = FakeHfApi(datasets=datasets)
    monkeypatch.setattr(refresh, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(refresh, "HfApi", lambda: api)
    (tmp_path / "config.py").write_text(source)
    refresh.main()
    return api


def test_malformed_config_fails_before_hub_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """All config syntax is validated before calling the Hub."""
    (tmp_path / "broken.py").write_text("DatasetConfig(\n")
    monkeypatch.setattr(refresh, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        refresh, "HfApi", lambda: pytest.fail("Hub must not be accessed")
    )
    with pytest.raises(SystemExit):
        refresh.main()


def test_missing_repo_leaves_all_files_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unresolved source aborts before any config is written."""
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text('DatasetConfig(source="org/known")\n')
    second.write_text('DatasetConfig(source="org/missing")\n')
    api = FakeHfApi(datasets={"org/known": SHA})
    monkeypatch.setattr(refresh, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(refresh, "HfApi", lambda: api)
    with pytest.raises(SystemExit):
        refresh.main()
    assert first.read_text() == 'DatasetConfig(source="org/known")\n'
    assert second.read_text() == 'DatasetConfig(source="org/missing")\n'


def test_repeat_run_is_noop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Running twice with the same revision leaves the file unchanged."""
    path = tmp_path / "config.py"
    text = f'DatasetConfig(source="org/data", revision="{SHA}")\n'
    path.write_text(text)
    run_main(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        source=text,
        datasets={"org/data": SHA},
    )
    assert path.read_text() == text


def test_updates_existing_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An outdated literal revision is replaced rather than duplicated."""
    run_main(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        source=f'DatasetConfig(source="org/data", revision="{SHA}")\n',
        datasets={"org/data": NEW_SHA},
    )
    assert (tmp_path / "config.py").read_text().count("revision=") == 1
    assert NEW_SHA in (tmp_path / "config.py").read_text()
