"""Tests for the mark_hf_models_open script."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    HFValidationError,
    RepositoryNotFoundError,
)

from src.scripts.mark_hf_models_open import (
    canonical_model_id,
    check_hf_repo_exists,
    process_jsonl_file,
    upload_changed_files,
)


class TestCanonicalModelId:
    """Tests for the canonical_model_id function."""

    def test_plain_model_id(self) -> None:
        """Should return plain org/repo for simple model IDs."""
        assert canonical_model_id("meta-llama/Llama-2-7b") == "meta-llama/Llama-2-7b"

    def test_strips_val_suffix(self) -> None:
        """Should strip (val) suffix."""
        result = canonical_model_id("meta-llama/Llama-2-7b (val)")
        assert result == "meta-llama/Llama-2-7b"

    def test_strips_zero_shot_suffix(self) -> None:
        """Should strip (zero-shot) suffix."""
        assert (
            canonical_model_id("meta-llama/Llama-2-7b (zero-shot)")
            == "meta-llama/Llama-2-7b"
        )

    def test_strips_combined_suffix(self) -> None:
        """Should strip combined (zero-shot, val) suffix."""
        assert (
            canonical_model_id("meta-llama/Llama-2-7b (zero-shot, val)")
            == "meta-llama/Llama-2-7b"
        )

    def test_strips_revision(self) -> None:
        """Should strip @revision suffix."""
        assert (
            canonical_model_id("meta-llama/Llama-2-7b@main") == "meta-llama/Llama-2-7b"
        )
        assert (
            canonical_model_id("meta-llama/Llama-2-7b@abc123")
            == "meta-llama/Llama-2-7b"
        )

    def test_strips_param_marker(self) -> None:
        """Should strip #param suffix."""
        assert (
            canonical_model_id("meta-llama/Llama-2-7b#no-thinking")
            == "meta-llama/Llama-2-7b"
        )
        assert (
            canonical_model_id("meta-llama/Llama-2-7b#thinking")
            == "meta-llama/Llama-2-7b"
        )

    def test_strips_anchor(self) -> None:
        """Should strip HTML anchor tags."""
        assert (
            canonical_model_id('<a href="#">meta-llama/Llama-2-7b</a>')
            == "meta-llama/Llama-2-7b"
        )

    def test_strips_all_combined(self) -> None:
        """Should strip all modifiers at once."""
        model_id = '<a href="#">meta-llama/Llama-2-7b (val)@main#thinking</a>'
        assert canonical_model_id(model_id) == "meta-llama/Llama-2-7b"


class TestCheckHfRepoExists:
    """Tests for the check_hf_repo_exists function with exponential backoff."""

    def test_repo_exists_returns_true(self) -> None:
        """Should return True when repo exists."""
        mock_api = MagicMock()
        mock_api.model_info.return_value = MagicMock()

        result = check_hf_repo_exists(hf_api=mock_api, repo_id="test/repo")

        assert result is True
        mock_api.model_info.assert_called_once_with(repo_id="test/repo")

    def test_gated_repo_returns_true(self) -> None:
        """Should return True for gated repos (they exist)."""
        mock_api = MagicMock()
        # Create exception without calling full constructor
        exc = GatedRepoError.__new__(GatedRepoError)
        exc.args = ("403 Client Error. You cannot access the gated repo.",)
        mock_api.model_info.side_effect = exc

        result = check_hf_repo_exists(hf_api=mock_api, repo_id="test/gated-repo")

        assert result is True

    def test_missing_repo_returns_false(self) -> None:
        """Should return False when repo doesn't exist."""
        mock_api = MagicMock()
        # Create exception without calling full constructor
        exc = RepositoryNotFoundError.__new__(RepositoryNotFoundError)
        exc.args = ("404 Client Error. Repository not found.",)
        mock_api.model_info.side_effect = exc

        result = check_hf_repo_exists(hf_api=mock_api, repo_id="test/missing")

        assert result is False

    def test_invalid_repo_returns_false(self) -> None:
        """Should return False for invalid repo IDs."""
        mock_api = MagicMock()
        mock_api.model_info.side_effect = HFValidationError("Invalid")

        result = check_hf_repo_exists(hf_api=mock_api, repo_id="invalid")

        assert result is False

    def test_transient_error_retries(self) -> None:
        """Should retry on transient HTTP errors."""
        mock_api = MagicMock()
        # Fail twice, then succeed
        # For unit tests, use simple exception with message
        mock_api.model_info.side_effect = [
            Exception("Rate limited"),
            Exception("Timeout"),
            MagicMock(),
        ]

        with patch("time.sleep"):  # Skip actual sleep in tests
            result = check_hf_repo_exists(
                hf_api=mock_api, repo_id="test/repo", base_delay=0.001
            )

        assert result is True
        assert mock_api.model_info.call_count == 3

    def test_max_retries_exceeded_returns_none(self) -> None:
        """Should return None after max retries exceeded."""
        mock_api = MagicMock()
        mock_api.model_info.side_effect = Exception("Always fails")

        with patch("time.sleep"):
            result = check_hf_repo_exists(
                hf_api=mock_api, repo_id="test/repo", max_retries=2, base_delay=0.001
            )

        assert result is None
        assert mock_api.model_info.call_count == 3  # initial + 2 retries

    def test_uses_exponential_backoff(self) -> None:
        """Should use exponential backoff between retries."""
        mock_api = MagicMock()
        mock_api.model_info.side_effect = Exception("Fails")

        sleep_times = []

        def capture_sleep(duration: float) -> None:
            sleep_times.append(duration)

        with patch("time.sleep", side_effect=capture_sleep):
            check_hf_repo_exists(
                hf_api=mock_api,
                repo_id="test/repo",
                max_retries=3,
                base_delay=1.0,
                max_delay=10.0,
            )

        # Should have exponential backoff: ~1, ~2, ~4 seconds (plus jitter)
        assert len(sleep_times) == 3
        assert sleep_times[0] < sleep_times[1] < sleep_times[2]

    def test_respects_max_delay(self) -> None:
        """Should not exceed max_delay."""
        mock_api = MagicMock()
        mock_api.model_info.side_effect = Exception("Fails")

        sleep_times = []

        def capture_sleep(duration: float) -> None:
            sleep_times.append(duration)

        with patch("time.sleep", side_effect=capture_sleep):
            check_hf_repo_exists(
                hf_api=mock_api,
                repo_id="test/repo",
                max_retries=10,
                base_delay=1.0,
                max_delay=5.0,
            )

        # All sleep times should be <= max_delay (plus small jitter)
        assert all(t <= 5.5 for t in sleep_times)


class TestProcessJsonlFile:
    """Tests for the process_jsonl_file function."""

    def test_marks_open_when_repo_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should mark records as open=True when HF repo exists."""
        # Create test JSONL file
        jsonl_file = tmp_path / "test.jsonl"
        record = {
            "model_info": {
                "id": "meta-llama/Llama-2-7b",
                "name": "Llama 2",
                "additional_details": {"open": False, "commercially_licensed": True},
            }
        }
        jsonl_file.write_text(json.dumps(record) + "\n")

        # Mock HF API to say repo exists
        mock_api = MagicMock()
        mock_api.model_info.return_value = MagicMock()

        existence_cache: dict[str, bool] = {}

        checked, changed, retries = process_jsonl_file(
            jsonl_path=jsonl_file,
            hf_api=mock_api,
            existence_cache=existence_cache,
            dry_run=False,
        )

        assert checked == 1
        assert changed == 1
        assert retries == 0

        # Verify file was updated
        updated = json.loads(jsonl_file.read_text().strip())
        assert updated["model_info"]["additional_details"]["open"] is True

    def test_skips_already_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should skip records already marked open=True."""
        jsonl_file = tmp_path / "test.jsonl"
        record = {
            "model_info": {
                "id": "meta-llama/Llama-2-7b",
                "name": "Llama 2",
                "additional_details": {"open": True},
            }
        }
        jsonl_file.write_text(json.dumps(record) + "\n")

        mock_api = MagicMock()
        existence_cache: dict[str, bool] = {}

        checked, changed, _ = process_jsonl_file(
            jsonl_path=jsonl_file, hf_api=mock_api, existence_cache=existence_cache
        )

        assert checked == 1
        assert changed == 0
        # Should not have called API since open=True
        mock_api.model_info.assert_not_called()

    def test_skips_missing_repo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should not mark open when HF repo doesn't exist."""
        jsonl_file = tmp_path / "test.jsonl"
        record = {
            "model_info": {
                "id": "fake/missing-repo",
                "name": "Missing",
                "additional_details": {"open": None},
            }
        }
        jsonl_file.write_text(json.dumps(record) + "\n")

        mock_api = MagicMock()
        # Create exception without calling full constructor
        exc = RepositoryNotFoundError.__new__(RepositoryNotFoundError)
        exc.args = ("Not found",)
        mock_api.model_info.side_effect = exc
        existence_cache: dict[str, bool] = {}

        checked, changed, _ = process_jsonl_file(
            jsonl_path=jsonl_file, hf_api=mock_api, existence_cache=existence_cache
        )

        assert checked == 1
        assert changed == 0

        # Verify file was NOT updated
        updated = json.loads(jsonl_file.read_text().strip())
        assert updated["model_info"]["additional_details"]["open"] is None

    def test_handles_concatenated_json_objects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should handle malformed JSONL with concatenated objects."""
        jsonl_file = tmp_path / "test.jsonl"
        # Two records on same line (malformed)
        record1 = {
            "model_info": {
                "id": "org/repo1",
                "name": "Repo1",
                "additional_details": {"open": False},
            }
        }
        record2 = {
            "model_info": {
                "id": "org/repo2",
                "name": "Repo2",
                "additional_details": {"open": None},
            }
        }
        # Malformed: concatenated without newline
        jsonl_file.write_text(json.dumps(record1) + json.dumps(record2) + "\n")

        mock_api = MagicMock()
        mock_api.model_info.return_value = MagicMock()
        existence_cache: dict[str, bool] = {}

        checked, changed, _ = process_jsonl_file(
            jsonl_path=jsonl_file, hf_api=mock_api, existence_cache=existence_cache
        )

        # Both records should be parsed
        assert checked == 2
        assert changed == 2

    def test_dry_run_no_modifications(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dry run should log without modifying files."""
        jsonl_file = tmp_path / "test.jsonl"
        original_content = json.dumps(
            {
                "model_info": {
                    "id": "meta-llama/Llama-2-7b",
                    "name": "Llama 2",
                    "additional_details": {"open": False},
                }
            }
        )
        jsonl_file.write_text(original_content + "\n")

        mock_api = MagicMock()
        mock_api.model_info.return_value = MagicMock()
        existence_cache: dict[str, bool] = {}

        checked, changed, _ = process_jsonl_file(
            jsonl_path=jsonl_file,
            hf_api=mock_api,
            existence_cache=existence_cache,
            dry_run=True,
        )

        assert checked == 1
        assert changed == 1

        # File should NOT be modified
        assert jsonl_file.read_text() == original_content + "\n"

    def test_uses_cache_for_repeated_repos(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should cache repo existence checks."""
        jsonl_file = tmp_path / "test.jsonl"
        # Multiple records with same model
        records = [
            {
                "model_info": {
                    "id": "meta-llama/Llama-2-7b",
                    "name": "Llama 2",
                    "additional_details": {"open": False},
                }
            },
            {
                "model_info": {
                    "id": "meta-llama/Llama-2-7b (val)",
                    "name": "Llama 2",
                    "additional_details": {"open": None},
                }
            },
        ]
        jsonl_file.write_text("".join(json.dumps(r) + "\n" for r in records))

        mock_api = MagicMock()
        mock_api.model_info.return_value = MagicMock()
        existence_cache: dict[str, bool] = {}

        process_jsonl_file(
            jsonl_path=jsonl_file, hf_api=mock_api, existence_cache=existence_cache
        )

        # Should only call API once for the same canonical ID
        assert mock_api.model_info.call_count == 1
        assert "meta-llama/Llama-2-7b" in existence_cache


class TestUploadChangedFiles:
    """Tests for the upload_changed_files function."""

    def test_dry_run_logs_without_upload(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Dry run should log what would be uploaded."""
        jsonl_file = tmp_path / "changed.jsonl"
        jsonl_file.write_text('{"test": true}\n')

        with caplog.at_level(logging.INFO):
            uploaded = upload_changed_files(changed_files=[jsonl_file], dry_run=True)

        assert uploaded == 1
        assert "Would upload changed.jsonl" in caplog.text

    def test_no_upload_without_token(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Should skip upload if no HF_TOKEN found."""
        jsonl_file = tmp_path / "changed.jsonl"
        jsonl_file.write_text('{"test": true}\n')

        # Ensure no token exists
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with patch("pathlib.Path.exists", return_value=False):
            with caplog.at_level(logging.WARNING):
                uploaded = upload_changed_files(
                    changed_files=[jsonl_file], dry_run=False
                )

        assert uploaded == 0
        assert "No HF_TOKEN found" in caplog.text
