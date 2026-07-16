"""Tests for the mark_hf_models_open script."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    HFValidationError,
    RepositoryNotFoundError,
)

from src.scripts.mark_hf_models_open import (
    UploadError,
    canonical_model_id,
    check_hf_repo_exists,
    parse_jsonl_file_strict,
    process_jsonl_file,
    process_parsed_records,
    upload_changed_files,
    write_jsonl_file,
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
    """Tests for the upload_changed_files function using hf buckets cp CLI."""

    def test_dry_run_logs_without_upload(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Dry run should log what would be uploaded."""
        jsonl_file = tmp_path / "changed.jsonl"
        jsonl_file.write_text('{"test": true}\n')

        with caplog.at_level(logging.INFO):
            uploaded = upload_changed_files(
                changed_files=[jsonl_file], results_dir=tmp_path, dry_run=True
            )

        assert uploaded == 1
        assert "Would copy changed.jsonl" in caplog.text

    def test_uses_hf_buckets_cp_for_upload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use hf buckets cp CLI for uploading."""
        jsonl_file = tmp_path / "changed.jsonl"
        jsonl_file.write_text('{"test": true}\n')

        # Mock subprocess.run to avoid actual CLI call
        mock_run = MagicMock()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", return_value=mock_run.return_value) as mock_sub:
            uploaded = upload_changed_files(
                changed_files=[jsonl_file], results_dir=tmp_path, dry_run=False
            )

        assert uploaded == 1
        mock_sub.assert_called_once()
        call_args = mock_sub.call_args
        # Verify hf buckets cp command structure
        cmd = call_args.args[0]
        assert cmd[0] == "hf"
        assert cmd[1] == "buckets"
        assert cmd[2] == "cp"
        assert str(jsonl_file) in cmd
        assert "hf://buckets/EuroEval/results/changed.jsonl" in cmd

    def test_upload_handles_cli_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI upload failure should raise UploadError."""
        jsonl_file = tmp_path / "changed.jsonl"
        jsonl_file.write_text('{"test": true}\n')

        # Mock subprocess.run to simulate failure
        mock_run = MagicMock(
            side_effect=subprocess.CalledProcessError(
                1, ["hf"], stderr="Authentication failed"
            )
        )

        with patch("subprocess.run", side_effect=mock_run):
            with pytest.raises(UploadError) as exc_info:
                upload_changed_files(
                    changed_files=[jsonl_file], results_dir=tmp_path, dry_run=False
                )

        assert str(jsonl_file) in str(exc_info.value)

    def test_upload_handles_missing_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing hf CLI should raise UploadError."""
        jsonl_file = tmp_path / "changed.jsonl"
        jsonl_file.write_text('{"test": true}\n')

        with patch("subprocess.run", side_effect=FileNotFoundError("hf not found")):
            with pytest.raises(UploadError) as exc_info:
                upload_changed_files(
                    changed_files=[jsonl_file], results_dir=tmp_path, dry_run=False
                )

        assert "hf CLI not found" in str(exc_info.value)


class TestProcessJsonlFileStrictParsing:
    """Tests for strict JSONL parsing behaviour."""

    def test_invalid_json_raises_value_error(self, tmp_path: Path) -> None:
        """Should raise ValueError on invalid JSON and not rewrite file."""
        jsonl_file = tmp_path / "test.jsonl"
        # File with one valid and one invalid line
        original_content = '{"model_info": {"id": "org/repo"}}\ninvalid json here\n'
        jsonl_file.write_text(original_content)

        mock_api = MagicMock()
        existence_cache: dict[str, bool] = {}

        # Should raise ValueError due to invalid JSON
        with pytest.raises(ValueError):
            process_jsonl_file(
                jsonl_path=jsonl_file,
                hf_api=mock_api,
                existence_cache=existence_cache,
                dry_run=False,
            )

        # File should NOT be modified - original content preserved
        assert jsonl_file.read_text() == original_content

    def test_creates_additional_details_if_missing(self, tmp_path: Path) -> None:
        """Should create additional_details dict if it's missing."""
        jsonl_file = tmp_path / "test.jsonl"
        # Record with no additional_details key
        record = {"model_info": {"id": "meta-llama/Llama-2-7b", "name": "Llama 2"}}
        jsonl_file.write_text(json.dumps(record) + "\n")

        mock_api = MagicMock()
        mock_api.model_info.return_value = MagicMock()
        existence_cache: dict[str, bool] = {}

        checked, changed, _ = process_jsonl_file(
            jsonl_path=jsonl_file,
            hf_api=mock_api,
            existence_cache=existence_cache,
            dry_run=False,
        )

        assert checked == 1
        assert changed == 1

        # Verify additional_details was created and open=True was set
        updated = json.loads(jsonl_file.read_text().strip())
        assert "additional_details" in updated["model_info"]
        assert updated["model_info"]["additional_details"]["open"] is True

    def test_creates_additional_details_if_none(self, tmp_path: Path) -> None:
        """Should create additional_details dict if it's None."""
        jsonl_file = tmp_path / "test.jsonl"
        # Record with additional_details explicitly None
        record = {
            "model_info": {
                "id": "meta-llama/Llama-2-7b",
                "name": "Llama 2",
                "additional_details": None,
            }
        }
        jsonl_file.write_text(json.dumps(record) + "\n")

        mock_api = MagicMock()
        mock_api.model_info.return_value = MagicMock()
        existence_cache: dict[str, bool] = {}

        checked, changed, _ = process_jsonl_file(
            jsonl_path=jsonl_file,
            hf_api=mock_api,
            existence_cache=existence_cache,
            dry_run=False,
        )

        assert checked == 1
        assert changed == 1

        # Verify additional_details was created and open=True was set
        updated = json.loads(jsonl_file.read_text().strip())
        assert isinstance(updated["model_info"]["additional_details"], dict)
        assert updated["model_info"]["additional_details"]["open"] is True


class TestTransactionalProcessing:
    """Tests for transactional two-phase processing."""

    def test_parse_all_files_before_writing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should parse all files before writing any - invalid JSON aborts early."""
        # Create two files: one valid, one invalid
        valid_file = tmp_path / "valid.jsonl"
        invalid_file = tmp_path / "invalid.jsonl"

        valid_record = {
            "model_info": {
                "id": "meta-llama/Llama-2-7b",
                "name": "Llama 2",
                "additional_details": {"open": False},
            }
        }
        valid_file.write_text(json.dumps(valid_record) + "\n")
        invalid_file.write_text('{"invalid json}\n')

        # Parse valid file - should succeed
        records = parse_jsonl_file_strict(jsonl_path=valid_file)
        assert len(records) == 1

        # Parse invalid file - should raise ValueError
        with pytest.raises(ValueError):
            parse_jsonl_file_strict(jsonl_path=invalid_file)

        # Valid file should NOT be modified (no write happened)
        original_content = json.dumps(valid_record) + "\n"
        assert valid_file.read_text() == original_content

    def test_process_parsed_records_returns_modified_subset(
        self, tmp_path: Path
    ) -> None:
        """process_parsed_records should return only modified records."""
        records = [
            {
                "model_info": {
                    "id": "org/repo-exists",
                    "name": "Exists",
                    "additional_details": {"open": False},
                }
            },
            {
                "model_info": {
                    "id": "org/repo-missing",
                    "name": "Missing",
                    "additional_details": {"open": False},
                }
            },
            {
                "model_info": {
                    "id": "org/repo-open",
                    "name": "AlreadyOpen",
                    "additional_details": {"open": True},
                }
            },
        ]

        mock_api = MagicMock()

        # Mock: repo-exists exists, repo-missing doesn't
        def mock_model_info(repo_id: str, **kwargs: object) -> object:
            if "repo-exists" in repo_id:
                return MagicMock()
            exc = RepositoryNotFoundError.__new__(RepositoryNotFoundError)
            exc.args = ("Not found",)
            raise exc

        mock_api.model_info.side_effect = mock_model_info
        existence_cache: dict[str, bool] = {}

        modified, checked, changed, retries = process_parsed_records(
            records=records, hf_api=mock_api, existence_cache=existence_cache
        )

        assert checked == 3
        assert changed == 1
        assert len(modified) == 1
        # The modified record should be the first one (repo-exists)
        assert modified[0]["model_info"]["id"] == "org/repo-exists"

    def test_missing_repo_no_additional_details_created(self, tmp_path: Path) -> None:
        """Records with missing HF repos should NOT get additional_details: {} added."""
        jsonl_file = tmp_path / "test.jsonl"
        # Record with no additional_details at all
        record_no_ad = {"model_info": {"id": "fake/missing-repo", "name": "Missing"}}
        # Record with additional_details but no 'open' key
        record_ad_no_open = {
            "model_info": {
                "id": "fake/missing-repo-2",
                "name": "Missing2",
                "additional_details": {"other_key": "value"},
            }
        }
        jsonl_file.write_text(
            json.dumps(record_no_ad) + "\n" + json.dumps(record_ad_no_open) + "\n"
        )

        mock_api = MagicMock()
        # Both repos missing
        exc = RepositoryNotFoundError.__new__(RepositoryNotFoundError)
        exc.args = ("Not found",)
        mock_api.model_info.side_effect = exc
        existence_cache: dict[str, bool] = {}

        checked, changed, _ = process_jsonl_file(
            jsonl_path=jsonl_file,
            hf_api=mock_api,
            existence_cache=existence_cache,
            dry_run=False,
        )

        assert checked == 2
        assert changed == 0

        # Verify file was NOT modified - no additional_details added
        lines = jsonl_file.read_text().strip().splitlines()
        updated_no_ad = json.loads(lines[0])
        updated_ad_no_open = json.loads(lines[1])

        # Record with no additional_details should still have none
        assert "additional_details" not in updated_no_ad["model_info"]

        # Record with additional_details but no open should keep original
        assert updated_ad_no_open["model_info"]["additional_details"] == {
            "other_key": "value"
        }
        assert "open" not in updated_ad_no_open["model_info"]["additional_details"]


class TestUploadErrorHandling:
    """Tests for upload error handling - failures should be fatal."""

    def test_upload_failure_raises_exception(self, tmp_path: Path) -> None:
        """Upload failures should raise UploadError with failed file paths."""
        jsonl_file = tmp_path / "changed.jsonl"
        jsonl_file.write_text('{"test": true}\n')

        # Mock subprocess.run to simulate failure
        mock_run = MagicMock(
            side_effect=subprocess.CalledProcessError(
                1, ["hf"], stderr="Authentication failed"
            )
        )

        with patch("subprocess.run", side_effect=mock_run):
            with pytest.raises(UploadError) as exc_info:
                upload_changed_files(
                    changed_files=[jsonl_file], results_dir=tmp_path, dry_run=False
                )

        # Exception message should include failed file path
        assert str(jsonl_file) in str(exc_info.value)
        assert "Failed to upload" in str(exc_info.value)

    def test_upload_cli_missing_raises_exception(self, tmp_path: Path) -> None:
        """Missing hf CLI should raise UploadError."""
        jsonl_file = tmp_path / "changed.jsonl"
        jsonl_file.write_text('{"test": true}\n')

        with patch("subprocess.run", side_effect=FileNotFoundError("hf not found")):
            with pytest.raises(UploadError) as exc_info:
                upload_changed_files(
                    changed_files=[jsonl_file], results_dir=tmp_path, dry_run=False
                )

        assert "hf CLI not found" in str(exc_info.value)

    def test_multiple_upload_failures_reported(self, tmp_path: Path) -> None:
        """Multiple upload failures should all be reported in exception."""
        file1 = tmp_path / "file1.jsonl"
        file2 = tmp_path / "file2.jsonl"
        file1.write_text('{"test": 1}\n')
        file2.write_text('{"test": 2}\n')

        mock_run = MagicMock(
            side_effect=subprocess.CalledProcessError(1, ["hf"], stderr="Upload failed")
        )

        with patch("subprocess.run", side_effect=mock_run):
            with pytest.raises(UploadError) as exc_info:
                upload_changed_files(
                    changed_files=[file1, file2], results_dir=tmp_path, dry_run=False
                )

        # Both files should be mentioned in the error
        assert str(file1) in str(exc_info.value)
        assert str(file2) in str(exc_info.value)


class TestParseAndWriteFunctions:
    """Tests for the new parse_jsonl_file_strict and write_jsonl_file functions."""

    def test_parse_jsonl_file_strict_valid(self, tmp_path: Path) -> None:
        """parse_jsonl_file_strict should parse valid JSONL correctly."""
        jsonl_file = tmp_path / "test.jsonl"
        records = [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}]
        jsonl_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        parsed = parse_jsonl_file_strict(jsonl_path=jsonl_file)

        assert len(parsed) == 2
        assert parsed[0]["id"] == 1
        assert parsed[1]["id"] == 2

    def test_parse_jsonl_file_strict_invalid(self, tmp_path: Path) -> None:
        """parse_jsonl_file_strict should raise ValueError on invalid JSON."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"valid": true}\ninvalid json here\n')

        with pytest.raises(ValueError):
            parse_jsonl_file_strict(jsonl_path=jsonl_file)

    def test_write_jsonl_file(self, tmp_path: Path) -> None:
        """write_jsonl_file should write records with trailing newline."""
        jsonl_file = tmp_path / "output.jsonl"
        records = [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}]

        write_jsonl_file(jsonl_path=jsonl_file, records=records)

        lines = jsonl_file.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"id": 1, "name": "first"}
        assert json.loads(lines[1]) == {"id": 2, "name": "second"}
        # Verify trailing newline
        assert jsonl_file.read_text().endswith("\n")
