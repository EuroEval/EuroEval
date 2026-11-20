"""Tests for the `leaderboards.result_processing` module."""

import json
import typing as t
from enum import Enum, auto
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from huggingface_hub import BucketFile
from huggingface_hub.errors import HfHubHTTPError

from leaderboards import bucket_sync, result_processing
from leaderboards.cache import Cache
from leaderboards.result_conflict_prompt import resolve_result_interactively
from leaderboards.result_conflicts import (
    ConflictChoice,
    ConflictDecision,
    resolve_result_conflict,
)
from leaderboards.result_identity import ResultIdentity, UnresolvedResultConflictError


class CallOrder(Enum):
    """Enum to track call order of functions."""

    NONE = auto()
    LOAD_RAW_RESULTS = auto()
    CACHE_FROM_RESULTS_DIR = auto()


class TestUploadPerModelFilesPerRecordTree:
    """Tests for the per-record JSON tree storage format."""

    def test_ambiguous_duplicate_aborts_before_writing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Different scores without chronology abort before local mutation.

        Args:
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
            tmp_path:
                Temporary directory for files created by the test.
        """
        monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)

        mock_api = MemoryBucketApi()

        # Older record (version 1.0.0)
        older_record: dict[str, t.Any] = {
            "model_info": {
                "id": "org/model",
                "name": "org/model",
                "additional_details": {"score": 50},
            },
            "dataset": {"name": "mmlu"},
            "task": {"name": "classification"},
            "eval_library": {
                "name": "euroeval",
                "version": "1.0.0",  # Older version
                "additional_details": {
                    "dataset": "mmlu",
                    "few_shot": False,
                    "validation_split": False,
                },
            },
            "results": {},
        }

        # Newer record (version 2.0.0)
        newer_record: dict[str, t.Any] = {
            "model_info": {
                "id": "org/model",
                "name": "org/model",
                "additional_details": {"score": 90},
            },
            "dataset": {"name": "mmlu"},
            "task": {"name": "classification"},
            "eval_library": {
                "name": "euroeval",
                "version": "2.0.0",  # Newer version
                "additional_details": {
                    "dataset": "mmlu",
                    "few_shot": False,
                    "validation_split": False,
                },
            },
            "results": {},
        }

        with patch("leaderboards.result_processing.HfApi", return_value=mock_api):
            with patch(
                "leaderboards.result_processing.resolve_hf_token",
                return_value="test_token",
            ):
                with pytest.raises(UnresolvedResultConflictError):
                    result_processing._upload_per_model_files(
                        processed_records=[older_record, newer_record],
                        upload_to_bucket=True,
                    )

        assert not list(tmp_path.rglob("*.json"))

    def test_invalid_model_record_dropped_with_log(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that records with unresolvable model identities are dropped.

        Args:
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
            tmp_path:
                Temporary directory for files created by the test.
            caplog:
                Fixture for capturing log records during the test.
        """
        monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)

        mock_api = MagicMock()

        # Invalid record (missing model_info.id and name)
        invalid_record: dict[str, t.Any] = {
            "model_info": {"additional_details": {}},
            "dataset": {"name": "mmlu"},
            "task": {"name": "classification"},
            "eval_library": {
                "name": "euroeval",
                "version": "1.0.0",
                "additional_details": {
                    "dataset": "mmlu",
                    "few_shot": False,
                    "validation_split": False,
                },
            },
            "results": {},
        }

        with patch("leaderboards.result_processing.HfApi", return_value=mock_api):
            with patch(
                "leaderboards.result_processing.resolve_hf_token",
                return_value="test_token",
            ):
                with caplog.at_level("WARNING"):
                    result_processing._upload_per_model_files(
                        processed_records=[invalid_record], upload_to_bucket=True
                    )

        # No files should be written
        assert list(tmp_path.iterdir()) == []

        # Should log a warning
        assert "unresolvable identity" in caplog.text.lower()

    def test_metadata_preserved_in_written_record(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test that all metadata fields are preserved in the written JSON.

        Args:
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
            tmp_path:
                Temporary directory for files created by the test.
        """
        monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)

        mock_api = MemoryBucketApi()
        record: dict[str, t.Any] = {
            "model_info": {
                "id": "org/model",
                "name": "org/model",
                "additional_details": {
                    "generative": True,
                    "generative_type": "chat",
                    "merge": False,
                    "open": True,
                    "commercially_licensed": False,
                    "trained_from_scratch": True,
                    "model_url": "https://huggingface.co/org/model",
                },
            },
            "dataset": {"name": "mmlu"},
            "task": {"name": "classification"},
            "eval_library": {
                "name": "euroeval",
                "version": "1.0.0",
                "additional_details": {
                    "dataset": "mmlu",
                    "few_shot": False,
                    "validation_split": False,
                },
            },
            "results": {},
        }

        with patch("leaderboards.result_processing.HfApi", return_value=mock_api):
            with patch(
                "leaderboards.result_processing.resolve_hf_token",
                return_value="test_token",
            ):
                result_processing._upload_per_model_files(
                    processed_records=[record], upload_to_bucket=True
                )

        model_dir = tmp_path / "org_model"
        result_file = model_dir / "mmlu__test__zeroshot.json"
        assert result_file.exists()

        content = result_file.read_text(encoding="utf-8")
        written_record = json.loads(content)

        # Verify metadata is preserved
        assert written_record["model_info"]["additional_details"]["generative"] is True
        assert (
            written_record["model_info"]["additional_details"]["generative_type"]
            == "chat"
        )
        assert written_record["model_info"]["additional_details"]["merge"] is False
        assert written_record["model_info"]["additional_details"]["open"] is True
        assert (
            written_record["model_info"]["additional_details"]["commercially_licensed"]
            is False
        )
        assert (
            written_record["model_info"]["additional_details"]["trained_from_scratch"]
            is True
        )

    def test_path_collision_raises_clear_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test that distinct identities mapping to same path raise clear error.

        Dataset names 'a/b' and 'a_b' both sanitise to 'a_b', causing a path
        collision. This should raise ValueError with a message naming the
        collision before any filesystem mutation occurs.

        Args:
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
            tmp_path:
                Temporary directory for files created by the test.
        """
        monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)

        mock_api = MagicMock()

        # Record with dataset 'a/b' -> sanitises to 'a_b'
        record_a: dict[str, t.Any] = {
            "model_info": {
                "id": "org/model",
                "name": "org/model",
                "additional_details": {"score": 50},
            },
            "dataset": {"name": "a/b"},
            "task": {"name": "classification"},
            "eval_library": {
                "name": "euroeval",
                "version": "1.0.0",
                "additional_details": {
                    "dataset": "a/b",
                    "few_shot": False,
                    "validation_split": False,
                },
            },
            "results": {},
        }

        # Record with dataset 'a_b' -> also sanitises to 'a_b'
        record_b: dict[str, t.Any] = {
            "model_info": {
                "id": "org/model",
                "name": "org/model",
                "additional_details": {"score": 90},
            },
            "dataset": {"name": "a_b"},
            "task": {"name": "classification"},
            "eval_library": {
                "name": "euroeval",
                "version": "1.0.0",
                "additional_details": {
                    "dataset": "a_b",
                    "few_shot": False,
                    "validation_split": False,
                },
            },
            "results": {},
        }

        with patch("leaderboards.result_processing.HfApi", return_value=mock_api):
            with patch(
                "leaderboards.result_processing.resolve_hf_token",
                return_value="test_token",
            ):
                with pytest.raises(ValueError, match="Identity collision detected"):
                    result_processing._upload_per_model_files(
                        processed_records=[record_a, record_b], upload_to_bucket=True
                    )

        # No files should be written on collision
        assert list(tmp_path.iterdir()) == []

    def test_process_results_reports_identity_normalised_conflict(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Catch distinct source splits collapsing after metadata repair.

        Args:
            monkeypatch: Fixture for replacing filesystem dependencies.
            tmp_path: Isolated result directory.
        """
        monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)
        records = [
            {
                "model_info": {"id": "org/model", "name": "org/model"},
                "dataset": {"name": "values"},
                "task": {"name": "european-values"},
                "eval_library": {
                    "name": "euroeval",
                    "version": "1.0.0",
                    "additional_details": {
                        "dataset": "values",
                        "task": "european-values",
                        "few_shot": False,
                        "validation_split": split,
                    },
                },
                "results": {"score": score},
            }
            for split, score in ((False, 0.5), (True, 0.9))
        ]
        load = MagicMock(return_value=records)
        load.cache_clear = MagicMock()
        monkeypatch.setattr(result_processing, "load_raw_results", load)
        monkeypatch.setattr(
            result_processing,
            "process_contamination_canaries",
            lambda records: (records, [], set(), {"status": "scored"}),
        )
        monkeypatch.setattr(Cache, "from_results_dir", lambda results_dir: Cache())
        monkeypatch.setattr(
            result_processing, "deduplicate_records", lambda records: records
        )
        monkeypatch.setattr(result_processing, "record_is_valid", lambda **kwargs: True)
        monkeypatch.setattr(
            result_processing, "add_missing_entries", lambda record, **kwargs: record
        )

        with pytest.raises(UnresolvedResultConflictError) as error:
            result_processing.process_results(
                min_version="0.0.0",
                min_number_of_model_records=0,
                banned_versions=[],
                banned_model_patterns=[],
                api_model_patterns=[],
                trained_from_scratch_patterns=[],
            )

        assert "org/model" in str(error.value)
        assert "results" in str(error.value)
        assert list(tmp_path.rglob("*.json")) == []

    @pytest.mark.parametrize(
        "scenario",
        [
            "keep_processed_winner",
            "keep_original",
            "new_identity",
            "richer_final",
            "richer_middle",
            "later_approval",
            "single_candidate",
            "single_candidate_stored",
            "stale_callback",
            "abort",
        ],
    )
    def test_three_contenders_resolve_against_processed_winner(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scenario: str
    ) -> None:
        """Keep the final selected record and bind only its exact approval.

        Args:
            monkeypatch: Fixture for replacing filesystem dependencies.
            tmp_path: Isolated result directory.
            scenario: Contender selection and canonical reconciliation case.
        """
        monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)
        records = [
            {
                "model_info": {"id": "org/model", "name": "org/model"},
                "dataset": {"name": "values"},
                "task": {"name": "classification"},
                "eval_library": {
                    "name": "euroeval",
                    "version": "1.0.0",
                    "additional_details": {
                        "dataset": "values",
                        "few_shot": False,
                        "validation_split": False,
                    },
                },
                "results": {"score": score},
            }
            for score in (0.1, 0.5, 0.9)
        ]
        choices = ["incoming", "stored", "incoming"]
        expected_pairs = [(0.1, 0.5), (0.5, 0.9), (0.1, 0.5)]
        expected_record = records[1]
        if scenario == "keep_original":
            choices[-1] = "stored"
            expected_record = records[0]
        elif scenario == "new_identity":
            choices.pop()
            expected_pairs.pop()
        elif scenario in ("richer_final", "richer_middle", "later_approval"):
            records[2]["results"] = {"score": 0.5}
            choices = ["incoming", "incoming"]
            expected_pairs = [(0.1, 0.5), (0.1, 0.5)]
            if scenario == "later_approval":
                choices[0] = "stored"
            else:
                rich = records[2] if scenario == "richer_final" else records[1]
                rich["model_info"] = {
                    "id": "org/model",
                    "name": "org/model",
                    "additional_details": {"open": True},
                }
                expected_record = rich
        elif scenario in (
            "single_candidate",
            "single_candidate_stored",
            "stale_callback",
        ):
            choices = [
                "stored" if scenario == "single_candidate_stored" else "incoming"
            ]
            expected_pairs = [(0.1, 0.5)]
            if scenario == "single_candidate_stored":
                expected_record = records[0]
        elif scenario == "abort":
            choices = ["incoming", "abort"]
            expected_pairs.pop()
        answers = iter(choices)
        result_file = tmp_path / "org_model" / "values__test__zeroshot.json"
        original_source = json.dumps(
            records[0],
            sort_keys=True,
            indent=2 if scenario not in ("richer_final", "later_approval") else None,
            separators=(",", ":"),
        ).encode()
        if scenario != "new_identity":
            result_file.parent.mkdir(parents=True)
            result_file.write_bytes(original_source)
        publisher = MagicMock()
        publisher.session.return_value.__enter__.return_value = publisher
        verified_payloads: list[bytes] = []

        def verify_publication(
            *,
            records: list[bytes],
            sources: dict[ResultIdentity, bytes],
            decisions: list[ConflictDecision],
        ) -> None:
            """Check final payload evidence and preflight-before-write ordering.

            Args:
                records: Final serialised publication candidates.
                sources: Captured canonical predecessor bytes.
                decisions: Exact incoming overwrite approvals.
            """
            assert len(records) == 1
            assert json.loads(records[0]) == expected_record
            if not verified_payloads:
                if scenario == "new_identity":
                    assert not result_file.exists()
                else:
                    assert result_file.read_bytes() == original_source
            if scenario in ("new_identity", "keep_original", "single_candidate_stored"):
                assert decisions == []
            else:
                assert len(decisions) == 1
                expected = resolve_result_conflict(
                    current=json.loads(original_source),
                    candidate=json.loads(records[0]),
                    current_bytes=original_source,
                    candidate_bytes=records[0],
                    choice=ConflictChoice.INCOMING,
                )
                assert decisions[0].selected is ConflictChoice.INCOMING
                assert decisions[0].evidence == expected.evidence
            assert list(sources.values()) == (
                [] if scenario == "new_identity" else [original_source]
            )
            verified_payloads.append(records[0])

        publisher.prepare.side_effect = verify_publication
        publisher.publish.side_effect = verify_publication
        monkeypatch.setattr(
            result_processing, "CanonicalPublisher", lambda **_: publisher
        )
        monkeypatch.setattr(result_processing, "resolve_hf_token", lambda: "test-token")
        monkeypatch.setattr(result_processing, "HfApi", MagicMock)
        monkeypatch.setattr(result_processing, "fix_metadata", lambda record: record)
        inputs = [records[1]] if scenario.startswith("single_candidate") else records
        load = MagicMock(return_value=inputs)
        load.cache_clear = MagicMock()
        monkeypatch.setattr(result_processing, "load_raw_results", load)
        monkeypatch.setattr(
            result_processing,
            "process_contamination_canaries",
            lambda records: (records, [], set(), {"status": "scored"}),
        )
        monkeypatch.setattr(Cache, "from_results_dir", lambda results_dir: Cache())
        monkeypatch.setattr(
            result_processing, "deduplicate_records", lambda records: records
        )
        monkeypatch.setattr(result_processing, "record_is_valid", lambda **kwargs: True)
        monkeypatch.setattr(
            result_processing, "add_missing_entries", lambda record, **kwargs: record
        )
        prompted: list[tuple[float, float]] = []

        def choose_incoming(
            *,
            current: dict[str, t.Any],
            candidate: dict[str, t.Any],
            current_bytes: bytes,
            candidate_bytes: bytes,
        ) -> ConflictDecision:
            """Apply one owner choice to the exact supplied contender pair.

            Args:
                current: Current selected record.
                candidate: Incoming contender.
                current_bytes: Exact current record bytes.
                candidate_bytes: Exact incoming record bytes.

            Returns:
                The explicit choice for the supplied pair.
            """
            prompted.append(
                (current["results"]["score"], candidate["results"]["score"])
            )
            answer = next(answers)
            if answer == "abort":
                return resolve_result_interactively(
                    current=current,
                    candidate=candidate,
                    current_bytes=current_bytes,
                    candidate_bytes=candidate_bytes,
                    input_fn=MagicMock(side_effect=EOFError),
                    interactive=True,
                )
            return resolve_result_conflict(
                current=current,
                candidate=candidate,
                current_bytes=(
                    b"unrelated current record"
                    if scenario == "stale_callback"
                    else current_bytes
                ),
                candidate_bytes=candidate_bytes,
                choice=ConflictChoice(answer),
            )

        run = partial(
            result_processing.process_results,
            min_version="0.0.0",
            min_number_of_model_records=0,
            banned_versions=[],
            banned_model_patterns=[],
            api_model_patterns=[],
            trained_from_scratch_patterns=[],
            upload_to_bucket=True,
            conflict_resolver=choose_incoming,
        )
        if scenario in ("abort", "stale_callback"):
            with pytest.raises(UnresolvedResultConflictError):
                run()
            assert prompted == expected_pairs
            assert result_file.read_bytes() == original_source
            publisher.prepare.assert_not_called()
            publisher.publish.assert_not_called()
            return
        leaderboard_records = run()
        assert prompted == expected_pairs
        written = json.loads(result_file.read_text(encoding="utf-8"))
        assert written["results"] == expected_record["results"]
        assert written["model_info"] == expected_record["model_info"]
        assert len(verified_payloads) == 2
        assert verified_payloads[0] == verified_payloads[1]
        assert json.loads(verified_payloads[0]) == written
        assert leaderboard_records == [written]

    def test_writes_expected_directory_tree(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test that _upload_per_model_files writes the expected directory structure.

        Args:
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
            tmp_path:
                Temporary directory for files created by the test.
        """
        monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)

        mock_api = MemoryBucketApi()
        record: dict[str, t.Any] = {
            "model_info": {
                "id": "org/Qwen3-0.6B",
                "name": "org/Qwen3-0.6B",
                "additional_details": {},
            },
            "dataset": {"name": "mmlu"},
            "task": {"name": "classification"},
            "eval_library": {
                "name": "euroeval",
                "version": "1.0.0",
                "additional_details": {
                    "dataset": "mmlu",
                    "few_shot": False,
                    "validation_split": False,
                },
            },
            "results": {},
        }

        with patch("leaderboards.result_processing.HfApi", return_value=mock_api):
            with patch(
                "leaderboards.result_processing.resolve_hf_token",
                return_value="test_token",
            ):
                result_processing._upload_per_model_files(
                    processed_records=[record], upload_to_bucket=True
                )

        # Check directory structure
        model_dir = tmp_path / "org_Qwen3-0.6B"
        assert model_dir.exists(), "Model directory should be created"

        # Check file exists with expected name
        result_file = model_dir / "mmlu__test__zeroshot.json"
        assert result_file.exists(), "Result file should be created"

        # Check content is valid JSON with metadata preserved
        content = result_file.read_text(encoding="utf-8")
        written_record = json.loads(content)
        assert written_record["model_info"]["id"] == "org/Qwen3-0.6B"
        assert "additional_details" in written_record["model_info"]


class MemoryBucketApi:
    """In-memory bucket adapter for publication and sync tests."""

    def __init__(self) -> None:
        """Initialise an empty bucket and call log."""
        self.objects: dict[str, bytes] = {}
        self.listed: list[tuple[str, bool, str]] = []
        self.downloaded: list[tuple[str, str]] = []
        self.uploaded: list[tuple[str, str]] = []
        self.fail_upload: BaseException | None = None

    def batch_bucket_files(
        self,
        bucket_id: str,
        add: list[tuple[Path, str]] | None = None,
        delete: list[str] | None = None,
        token: str = "",
    ) -> None:
        """Apply writes and deletes to the in-memory bucket."""
        if self.fail_upload is not None:
            raise self.fail_upload
        for local, name in add or []:
            self.uploaded.append((bucket_id, token))
            self.objects[name] = local.read_bytes()
        for name in delete or []:
            self.objects.pop(name, None)

    def download_bucket_files(
        self,
        bucket_id: str,
        files: list[tuple[str | BucketFile, str | Path]],
        *,
        raise_on_missing_files: bool = False,
        token: str = "",
    ) -> None:
        """Write requested bucket objects to local destinations."""
        del raise_on_missing_files
        for name, destination in files:
            path = name.path if isinstance(name, BucketFile) else name
            self.downloaded.append((bucket_id, token))
            if path not in self.objects:
                continue
            local = Path(destination)
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(self.objects[path])

    def list_bucket_tree(
        self, bucket_id: str, recursive: bool, token: str
    ) -> list[BucketFile]:
        """List bucket objects as Hugging Face bucket entries.

        Args:
            bucket_id: Bucket to list.
            recursive: Whether to list nested objects.
            token: Authentication token.

        Returns:
            Bucket entries for all stored objects.
        """
        self.listed.append((bucket_id, recursive, token))
        return [
            BucketFile(
                type="file", path=path, size=len(content), xetHash=f"hash-{path}"
            )
            for path, content in self.objects.items()
        ]


def test_cache_freshness_load_before_cache_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify that load_raw_results is called before Cache.from_results_dir.

    This ensures the results bucket is synced before cache construction,
    preventing stale cache issues. The test fails if cache construction
    happens before loading/syncing completes.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    call_sequence: list[CallOrder] = []

    def fake_load_raw_results() -> list[dict[str, t.Any]]:
        """Record the raw-results load in the call sequence.

        Returns:
            An empty result list.
        """
        call_sequence.append(CallOrder.LOAD_RAW_RESULTS)
        return []

    # Attach cache_clear to mimic functools.cached_function
    fake_load_raw_results.cache_clear = lambda: None  # type: ignore[attr-defined]

    def fake_cache_from_results_dir(results_dir: Path) -> Cache:
        """Fake cache from results dir for this test.

        Args:
            results_dir:
                Results dir supplied to the test.

        Returns:
            The value returned by the test helper.
        """
        call_sequence.append(CallOrder.CACHE_FROM_RESULTS_DIR)
        # Return a minimal Cache with empty dicts - the actual cache
        # is used for metadata lookups in add_missing_entries
        return Cache()

    monkeypatch.setattr(result_processing, "load_raw_results", fake_load_raw_results)
    monkeypatch.setattr(Cache, "from_results_dir", fake_cache_from_results_dir)
    monkeypatch.setattr(
        result_processing,
        "_upload_per_model_files",
        lambda processed_records, upload_to_bucket=False, local_only_label=None: (
            processed_records
        ),
    )

    result_processing.process_results(
        min_version="0.0.0",
        min_number_of_model_records=0,
        banned_versions=[],
        banned_model_patterns=[],
        api_model_patterns=[],
        trained_from_scratch_patterns=[],
    )

    assert len(call_sequence) == 2, (
        f"Expected 2 calls, got {len(call_sequence)}: {call_sequence}"
    )
    assert call_sequence[0] == CallOrder.LOAD_RAW_RESULTS, (
        f"load_raw_results must be called first, but got: {call_sequence[0]}"
    )
    assert call_sequence[1] == CallOrder.CACHE_FROM_RESULTS_DIR, (
        f"Cache.from_results_dir must be called second, but got: {call_sequence[1]}"
    )


def test_configure_canary_filter_when_processing_is_skipped() -> None:
    """Repeated generation still applies private canary exclusions."""
    records = [{"record": 1}]
    with (
        patch.object(result_processing, "reset_leaderboard_result_filter") as reset,
        patch.object(
            result_processing, "load_raw_results", return_value=records
        ) as load,
        patch.object(
            result_processing,
            "process_contamination_canaries",
            return_value=(records, [], {"org/model"}, {"status": "ok"}),
        ) as process,
        patch.object(
            result_processing, "configure_leaderboard_result_filter"
        ) as configure,
    ):
        result_processing.configure_canary_result_filter()

    reset.assert_called_once_with()
    load.assert_called_once_with()
    process.assert_called_once_with(records=records)
    configure.assert_called_once_with(excluded_models={"org/model"})
    load.cache_clear.assert_called_once_with()


def test_process_results_clears_cache_after_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that load_raw_results cache is cleared after processing.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    # Track whether cache_clear was called
    cache_clear_called = False

    def fake_cache_clear() -> None:
        """Record that the result cache was cleared."""
        nonlocal cache_clear_called
        cache_clear_called = True

    def fake_load_raw_results() -> list[dict[str, t.Any]]:
        """Return no raw results for this test.

        Returns:
            An empty list of raw results.
        """
        return []

    # Attach cache_clear to mimic functools.cached_function
    fake_load_raw_results.cache_clear = fake_cache_clear  # type: ignore[attr-defined]

    def fake_cache_from_results_dir(results_dir: Path) -> Cache:
        """Fake cache from results dir for this test.

        Args:
            results_dir:
                Results dir supplied to the test.

        Returns:
            The value returned by the test helper.
        """
        return Cache()

    monkeypatch.setattr(result_processing, "load_raw_results", fake_load_raw_results)
    monkeypatch.setattr(Cache, "from_results_dir", fake_cache_from_results_dir)
    monkeypatch.setattr(
        result_processing,
        "_upload_per_model_files",
        lambda processed_records, upload_to_bucket=False, local_only_label=None: (
            processed_records
        ),
    )

    result_processing.process_results(
        min_version="0.0.0",
        min_number_of_model_records=0,
        banned_versions=[],
        banned_model_patterns=[],
        api_model_patterns=[],
        trained_from_scratch_patterns=[],
    )

    assert cache_clear_called, (
        "load_raw_results.cache_clear() should be called after upload"
    )


def test_process_results_keeps_canary_records_out_of_public_uploads() -> None:
    """Only ordinary records are eligible for canonical bucket upload."""
    canary = {"private": "evidence"}
    load = MagicMock(return_value=[])
    load.cache_clear = MagicMock()
    with (
        patch.object(result_processing, "load_raw_results", load),
        patch.object(
            result_processing,
            "process_contamination_canaries",
            return_value=([], [canary], set(), {"status": "scored"}),
        ),
        patch.object(Cache, "from_results_dir", return_value=Cache()),
        patch.object(result_processing, "_upload_per_model_files") as upload,
    ):
        upload.return_value = []
        result_processing.process_results(
            min_version="0.0.0",
            min_number_of_model_records=0,
            banned_versions=[],
            banned_model_patterns=[],
            api_model_patterns=[],
            trained_from_scratch_patterns=[],
            upload_to_bucket=True,
        )

    assert upload.call_args_list == [
        call(processed_records=[], upload_to_bucket=True),
        call(
            processed_records=[canary],
            upload_to_bucket=False,
            local_only_label="private contamination-canary",
        ),
    ]


def test_process_results_returns_leaderboard_records_without_canary_exclusions() -> (
    None
):
    """Return only processed records eligible for public leaderboards."""
    records = [{"id": "kept"}, {"id": "excluded"}]
    load = MagicMock(return_value=records)
    load.cache_clear = MagicMock()
    uploaded: list[list[dict[str, str]]] = []
    with (
        patch.object(result_processing, "load_raw_results", load),
        patch.object(
            result_processing,
            "process_contamination_canaries",
            return_value=(records, [], {"excluded"}, {"status": "scored"}),
        ),
        patch.object(Cache, "from_results_dir", return_value=Cache()),
        patch.object(
            result_processing,
            "deduplicate_records",
            side_effect=lambda records: records,
        ),
        patch.object(
            result_processing, "fix_metadata", side_effect=lambda record: record
        ),
        patch.object(result_processing, "record_is_valid", return_value=True),
        patch.object(
            result_processing, "get_model_name", side_effect=lambda record: record["id"]
        ),
        patch.object(
            result_processing,
            "identity_from_eee_record",
            side_effect=lambda record: (record["id"], "dataset", False, False),
        ),
        patch.object(
            result_processing,
            "add_missing_entries",
            side_effect=lambda record, **kwargs: record,
        ),
        patch.object(
            result_processing,
            "_upload_per_model_files",
            side_effect=lambda processed_records, **kwargs: (
                uploaded.append(processed_records) or processed_records
            ),
        ),
    ):
        processed = result_processing.process_results(
            min_version="0.0.0",
            min_number_of_model_records=0,
            banned_versions=[],
            banned_model_patterns=[],
            api_model_patterns=[],
            trained_from_scratch_patterns=[],
            upload_to_bucket=True,
        )

    assert processed == [{"id": "kept"}]
    assert uploaded[0] == records


def test_process_results_returns_the_written_conflict_winner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Leaderboard input uses the same payload selected and written by storage."""
    monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)
    records: list[dict[str, t.Any]] = []
    for score in (0.2, 0.8):
        records.append(
            {
                "model_info": {"id": "org/model", "name": "org/model"},
                "eval_library": {
                    "name": "euroeval",
                    "version": "1.0.0",
                    "additional_details": {
                        "dataset": "dataset",
                        "few_shot": False,
                        "validation_split": False,
                    },
                },
                "evaluation_results": [{"score": score}],
            }
        )
    load = MagicMock(return_value=records)
    load.cache_clear = MagicMock()

    def incoming_choice(
        *,
        current: dict[str, t.Any],
        candidate: dict[str, t.Any],
        current_bytes: bytes | None = None,
        candidate_bytes: bytes | None = None,
    ) -> ConflictDecision:
        return resolve_result_conflict(
            current=current,
            candidate=candidate,
            current_bytes=current_bytes,
            candidate_bytes=candidate_bytes,
            choice=ConflictChoice.INCOMING,
        )

    with (
        patch.object(result_processing, "load_raw_results", load),
        patch.object(
            result_processing,
            "process_contamination_canaries",
            return_value=(records, [], set(), {"status": "scored"}),
        ),
        patch.object(Cache, "from_results_dir", return_value=Cache()),
        patch.object(
            result_processing,
            "deduplicate_records",
            side_effect=lambda records: records,
        ),
        patch.object(
            result_processing, "fix_metadata", side_effect=lambda record: record
        ),
        patch.object(result_processing, "record_is_valid", return_value=True),
        patch.object(
            result_processing,
            "add_missing_entries",
            side_effect=lambda record, **kwargs: record,
        ),
    ):
        leaderboard_records = result_processing.process_results(
            min_version="0.0.0",
            min_number_of_model_records=0,
            banned_versions=[],
            banned_model_patterns=[],
            api_model_patterns=[],
            trained_from_scratch_patterns=[],
            conflict_resolver=incoming_choice,
        )

    written_record = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert leaderboard_records == [written_record]
    assert leaderboard_records[0]["evaluation_results"] == [{"score": 0.8}]


def test_sync_bucket_passes_hf_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that sync_bucket passes the resolved HF token.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    monkeypatch.setattr(bucket_sync, "RESULTS_DIR", tmp_path)
    mock_api = MemoryBucketApi()
    raw_record = json.dumps(
        {
            "schema_version": "0.3.0",
            "model_info": {"id": "org/model"},
            "eval_library": {
                "name": "euroeval",
                "version": "18.4.0",
                "additional_details": {
                    "dataset": "dataset",
                    "validation_split": False,
                    "few_shot": False,
                },
            },
            "evaluation_results": [],
        }
    ).encode()
    mock_api.objects["org_model/dataset__test__zeroshot.json"] = raw_record

    with patch("leaderboards.bucket_sync.HfApi", return_value=mock_api):
        with patch(
            "leaderboards.bucket_sync.resolve_hf_token", return_value="test_token"
        ):
            bucket_sync.sync_bucket(delete_empty=False)

    assert mock_api.listed == [("EuroEval/results", True, "test_token")]
    assert mock_api.downloaded == [("EuroEval/results", "test_token")]
    assert (
        tmp_path / "org_model/dataset__test__zeroshot.json"
    ).read_bytes() == raw_record


def test_sync_bucket_raises_without_hf_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that sync_bucket raises without HF_TOKEN set.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    monkeypatch.setattr(bucket_sync, "RESULTS_DIR", tmp_path)

    with patch("leaderboards.bucket_sync.resolve_hf_token", return_value=None):
        with pytest.raises(RuntimeError, match="HF_TOKEN not set"):
            bucket_sync.sync_bucket()


def test_upload_per_model_files_passes_hf_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that _upload_per_model_files passes the resolved HF token.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)

    mock_api = MemoryBucketApi()
    record: dict[str, t.Any] = {
        "model_info": {
            "id": "org/model",
            "name": "org/model",
            "additional_details": {},
        },
        "dataset": {"name": "dataset"},
        "task": {"name": "classification"},
        "eval_library": {
            "name": "euroeval",
            "version": "1.0.0",
            "additional_details": {
                "dataset": "dataset",
                "few_shot": False,
                "validation_split": False,
            },
        },
        "results": {},
    }

    with patch("leaderboards.result_processing.HfApi", return_value=mock_api):
        with patch(
            "leaderboards.result_processing.resolve_hf_token", return_value="test_token"
        ):
            result_processing._upload_per_model_files(
                processed_records=[record], upload_to_bucket=True
            )

    assert mock_api.uploaded == [("EuroEval/results", "test_token")]


def test_upload_per_model_files_raises_on_sync_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that _upload_per_model_files raises when bucket sync fails.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)

    mock_api = MemoryBucketApi()
    mock_api.fail_upload = HfHubHTTPError(
        "Bucket sync failed", response=MagicMock(status_code=500)
    )

    with patch("leaderboards.result_processing.HfApi", return_value=mock_api):
        with patch(
            "leaderboards.result_processing.resolve_hf_token", return_value="test_token"
        ):
            with pytest.raises(HfHubHTTPError, match="Bucket sync failed"):
                result_processing._upload_per_model_files(
                    processed_records=[
                        {
                            "model_info": {"id": "org/model"},
                            "eval_library": {
                                "additional_details": {"dataset": "dataset"}
                            },
                        }
                    ],
                    upload_to_bucket=True,
                )


def test_upload_per_model_files_raises_without_hf_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that _upload_per_model_files raises without HF_TOKEN set.

    RuntimeError is raised when HF_TOKEN is missing.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    # Patch RESULTS_DIR to use a temp directory
    monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)

    # Patch resolve_hf_token to return None (simulating missing HF_TOKEN)
    with patch("leaderboards.result_processing.resolve_hf_token", return_value=None):
        with pytest.raises(RuntimeError, match="HF_TOKEN not set"):
            result_processing._upload_per_model_files(
                processed_records=[], upload_to_bucket=True
            )


def test_upload_results_to_bucket_passes_hf_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that upload_results_to_bucket passes the resolved HF token.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    results_dir = tmp_path / "results"
    results_file = tmp_path / "results.jsonl"

    # Write valid record in EEE format
    valid_record = {
        "model_info": {"id": "org/model"},
        "eval_library": {
            "additional_details": {"dataset": "test_ds"},
            "version": "1.0.0",
        },
        "retrieved_timestamp": "2024-01-01T00:00:00Z",
    }
    results_file.write_text(json.dumps(valid_record) + "\n", encoding="utf-8")
    monkeypatch.setattr(bucket_sync, "RESULTS_DIR", results_dir)

    mock_api = MemoryBucketApi()

    with patch("leaderboards.bucket_sync.HfApi", return_value=mock_api):
        with patch(
            "leaderboards.bucket_sync.resolve_hf_token", return_value="test_token"
        ):
            bucket_sync.upload_results_to_bucket(results_file=results_file)

    assert mock_api.uploaded == [("EuroEval/results", "test_token")]


def test_upload_results_to_bucket_raises_on_sync_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that upload_results_to_bucket raises when bucket sync fails.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    results_dir = tmp_path / "results"
    results_file = tmp_path / "results.jsonl"

    # Write valid record in EEE format
    valid_record = {
        "model_info": {"id": "org/model"},
        "eval_library": {
            "additional_details": {"dataset": "test_ds"},
            "version": "1.0.0",
        },
        "retrieved_timestamp": "2024-01-01T00:00:00Z",
    }
    results_file.write_text(json.dumps(valid_record) + "\n", encoding="utf-8")
    monkeypatch.setattr(bucket_sync, "RESULTS_DIR", results_dir)

    mock_api = MemoryBucketApi()
    mock_api.fail_upload = HfHubHTTPError(
        "Bucket sync failed", response=MagicMock(status_code=500)
    )

    with patch("leaderboards.bucket_sync.HfApi", return_value=mock_api):
        with patch(
            "leaderboards.bucket_sync.resolve_hf_token", return_value="test_token"
        ):
            with pytest.raises(HfHubHTTPError, match="Bucket sync failed"):
                bucket_sync.upload_results_to_bucket(results_file=results_file)


def test_upload_results_to_bucket_raises_without_hf_token(tmp_path: Path) -> None:
    """Test that upload_results_to_bucket raises without HF_TOKEN set.

    Args:
        tmp_path:
            Temporary directory for files created by the test.
    """
    results_file = tmp_path / "results.jsonl"
    results_file.write_text("{}\n", encoding="utf-8")

    with patch("leaderboards.bucket_sync.resolve_hf_token", return_value=None):
        with pytest.raises(RuntimeError, match="HF_TOKEN not set"):
            bucket_sync.upload_results_to_bucket(results_file=results_file)
