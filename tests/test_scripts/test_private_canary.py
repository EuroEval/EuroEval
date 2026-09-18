"""Focused tests for the private contamination-canary prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import typing as t
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

import euroeval.private_canary as private_canary
from euroeval.private_canary import (
    CANARY_ROW_COUNT,
    EXPOSURE_LEVELS,
    CanaryRecord,
    CanarySchedule,
    analyse_canary_results,
    build_canary_schedule,
    generate_canary_corpus,
    load_key_0600,
    score_association,
    target_continuation,
    validate_arm_payload,
)
from src.scripts.canary import run_private_canary_exposure as canary_runner
from src.scripts.canary.run_private_canary_exposure import (
    _fingerprint,
    _read_records,
    _read_rows,
    _static_identity,
    _train,
    _training_device,
    _validate_existing,
    run_private_study,
)


class WordTokenizer:
    """Small deterministic tokenizer with stable word boundaries."""

    all_special_ids = (0,)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Encode words and punctuation as stable token IDs.

        Returns:
            Stable integer token IDs.
        """
        del add_special_tokens
        values = text.replace(".", " .").split()
        return [
            int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")
            for value in values
        ]


class UnstableTriggerTokenizer(WordTokenizer):
    """Tokenizer that rejects one trigger boundary while preserving all others."""

    def __init__(self, unstable_trigger: str) -> None:
        """Initialise the tokenizer with the trigger to destabilise.

        Args:
            unstable_trigger:
              Trigger whose boundary should fail validation.
        """
        self.unstable_trigger = unstable_trigger

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Encode text with an unstable prefix for one trigger candidate.

        Returns:
            Token IDs with a deliberately broken trigger prefix when applicable.
        """
        before = private_canary.TEMPLATE_PREFIX.split("{trigger}", maxsplit=1)[0]
        encoded = super().encode(text, add_special_tokens=add_special_tokens)
        if text == before + self.unstable_trigger:
            return [encoded[0] + 1, *encoded[1:]]
        return encoded


class AlwaysUnstableTriggerTokenizer(WordTokenizer):
    """Tokenizer that makes every trigger boundary unstable."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Encode text with a deliberately broken trigger prefix.

        Returns:
            Token IDs with a broken prefix for trigger validation inputs.
        """
        before = private_canary.TEMPLATE_PREFIX.split("{trigger}", maxsplit=1)[0]
        encoded = super().encode(text, add_special_tokens=add_special_tokens)
        if text.startswith(before) and text != before:
            return [encoded[0] + 1, *encoded[1:]]
        return encoded


def test_leading_space_belongs_to_target_continuation() -> None:
    """A target separator is scored with the target, not as part of the prompt."""
    prompt = "In the quiet archive, the note marked amber referred to"
    continuation = target_continuation(target="ancient anchor")
    assert not prompt.endswith(" ")
    assert continuation == " ancient anchor"
    target_ids = private_canary._validated_target_ids(
        tokenizer=WordTokenizer(), prompt=prompt, target="ancient anchor"
    )
    assert len(target_ids) == 2


def test_pinned_tokenizer_has_stable_canary_candidates() -> None:
    """The pinned tokenizer has enough real two-token canary associations."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            private_canary.MODEL_ID,
            revision=private_canary.MODEL_REVISION,
            local_files_only=True,
        )
    except (OSError, ValueError, RuntimeError) as error:
        pytest.skip(f"pinned tokenizer is not cached locally: {error}")
    stable_triggers = []
    for trigger in private_canary.TRIGGERS:
        try:
            private_canary._validated_trigger(tokenizer=tokenizer, trigger=trigger)
        except ValueError:
            continue
        stable_triggers.append(trigger)
    assert len(stable_triggers) == len(private_canary.TRIGGERS) == 32
    prompt = private_canary.TEMPLATE_PREFIX.format(trigger=stable_triggers[0])
    candidates = [
        f"{first} {second}"
        for first in private_canary.TARGET_FIRST
        for second in private_canary.TARGET_SECOND
    ]
    valid_targets = [
        target
        for target in candidates
        if _valid_target(tokenizer=tokenizer, prompt=prompt, target=target)
    ]
    assert len(valid_targets) >= 2 * CANARY_ROW_COUNT


def _valid_target(*, tokenizer: object, prompt: str, target: str) -> bool:
    try:
        private_canary._validated_target_ids(
            tokenizer=tokenizer, prompt=prompt, target=target
        )
    except ValueError:
        return False
    return True


class PredictingModel:
    """Toy causal model used to check all three scoring metrics."""

    def __call__(self, *, input_ids: torch.Tensor) -> SimpleNamespace:
        """Return logits favouring the two-token target."""
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 8)
        logits[:, :, 1] = 2.0
        if int(input_ids[0, -1]) == 1:
            logits[:, -1, 2] = 4.0
        return SimpleNamespace(logits=logits)


def _key(path: Path, value: bytes = b"a" * 32) -> None:
    path.write_bytes(value)
    os.chmod(path, 0o600)


def _source(path: Path) -> None:
    path.write_text(
        "\n".join(
            json.dumps({"row_id": f"row-{index}", "text": "A clean source sentence."})
            for index in range(CANARY_ROW_COUNT)
        )
        + "\n",
        encoding="utf-8",
    )


def test_generation_is_key_sensitive_private_and_leak_free(tmp_path: Path) -> None:
    """Generation is deterministic, key-sensitive and restrictive on disk."""
    source = tmp_path / "source.jsonl"
    key = tmp_path / "key"
    _source(source)
    _key(key)
    first_augmented = tmp_path / "augmented-a"
    first_private = tmp_path / "private-a"
    generate_canary_corpus(
        corpus_jsonl=source,
        key_path=key,
        augmented_dir=first_augmented,
        private_dir=first_private,
        tokenizer=WordTokenizer(),
    )
    first_records = first_private.joinpath("canary-records.jsonl").read_text()
    repeat_private = tmp_path / "private-repeat"
    generate_canary_corpus(
        corpus_jsonl=source,
        key_path=key,
        augmented_dir=tmp_path / "augmented-repeat",
        private_dir=repeat_private,
        tokenizer=WordTokenizer(),
    )
    assert repeat_private.joinpath("canary-records.jsonl").read_text() == first_records
    second_key = tmp_path / "key-b"
    _key(second_key, b"b" * 32)
    second_private = tmp_path / "private-b"
    generate_canary_corpus(
        corpus_jsonl=source,
        key_path=second_key,
        augmented_dir=tmp_path / "augmented-b",
        private_dir=second_private,
        tokenizer=WordTokenizer(),
    )
    second_records = second_private.joinpath("canary-records.jsonl").read_text()
    assert first_records != second_records
    assert stat.S_IMODE(first_private.stat().st_mode) == 0o700
    assert (
        stat.S_IMODE(first_private.joinpath("canary-records.jsonl").stat().st_mode)
        == 0o600
    )
    augmented = first_augmented.joinpath("augmented.jsonl").read_text()
    for line in first_records.splitlines():
        record = json.loads(line)
        assert record["control_text"] not in augmented
        assert record["control_target"] not in augmented
        assert record["exposed_text"] in augmented
        assert record["exposed_prompt"] == record["control_prompt"]
        assert record["exposed_prompt"].startswith("A clean source sentence.\n")
        assert len(record["exposed_target_ids"]) == 2
        assert len(record["control_target_ids"]) == 2


def test_generation_rejects_unstable_trigger(tmp_path: Path) -> None:
    """Generation fails closed when one of the 32 triggers is unstable."""
    source = tmp_path / "source.jsonl"
    key = tmp_path / "key"
    _source(source)
    _key(key)
    first_index = private_canary._keyed_index(
        key=b"a" * 32,
        domain=b"trigger",
        value="row-0",
        size=len(private_canary.TRIGGERS),
    )
    unstable_trigger = private_canary.TRIGGERS[first_index]
    with pytest.raises(ValueError, match="32 unique validated triggers"):
        generate_canary_corpus(
            corpus_jsonl=source,
            key_path=key,
            augmented_dir=tmp_path / "augmented",
            private_dir=tmp_path / "private",
            tokenizer=UnstableTriggerTokenizer(unstable_trigger),
        )


def test_manifest_rejects_reused_trigger(tmp_path: Path) -> None:
    """A manifest must retain the complete trigger allocation."""
    source = tmp_path / "source.jsonl"
    key = tmp_path / "key"
    _source(source)
    _key(key)
    private = tmp_path / "private"
    augmented = tmp_path / "augmented"
    generate_canary_corpus(
        corpus_jsonl=source,
        key_path=key,
        augmented_dir=augmented,
        private_dir=private,
        tokenizer=WordTokenizer(),
    )
    manifest = json.loads((private / "canary-manifest.json").read_text())
    manifest["triggers"][0] = manifest["triggers"][1]
    with pytest.raises(ValueError, match="trigger allocation"):
        canary_runner._validate_inputs(
            manifest=manifest,
            records=canary_runner._read_records(private / "canary-records.jsonl"),
            source_rows=canary_runner._read_rows(source),
            augmented_rows=canary_runner._read_rows(augmented / "augmented.jsonl"),
            corpus_path=source,
            key_path=key,
        )


def test_failed_generation_writes_no_partial_corpus(tmp_path: Path) -> None:
    """A failed selection does not leave output directories or partial records."""
    source = tmp_path / "source.jsonl"
    key = tmp_path / "key"
    _source(source)
    _key(key)
    augmented = tmp_path / "augmented"
    private = tmp_path / "private"

    with pytest.raises(ValueError, match="32 unique validated triggers"):
        generate_canary_corpus(
            corpus_jsonl=source,
            key_path=key,
            augmented_dir=augmented,
            private_dir=private,
            tokenizer=AlwaysUnstableTriggerTokenizer(),
        )

    assert not augmented.exists()
    assert not private.exists()


def test_failed_output_publish_writes_no_partial_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key change during output creation leaves no published output."""
    source = tmp_path / "source.jsonl"
    key = tmp_path / "key"
    _source(source)
    _key(key)
    augmented = tmp_path / "augmented"
    private = tmp_path / "private"
    original_write = private_canary._atomic_write

    def change_key_after_write(*, path: Path, payload: bytes) -> None:
        original_write(path=path, payload=payload)
        key.write_bytes(b"b" * 32)

    monkeypatch.setattr(private_canary, "_atomic_write", change_key_after_write)
    with pytest.raises(ValueError, match="key changed"):
        generate_canary_corpus(
            corpus_jsonl=source,
            key_path=key,
            augmented_dir=augmented,
            private_dir=private,
            tokenizer=WordTokenizer(),
        )
    assert not augmented.exists()
    assert not private.exists()


def test_generation_skips_target_found_in_another_source(tmp_path: Path) -> None:
    """Allocation checks a candidate against every source before accepting it."""
    candidates = [
        f"{first} {second}"
        for first in private_canary.TARGET_FIRST
        for second in private_canary.TARGET_SECOND
    ]
    blocked_target = candidates[
        private_canary._keyed_index(
            key=b"a" * 32, domain=b"exposed-target", value="row-0", size=len(candidates)
        )
    ]
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "row_id": f"row-{index}",
                    "text": (
                        f"Another source contains {blocked_target}."
                        if index == 1
                        else "A clean source sentence."
                    ),
                }
            )
            + "\n"
            for index in range(CANARY_ROW_COUNT)
        ),
        encoding="utf-8",
    )
    key = tmp_path / "key"
    _key(key)
    private = tmp_path / "private"

    generate_canary_corpus(
        corpus_jsonl=source,
        key_path=key,
        augmented_dir=tmp_path / "augmented",
        private_dir=private,
        tokenizer=WordTokenizer(),
    )

    records = [
        json.loads(line)
        for line in private.joinpath("canary-records.jsonl").read_text().splitlines()
    ]
    assert blocked_target not in {
        target
        for record in records
        for target in (record["exposed_target"], record["control_target"])
    }


def test_generation_has_256_unique_prompt_units(tmp_path: Path) -> None:
    """Large studies deterministically produce independent prompt units."""
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps({"row_id": str(index), "text": "A clean source sentence."})
            + "\n"
            for index in range(256)
        ),
        encoding="utf-8",
    )
    key = tmp_path / "key"
    _key(key)
    private = tmp_path / "private"
    generate_canary_corpus(
        corpus_jsonl=source,
        key_path=key,
        augmented_dir=tmp_path / "augmented",
        private_dir=private,
        tokenizer=WordTokenizer(),
    )
    records = [
        json.loads(line)
        for line in private.joinpath("canary-records.jsonl").read_text().splitlines()
    ]
    assert len(records) == 256
    assert len({record["exposed_prompt"] for record in records}) == 256
    assert len({record["control_target"] for record in records}) == 32
    assert len({record["exposed_target"] for record in records}) == 32
    assert len({record["group_id"] for record in records}) == 32
    assert {record["trigger"] for record in records} == set(private_canary.TRIGGERS)
    assert all(
        sum(record["group_id"] == group_id for record in records) == 8
        for group_id in {record["group_id"] for record in records}
    )
    assert not {record["exposed_target"] for record in records} & {
        record["control_target"] for record in records
    }
    all_target_ids = [
        tuple(record[field])
        for record in records
        for field in ("exposed_target_ids", "control_target_ids")
    ]
    assert len(set(all_target_ids)) == 2 * 32

    repeat_private = tmp_path / "private-repeat-256"
    generate_canary_corpus(
        corpus_jsonl=source,
        key_path=key,
        augmented_dir=tmp_path / "augmented-repeat-256",
        private_dir=repeat_private,
        tokenizer=WordTokenizer(),
    )
    assert (
        repeat_private.joinpath("canary-records.jsonl").read_text()
        == private.joinpath("canary-records.jsonl").read_text()
    )


def test_generation_rejects_one_source_row(tmp_path: Path) -> None:
    """Generation rejects a source corpus that is short by one row."""
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"row_id": "only", "text": "A clean source sentence."}) + "\n",
        encoding="utf-8",
    )
    key = tmp_path / "key"
    _key(key)
    with pytest.raises(ValueError, match="exactly 256"):
        generate_canary_corpus(
            corpus_jsonl=source,
            key_path=key,
            augmented_dir=tmp_path / "augmented",
            private_dir=tmp_path / "private",
            tokenizer=WordTokenizer(),
        )


def test_schedule_has_equal_token_parity_and_nested_doses() -> None:
    """All arms have equal slots and nested exposed examples."""
    row_ids = [str(index) for index in range(CANARY_ROW_COUNT)]
    augmented_token_ids = {
        row_id: [1, 2, 3] if index % 2 == 0 else [4, 5]
        for index, row_id in enumerate(row_ids)
    }
    clean_token_ids = {
        row_id: [9] if index % 2 == 0 else [8, 7]
        for index, row_id in enumerate(row_ids)
    }
    schedule = build_canary_schedule(
        row_ids=row_ids,
        augmented_token_ids=augmented_token_ids,
        clean_token_ids=clean_token_ids,
        seed=1729,
    )
    schedule.validate()
    for dose in EXPOSURE_LEVELS:
        assert len(schedule.arms[dose]) == CANARY_ROW_COUNT * 8
        assert (
            sum(slot.exposed for slot in schedule.arms[dose]) == CANARY_ROW_COUNT * dose
        )
        assert {len(slot.input_ids) for slot in schedule.arms[dose]} <= {2, 3}


def _seeded_schedule(seed: int) -> CanarySchedule:
    row_ids = [str(index) for index in range(CANARY_ROW_COUNT)]
    augmented_token_ids = {
        row_id: [index + 1, index + 2] for index, row_id in enumerate(row_ids)
    }
    clean_token_ids = {row_id: [index + 1000] for index, row_id in enumerate(row_ids)}
    return build_canary_schedule(
        row_ids=row_ids,
        augmented_token_ids=augmented_token_ids,
        clean_token_ids=clean_token_ids,
        seed=seed,
    )


def test_schedule_seed_is_reproducible_and_changes_order() -> None:
    """Seeds change the ordered schedule, but not its controlled design."""
    first = _seeded_schedule(seed=1729)
    repeat = _seeded_schedule(seed=1729)
    changed = _seeded_schedule(seed=1730)
    first_fingerprint = _fingerprint(static_identity={}, schedule=first)
    repeat_fingerprint = _fingerprint(static_identity={}, schedule=repeat)
    changed_fingerprint = _fingerprint(static_identity={}, schedule=changed)
    first_identity = t.cast(dict[str, object], first_fingerprint["identity"])
    repeat_identity = t.cast(dict[str, object], repeat_fingerprint["identity"])
    changed_identity = t.cast(dict[str, object], changed_fingerprint["identity"])

    assert first == repeat
    assert first_identity["schedule_sha256"] == repeat_identity["schedule_sha256"]
    assert first != changed
    assert first_identity["schedule_sha256"] != changed_identity["schedule_sha256"]
    common_keys = [(slot.row_id, slot.slot_index) for slot in first.arms[0]]
    assert common_keys != [(slot.row_id, slot.slot_index) for slot in changed.arms[0]]
    for dose in EXPOSURE_LEVELS:
        assert [
            (slot.row_id, slot.slot_index) for slot in first.arms[dose]
        ] == common_keys
    for low, high in zip(EXPOSURE_LEVELS, EXPOSURE_LEVELS[1:]):
        low_membership = {
            (slot.row_id, slot.slot_index) for slot in first.arms[low] if slot.exposed
        }
        high_membership = {
            (slot.row_id, slot.slot_index) for slot in first.arms[high] if slot.exposed
        }
        assert low_membership < high_membership
    token_budgets = {
        sum(len(slot.input_ids) for slot in first.arms[dose])
        for dose in EXPOSURE_LEVELS
    }
    assert token_budgets == {sum(len(slot.input_ids) for slot in first.arms[0])}


def test_training_sees_seeded_schedule_order() -> None:
    """Training consumes the common slot order selected by the seed."""

    class RecordingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.ones(1))
            self.seen: list[int] = []

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor,
        ) -> SimpleNamespace:
            del attention_mask, labels
            self.seen.extend(int(value) for value in input_ids[:, 0])
            return SimpleNamespace(loss=self.parameter * input_ids.float().mean())

    args = argparse.Namespace(
        epochs=1, batch_size=1, learning_rate=1e-5, pad_token_id=0
    )
    first_model = RecordingModel()
    changed_model = RecordingModel()
    _train(
        model=first_model,
        slots=_seeded_schedule(seed=1729).arms[8],
        device=torch.device("cpu"),
        args=args,
    )
    _train(
        model=changed_model,
        slots=_seeded_schedule(seed=1730).arms[8],
        device=torch.device("cpu"),
        args=args,
    )
    assert first_model.seen != changed_model.seen


def test_schedule_rejects_bool_exposure_keys() -> None:
    """Boolean arm keys cannot masquerade as the zero and one doses."""
    row_ids = [str(index) for index in range(CANARY_ROW_COUNT)]
    token_ids = {row_id: [1, 2] for row_id in row_ids}
    schedule = build_canary_schedule(
        row_ids=row_ids,
        augmented_token_ids=token_ids,
        clean_token_ids=token_ids,
        seed=1729,
    )
    invalid_keys = (False, True, 2, 4, 8)
    invalid_schedule = CanarySchedule(
        arms=dict(zip(invalid_keys, schedule.arms.values(), strict=True)),
        row_lengths=schedule.row_lengths,
    )
    with pytest.raises(ValueError, match="schedule must contain"):
        invalid_schedule.validate()


def test_scoring_reports_exact_partial_and_teacher_forced_logprob() -> None:
    """Scoring distinguishes exact, partial and teacher-forced outcomes."""
    scores = score_association(
        model=PredictingModel(),
        prompt_ids=[5, 6],
        target_ids=[1, 2],
        device=torch.device("cpu"),
    )
    assert scores["exact"] == 1
    assert scores["prefix"] == 1.0
    assert math.isclose(float(scores["mean_target_logprob"]), -0.443, abs_tol=0.01)
    partial = score_association(
        model=PredictingModel(),
        prompt_ids=[5, 6],
        target_ids=[1, 3],
        device=torch.device("cpu"),
    )
    assert partial["exact"] == 0
    assert partial["prefix"] == 0.5


def _arms(*, passing: bool) -> dict[int, list[dict[str, object]]]:
    arms: dict[int, list[dict[str, object]]] = {}
    for dose in EXPOSURE_LEVELS:
        rows: list[dict[str, object]] = []
        for index in range(CANARY_ROW_COUNT):
            exposed = 1.0 if dose == 8 and passing else 0.0
            rows.append(
                {
                    "row_id": str(index),
                    "group_id": f"group-{index // 8:02d}",
                    "trigger": private_canary.TRIGGERS[index // 8],
                    "exposed_target": f"exposed-{index // 8:02d}",
                    "control_target": f"control-{index // 8:02d}",
                    "dose": dose,
                    "exact_exposed": exposed,
                    "prefix_exposed": 1.0 if exposed else 0.5,
                    "mean_target_logprob_exposed": exposed,
                    "exact_control": 0.0,
                    "prefix_control": 0.5,
                    "mean_target_logprob_control": 0.0,
                }
            )
        arms[dose] = rows
    return arms


def test_analysis_rejects_bool_exposure_keys() -> None:
    """Boolean mapping keys cannot masquerade as the zero and one doses."""
    valid_arms = _arms(passing=True)
    invalid_keys = (False, True, 2, 4, 8)
    invalid_arms = dict(zip(invalid_keys, valid_arms.values(), strict=True))
    with pytest.raises(ValueError, match="all five exposure levels"):
        analyse_canary_results(invalid_arms)


def test_synthetic_gate_reports_pass_and_fail() -> None:
    """Synthetic row effects exercise both report outcomes."""
    passing = analyse_canary_results(_arms(passing=True))
    failing = analyse_canary_results(_arms(passing=False))
    assert passing["status"] == "pass"
    assert isinstance(passing["paired_randomisation_p"], float)
    assert passing["paired_randomisation_p"] <= 0.01
    passing_gates = t.cast(dict[str, bool], passing["gates"])
    failing_gates = t.cast(dict[str, bool], failing["gates"])
    assert passing_gates["difference_in_difference_prefix"] is True
    assert failing["status"] == "fail"
    assert failing_gates["difference_in_difference_exact"] is False


def test_grouped_learning_passes_with_group_level_analysis() -> None:
    """A consistent eight-row mapping passes the unchanged strict gates."""
    report = analyse_canary_results(_arms(passing=True))
    assert report["status"] == "pass"
    assert report["group_count"] == 32
    assert report["rows_per_group"] == 8


def test_group_imbalance_is_rejected() -> None:
    """A row reassigned to a 33rd group cannot be analysed."""
    arms = _arms(passing=True)
    arms[0][0]["group_id"] = "unbalanced"
    with pytest.raises(ValueError, match="32 groups"):
        analyse_canary_results(arms)


def test_reused_trigger_is_rejected_from_arm_and_analysis() -> None:
    """Reusing one cue cannot be counted as two independent groups."""
    arms = _arms(passing=True)
    for row in arms[0][:8]:
        row["trigger"] = arms[0][8]["trigger"]
    with pytest.raises(ValueError, match="unique validated triggers"):
        validate_arm_payload(
            {
                "protocol": private_canary.CANARY_PROTOCOL,
                "dose": 0,
                "fingerprint": "run",
                "rows": arms[0],
            },
            expected_dose=0,
        )
    with pytest.raises(ValueError, match="unique validated triggers"):
        analyse_canary_results(arms)


def test_row_duplication_is_rejected() -> None:
    """Duplicating a row ID cannot create a valid grouped arm."""
    arms = _arms(passing=True)
    arms[0][0]["row_id"] = arms[0][1]["row_id"]
    with pytest.raises(ValueError, match="row IDs"):
        analyse_canary_results(arms)


def test_row_pseudo_replication_cannot_drive_significance() -> None:
    """Six learning groups fail the p-value despite 48 positive rows."""
    arms = _arms(passing=False)
    for dose in EXPOSURE_LEVELS:
        if dose != 8:
            continue
        for row in arms[dose]:
            if int(str(row["group_id"]).split("-")[-1]) < 6:
                row["exact_exposed"] = 1.0
                row["prefix_exposed"] = 1.0
    report = analyse_canary_results(arms)
    assert report["paired_randomisation_p"] > 0.01
    assert report["status"] == "fail"


def test_declining_controls_cannot_pass() -> None:
    """Control deterioration fails the two-sided stability gate."""
    arms = _arms(passing=True)
    for row in arms[8]:
        row["exact_control"] = 1.0
        row["prefix_control"] = 1.0
    report = analyse_canary_results(arms)
    assert report["status"] == "fail"
    gates = t.cast(dict[str, bool], report["gates"])
    assert gates["control_stability_8"] is False


def test_manifest_rejects_bool_exposure_levels(tmp_path: Path) -> None:
    """Manifest booleans cannot masquerade as the zero and one exposure levels."""
    key_path = tmp_path / "key"
    _key(key_path)
    with pytest.raises(ValueError, match="invalid schedule"):
        canary_runner._validate_inputs(
            manifest={
                "protocol": private_canary.CANARY_PROTOCOL,
                "design_version": private_canary.CANARY_DESIGN_VERSION,
                "hash_version": private_canary.CANARY_HASH_VERSION,
                "fingerprint_version": private_canary.CANARY_FINGERPRINT_VERSION,
                "exposure_levels": [False, True, 2, 4, 8],
            },
            records=[],
            source_rows={},
            augmented_rows={},
            corpus_path=tmp_path / "source.jsonl",
            key_path=key_path,
        )


def test_arm_metadata_rejects_relabel_and_missing_fingerprint() -> None:
    """Persisted arms cannot be relabelled or detached from a run."""
    value = {
        "protocol": private_canary.CANARY_PROTOCOL,
        "dose": 0,
        "fingerprint": "run",
        "rows": [
            {
                "row_id": f"row-{index}",
                "group_id": f"group-{index // 8:02d}",
                "trigger": private_canary.TRIGGERS[index // 8],
                "exposed_target": f"exposed-{index // 8:02d}",
                "control_target": f"control-{index // 8:02d}",
                "dose": 0,
                "exact_exposed": 0.0,
                "prefix_exposed": 0.0,
                "mean_target_logprob_exposed": 0.0,
                "exact_control": 0.0,
                "prefix_control": 0.0,
                "mean_target_logprob_control": 0.0,
            }
            for index in range(CANARY_ROW_COUNT)
        ],
    }
    validate_arm_payload(value, expected_dose=0, filename="arm-0.json")
    with pytest.raises(ValueError, match="filename"):
        validate_arm_payload(value, expected_dose=0, filename="arm-1.json")
    value.pop("fingerprint")
    with pytest.raises(ValueError, match="fingerprint"):
        validate_arm_payload(value, expected_dose=0)


def test_arm_and_row_doses_reject_booleans() -> None:
    """JSON booleans cannot masquerade as arm or row dose integers."""
    for dose, boolean in ((0, False), (1, True)):
        value: dict[str, object] = {
            "protocol": private_canary.CANARY_PROTOCOL,
            "dose": boolean,
            "fingerprint": "run",
            "rows": _arms(passing=True)[dose],
        }
        for row in value["rows"]:
            row["dose"] = boolean
        with pytest.raises(ValueError, match="dose"):
            validate_arm_payload(value, expected_dose=dose, filename=f"arm-{dose}.json")

    value: dict[str, object] = {
        "protocol": private_canary.CANARY_PROTOCOL,
        "dose": 0,
        "fingerprint": "run",
        "rows": _arms(passing=True)[0],
    }
    rows = value["rows"]
    rows[0]["dose"] = False
    with pytest.raises(ValueError, match="dose"):
        validate_arm_payload(value, expected_dose=0, filename="arm-0.json")


def test_execution_rejects_one_row() -> None:
    """An execution arm cannot be analysed with one missing row."""
    value: dict[str, object] = {
        "protocol": private_canary.CANARY_PROTOCOL,
        "dose": 0,
        "fingerprint": "run",
        "rows": _arms(passing=True)[0],
    }
    value["rows"].pop()
    with pytest.raises(ValueError, match="exactly 256"):
        validate_arm_payload(value, expected_dose=0, filename="arm-0.json")


def test_key_mode_is_strict(tmp_path: Path) -> None:
    """Keys with permissive file modes are rejected."""
    path = tmp_path / "key"
    path.write_bytes(b"a" * 32)
    os.chmod(path, 0o644)
    with pytest.raises(ValueError, match="0600"):
        load_key_0600(path)


def test_resume_rejects_mismatched_fingerprint_before_mutation(tmp_path: Path) -> None:
    """A changed run identity cannot relabel existing output."""
    output = tmp_path / "results"
    output.mkdir()
    (output / "fingerprint.json").write_text(json.dumps({"sha256": "old"}))
    with pytest.raises(ValueError, match="mismatched fingerprint"):
        _validate_existing(
            output_dir=output, fingerprint={"sha256": "new"}, row_ids=["row"]
        )
    assert not (output / "arm-0.json").exists()


def test_stale_resume_preflight_is_inert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale static identity cannot invoke loaders or create the cache."""
    source = tmp_path / "source.jsonl"
    key = tmp_path / "key"
    _source(source)
    _key(key)
    private_dir = tmp_path / "private"
    augmented_dir = tmp_path / "augmented"
    generate_canary_corpus(
        corpus_jsonl=source,
        key_path=key,
        augmented_dir=augmented_dir,
        private_dir=private_dir,
        tokenizer=WordTokenizer(),
    )
    records = _read_records(private_dir / "canary-records.jsonl")
    source_rows = _read_rows(source)
    augmented_path = augmented_dir / "augmented.jsonl"
    augmented_rows = _read_rows(augmented_path)
    identity_args = argparse.Namespace(
        batch_size=1, epochs=1, learning_rate=1e-5, pad_token_id=0, seed=17
    )
    static_identity = _static_identity(
        manifest=json.loads((private_dir / "canary-manifest.json").read_text()),
        key=key.read_bytes(),
        device=str(_training_device()),
        args=identity_args,
        records=records,
        row_ids=[record.row_id for record in records],
        source_rows=source_rows,
        augmented_rows=augmented_rows,
    )
    stale_identity = {**static_identity}
    stale_hyperparameters = dict(static_identity["hyperparameters"])
    stale_hyperparameters["epochs"] = 2
    stale_identity["hyperparameters"] = stale_hyperparameters
    fingerprint = {"identity": {**stale_identity, "schedule_sha256": "stale"}}
    fingerprint["sha256"] = hashlib.sha256(
        json.dumps(
            fingerprint["identity"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    fingerprint_path = output_dir / "fingerprint.json"
    fingerprint_path.write_text(json.dumps(fingerprint))
    cache_dir = tmp_path / "cache"
    args = argparse.Namespace(
        corpus_jsonl=source,
        augmented_jsonl=augmented_path,
        private_dir=private_dir,
        key=key,
        output_dir=output_dir,
        cache_dir=cache_dir,
        batch_size=1,
        epochs=1,
        learning_rate=1e-5,
        seed=17,
        pad_token_id=0,
        keep_checkpoints=False,
    )

    def fail_loader(*args: object, **kwargs: object) -> object:
        raise AssertionError("loader called before stale resume was rejected")

    monkeypatch.setattr(canary_runner.AutoTokenizer, "from_pretrained", fail_loader)
    monkeypatch.setattr(canary_runner, "load_dataset", fail_loader)
    monkeypatch.setattr(
        canary_runner.AutoModelForCausalLM, "from_pretrained", fail_loader
    )
    with pytest.raises(ValueError, match="mismatched hyperparameters"):
        run_private_study(args=args)
    assert not cache_dir.exists()
    assert fingerprint_path.read_text() == json.dumps(fingerprint)


def test_record_type_is_private_data_shape() -> None:
    """The private record retains both paired target token sequences."""
    record = CanaryRecord(
        row_id="x",
        source_text_sha256="0",
        exposed_prompt="p",
        exposed_target="a b",
        exposed_text="p a b",
        control_prompt="p",
        control_target="c d",
        control_text="p c d",
        exposed_target_ids=(1, 2),
        control_target_ids=(3, 4),
        group_id="group-00",
        trigger="amber",
    )
    assert record.exposed_target_ids != record.control_target_ids
    assert record.group_id == "group-00"
