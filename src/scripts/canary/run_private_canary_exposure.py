"""Run the private local SmolLM2 contamination-canary exposure study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import random
import tempfile
import typing as t
from pathlib import Path

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

import euroeval.private_canary as private_canary
from euroeval.private_canary import (
    CANARY_DESIGN_VERSION,
    CANARY_FINGERPRINT_VERSION,
    CANARY_GROUP_COUNT,
    CANARY_HASH_VERSION,
    CANARY_PROTOCOL,
    CANARY_ROW_COUNT,
    EXPOSURE_LEVELS,
    MODEL_ID,
    MODEL_REVISION,
    ROWS_PER_GROUP,
    SLOTS_PER_ROW,
    TEMPLATE_SUFFIX,
    TRIGGERS,
    CanaryRecord,
    CanarySchedule,
    CanarySlot,
    build_canary_schedule,
    canary_records_hash,
    load_key_0600,
    repository_root,
    runtime_identity,
    score_association,
    target_continuation,
    validate_arm_payload,
)

LOGGER = logging.getLogger(__name__)
CLEAN_CORPUS_ID = "Salesforce/wikitext"
CLEAN_CORPUS_CONFIG = "wikitext-103-raw-v1"
CLEAN_CORPUS_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def main() -> None:
    """Parse arguments and run incomplete private canary arms."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_private_study(args=args)


def run_private_study(*, args: argparse.Namespace) -> None:
    """Train and score a resumable five-arm study without network uploads.

    Raises:
        ValueError: If private inputs or existing resumable state is invalid.
    """
    corpus_path = args.corpus_jsonl.expanduser().resolve()
    private_dir = args.private_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    augmented_path = args.augmented_jsonl.expanduser().resolve()
    key_path = args.key.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    repository = repository_root()
    if any(
        path.is_relative_to(repository)
        for path in (
            corpus_path,
            private_dir,
            output_dir,
            augmented_path,
            key_path,
            cache_dir,
        )
    ):
        raise ValueError(
            "canary inputs, cache and results must be outside the repository"
        )
    manifest = _read_object(private_dir / "canary-manifest.json")
    records = _read_records(private_dir / "canary-records.jsonl")
    source_rows = _read_rows(corpus_path)
    augmented_rows = _read_rows(augmented_path)
    _validate_inputs(
        manifest=manifest,
        records=records,
        source_rows=source_rows,
        augmented_rows=augmented_rows,
        corpus_path=corpus_path,
        key_path=key_path,
    )
    key = load_key_0600(key_path)
    row_ids = [record.row_id for record in records]
    device = _training_device()
    static_identity = _static_identity(
        manifest=manifest,
        key=key,
        device=str(device),
        args=args,
        records=records,
        row_ids=row_ids,
        source_rows=source_rows,
        augmented_rows=augmented_rows,
    )
    # This preflight intentionally happens before creating a cache or invoking any
    # tokenizer, dataset or model loader.  A malformed resume state must be inert.
    expected_group_ids = {record.row_id: record.group_id for record in records}
    _validate_existing_metadata(
        output_dir=output_dir,
        expected_static_identity=static_identity,
        row_ids=row_ids,
        expected_group_ids=expected_group_ids,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.chmod(0o700)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=str(cache_dir)
    )
    augmented_ids = {
        row_id: _encode(tokenizer, row["text"])
        for row_id, row in augmented_rows.items()
    }
    clean_ids = _load_clean_ids(
        tokenizer=tokenizer,
        cache_dir=cache_dir,
        row_ids=row_ids,
        row_lengths={row_id: len(augmented_ids[row_id]) for row_id in row_ids},
        exposed_target_ids=[record.exposed_target_ids for record in records],
        control_target_ids=[record.control_target_ids for record in records],
        exposed_target_texts=[record.exposed_target for record in records],
        control_target_texts=[record.control_target for record in records],
    )
    schedule = build_canary_schedule(
        row_ids=row_ids,
        augmented_token_ids=augmented_ids,
        clean_token_ids=clean_ids,
        group_ids={record.row_id: record.group_id for record in records},
        seed=args.seed,
    )
    fingerprint = _fingerprint(static_identity=static_identity, schedule=schedule)
    _validate_existing(
        output_dir=output_dir,
        fingerprint=fingerprint,
        row_ids=row_ids,
        expected_group_ids=expected_group_ids,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(output_dir, 0o700)
    _atomic_json(output_dir / "fingerprint.json", fingerprint)
    completed = {
        dose: _read_arm(
            output_dir / f"arm-{dose}.json",
            fingerprint=fingerprint,
            expected_group_ids=expected_group_ids,
        )
        for dose in EXPOSURE_LEVELS
        if (output_dir / f"arm-{dose}.json").exists()
    }
    by_id = {record.row_id: record for record in records}
    for dose in EXPOSURE_LEVELS:
        if dose in completed:
            continue
        _set_seed(args.seed)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, cache_dir=str(cache_dir)
        )
        typed_model = t.cast(t.Any, model)
        typed_model.to(device)
        checkpoint = output_dir / f"arm-{dose}.pt"
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            _validate_checkpoint(state=state, dose=dose, fingerprint=fingerprint)
            typed_model.load_state_dict(state["model"])
        else:
            _train(model=model, slots=schedule.arms[dose], device=device, args=args)
            _atomic_torch(
                checkpoint,
                {
                    "model": typed_model.state_dict(),
                    "dose": dose,
                    "fingerprint": fingerprint["sha256"],
                    "schedule_seed": args.seed,
                    "schedule_sha256": _schedule_sha256(fingerprint=fingerprint),
                },
            )
            _atomic_json(
                _checkpoint_metadata_path(checkpoint),
                {
                    "dose": dose,
                    "fingerprint": fingerprint["sha256"],
                    "schedule_seed": args.seed,
                    "schedule_sha256": _schedule_sha256(fingerprint=fingerprint),
                },
            )
        result_rows = _score_rows(
            model=model,
            tokenizer=tokenizer,
            records=by_id,
            row_ids=row_ids,
            device=device,
            dose=dose,
        )
        result = {
            "protocol": CANARY_PROTOCOL,
            "dose": dose,
            "fingerprint": fingerprint["sha256"],
            "schedule_seed": args.seed,
            "schedule_sha256": _schedule_sha256(fingerprint=fingerprint),
            "rows": result_rows,
            "aggregate": _aggregate(result_rows),
        }
        _atomic_json(output_dir / f"arm-{dose}.json", result)
        if not args.keep_checkpoints:
            checkpoint.unlink(missing_ok=True)
            _checkpoint_metadata_path(checkpoint).unlink(missing_ok=True)
    study = {
        "protocol": CANARY_PROTOCOL,
        "fingerprint": fingerprint["sha256"],
        "schedule_seed": args.seed,
        "schedule_sha256": _schedule_sha256(fingerprint=fingerprint),
        "arms": {
            str(dose): _read_arm(
                output_dir / f"arm-{dose}.json",
                fingerprint=fingerprint,
                expected_group_ids=expected_group_ids,
            )
            for dose in EXPOSURE_LEVELS
        },
    }
    _atomic_json(output_dir / "study.json", study)
    LOGGER.info("private canary exposure study completed")


def _load_clean_ids(
    *,
    tokenizer: object,
    cache_dir: Path,
    row_ids: list[str],
    row_lengths: dict[str, int],
    exposed_target_ids: list[tuple[int, ...]],
    control_target_ids: list[tuple[int, ...]],
    exposed_target_texts: list[str],
    control_target_texts: list[str],
) -> dict[str, list[int]]:
    """Load pinned Wikitext fillers excluding both target roles.

    Returns:
        One clean token sequence per corpus row.

    Raises:
        ValueError: If no safe filler can be constructed.
    """
    dataset = load_dataset(
        CLEAN_CORPUS_ID,
        name=CLEAN_CORPUS_CONFIG,
        split="train",
        revision=CLEAN_CORPUS_REVISION,
        cache_dir=str(cache_dir),
    )
    target_ids = [*exposed_target_ids, *control_target_ids]
    target_texts = [*exposed_target_texts, *control_target_texts]
    sequences: list[list[int]] = []
    for item in dataset:
        text = str(item["text"])
        values = _encode(tokenizer, text)
        if (
            values
            and not any(target.casefold() in text.casefold() for target in target_texts)
            and not any(_contains(values, target) for target in target_ids)
        ):
            sequences.append(values)
        if len(sequences) >= 1024:
            break
    if not sequences:
        raise ValueError("pinned Wikitext has no safe clean filler")
    clean: dict[str, list[int]] = {}
    for index, row_id in enumerate(row_ids):
        for offset in range(len(sequences)):
            values = sequences[(index + offset) % len(sequences)]
            equalised = _repeat(values, row_lengths[row_id])
            if not any(_contains(equalised, target) for target in target_ids):
                clean[row_id] = equalised
                break
        else:
            raise ValueError("cannot construct control-free clean filler")
    return clean


def _contains(values: t.Sequence[int], needle: t.Sequence[int]) -> bool:
    width = len(needle)
    return any(
        tuple(values[index : index + width]) == tuple(needle)
        for index in range(len(values) - width + 1)
    )


def _repeat(values: t.Sequence[int], length: int) -> list[int]:
    return [int(values[index % len(values)]) for index in range(length)]


def _score_rows(
    *,
    model: object,
    tokenizer: object,
    records: dict[str, CanaryRecord],
    row_ids: list[str],
    device: torch.device,
    dose: int,
) -> list[dict[str, object]]:
    t.cast(t.Any, model).eval()
    result: list[dict[str, object]] = []
    for row_id in row_ids:
        record = records[row_id]
        exposed = score_association(
            model=model,
            prompt_ids=_encode(tokenizer, record.exposed_prompt),
            target_ids=record.exposed_target_ids,
            device=device,
        )
        control = score_association(
            model=model,
            prompt_ids=_encode(tokenizer, record.control_prompt),
            target_ids=record.control_target_ids,
            device=device,
        )
        result.append(
            {
                "row_id": row_id,
                "group_id": record.group_id,
                "trigger": record.trigger,
                "exposed_target": record.exposed_target,
                "control_target": record.control_target,
                "dose": dose,
                **{f"{key}_exposed": value for key, value in exposed.items()},
                **{f"{key}_control": value for key, value in control.items()},
            }
        )
    if len(result) != CANARY_ROW_COUNT:
        raise ValueError("execution must produce exactly 256 rows")
    return result


def _train(
    *,
    model: object,
    slots: tuple[CanarySlot, ...],
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    typed_model = t.cast(t.Any, model)
    optimizer = torch.optim.AdamW(typed_model.parameters(), lr=args.learning_rate)
    typed_model.train()
    for _ in range(args.epochs):
        for start in range(0, len(slots), args.batch_size):
            batch = slots[start : start + args.batch_size]
            tensors = [torch.tensor(slot.input_ids, dtype=torch.long) for slot in batch]
            input_ids = pad_sequence(
                tensors, batch_first=True, padding_value=args.pad_token_id
            ).to(device)
            labels = input_ids.clone()
            attention = torch.zeros_like(input_ids)
            for index, slot in enumerate(batch):
                labels[index, len(slot.input_ids) :] = -100
                attention[index, : len(slot.input_ids)] = 1
            output = typed_model(
                input_ids=input_ids, attention_mask=attention, labels=labels
            )
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


def _validate_group_structure(
    *, records: list[CanaryRecord], exposed_prompts: set[str], control_prompts: set[str]
) -> tuple[set[str], set[str]]:
    """Validate balanced groups and return their target sets.

    Returns:
        The disjoint exposed and control target sets.

    Raises:
        ValueError: If group membership or mapping invariants are invalid.
    """
    grouped: dict[str, list[CanaryRecord]] = {}
    for record in records:
        if not record.group_id or not record.trigger:
            raise ValueError("canary records have incomplete group metadata")
        grouped.setdefault(record.group_id, []).append(record)
    if len(grouped) != CANARY_GROUP_COUNT or any(
        len(group) != ROWS_PER_GROUP for group in grouped.values()
    ):
        raise ValueError("canary records must contain exactly 32 groups of 8 rows")
    triggers = {record.trigger for record in records}
    if triggers != set(TRIGGERS):
        raise ValueError("canary records must contain 32 unique validated triggers")
    mappings = {
        group_id: {
            (record.trigger, record.exposed_target, record.control_target)
            for record in group
        }
        for group_id, group in grouped.items()
    }
    if any(len(mapping) != 1 for mapping in mappings.values()):
        raise ValueError("group mappings are not stable")
    exposed_targets = {next(iter(mapping))[1] for mapping in mappings.values()}
    control_targets = {next(iter(mapping))[2] for mapping in mappings.values()}
    if (
        len(exposed_prompts) != len(records)
        or len(control_prompts) != len(records)
        or len(exposed_targets) != CANARY_GROUP_COUNT
        or len(control_targets) != CANARY_GROUP_COUNT
        or exposed_targets & control_targets
    ):
        raise ValueError("canary prompt units or targets are not independent")
    return exposed_targets, control_targets


def _validate_no_training_leakage(
    *,
    records: list[CanaryRecord],
    source_rows: dict[str, dict[str, str]],
    augmented_rows: dict[str, dict[str, str]],
) -> None:
    """Ensure neither target arm can be learned from corpus text.

    Raises:
        ValueError: If an exposed or control target occurs in training text.
    """
    source_text = "\n".join(row["text"] for row in source_rows.values()).casefold()
    augmented_text = "\n".join(
        row["text"] for row in augmented_rows.values()
    ).casefold()
    targets = {
        target.casefold()
        for record in records
        for target in (record.exposed_target, record.control_target)
    }
    if any(target in source_text for target in targets) or any(
        record.control_target.casefold() in augmented_text for record in records
    ):
        raise ValueError("source or augmented corpus contains a target leakage")


def _validate_inputs(
    *,
    manifest: dict[str, object],
    records: list[CanaryRecord],
    source_rows: dict[str, dict[str, str]],
    corpus_path: Path,
    augmented_rows: dict[str, dict[str, str]],
    key_path: Path,
) -> None:
    key = load_key_0600(key_path)
    if (
        manifest.get("protocol") != CANARY_PROTOCOL
        or manifest.get("design_version") != CANARY_DESIGN_VERSION
        or manifest.get("hash_version") != CANARY_HASH_VERSION
        or manifest.get("fingerprint_version") != CANARY_FINGERPRINT_VERSION
    ):
        raise ValueError("unsupported canary manifest")
    exposure_levels = manifest.get("exposure_levels")
    if (
        not isinstance(exposure_levels, list)
        or any(type(level) is not int for level in exposure_levels)
        or exposure_levels != list(EXPOSURE_LEVELS)
    ):
        raise ValueError("canary manifest has an invalid schedule")
    if (
        manifest.get("group_count") != CANARY_GROUP_COUNT
        or manifest.get("rows_per_group") != ROWS_PER_GROUP
    ):
        raise ValueError("canary manifest has an invalid group design")
    manifest_triggers = manifest.get("triggers")
    if (
        not isinstance(manifest_triggers, list)
        or any(not isinstance(trigger, str) for trigger in manifest_triggers)
        or len(manifest_triggers) != CANARY_GROUP_COUNT
        or len(set(manifest_triggers)) != CANARY_GROUP_COUNT
        or set(manifest_triggers) != set(TRIGGERS)
    ):
        raise ValueError("canary manifest has invalid trigger allocation")
    if manifest.get("slots_per_row") != SLOTS_PER_ROW or manifest.get("tokenizer") != {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
    }:
        raise ValueError("canary manifest is not pinned to SmolLM2")
    if manifest.get("key_sha256") != hashlib.sha256(key).hexdigest():
        raise ValueError("canary key does not match its manifest")
    if (
        manifest.get("corpus_sha256")
        != hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    ):
        raise ValueError("source corpus does not match its manifest")
    if manifest.get("canary_records_sha256") != canary_records_hash(records=records):
        raise ValueError("canary records do not match their manifest")
    if (
        len(records) != CANARY_ROW_COUNT
        or len(source_rows) != CANARY_ROW_COUNT
        or len(augmented_rows) != CANARY_ROW_COUNT
        or manifest.get("row_count") != CANARY_ROW_COUNT
        or manifest.get("prompt_units") != CANARY_ROW_COUNT
    ):
        raise ValueError("canary inputs must contain exactly 256 rows")
    exposed_prompts = {record.exposed_prompt for record in records}
    control_prompts = {record.control_prompt for record in records}
    _validate_group_structure(
        records=records,
        exposed_prompts=exposed_prompts,
        control_prompts=control_prompts,
    )
    _validate_no_training_leakage(
        records=records, source_rows=source_rows, augmented_rows=augmented_rows
    )
    if set(source_rows) != {record.row_id for record in records} or set(
        augmented_rows
    ) != set(source_rows):
        raise ValueError("source, augmented and canary row IDs differ")
    for record in records:
        source = source_rows[record.row_id]["text"]
        augmented = augmented_rows[record.row_id]["text"]
        if hashlib.sha256(source.encode()).hexdigest() != record.source_text_sha256:
            raise ValueError("source corpus changed since canary generation")
        try:
            exposed_continuation = target_continuation(target=record.exposed_target)
            control_continuation = target_continuation(target=record.control_target)
        except ValueError as error:
            raise ValueError("canary targets have invalid semantic text") from error
        canary_prefix = record.exposed_prompt.removeprefix(source + "\n")
        if record.exposed_prompt != source + "\n" + canary_prefix:
            raise ValueError("training and scoring contexts differ")
        if record.exposed_text != (
            canary_prefix + exposed_continuation + TEMPLATE_SUFFIX
        ) or record.control_text != (
            canary_prefix + control_continuation + TEMPLATE_SUFFIX
        ):
            raise ValueError("canary text does not match its scored continuation")
        expected = source + "\n" + record.exposed_text
        if augmented != expected:
            raise ValueError("augmented corpus has changed")
        if record.control_prompt != record.exposed_prompt:
            raise ValueError("exposed and control contexts are not matched")


def _validate_existing_metadata(
    *,
    output_dir: Path,
    expected_static_identity: dict[str, object],
    row_ids: list[str],
    expected_group_ids: dict[str, str] | None = None,
) -> None:
    """Validate all resumable metadata before cache or loader side effects.

    Raises:
        ValueError: If existing files are not bound to this study.
    """
    if not output_dir.exists():
        return
    fingerprint_path = output_dir / "fingerprint.json"
    existing_files = list(output_dir.iterdir())
    if existing_files and not fingerprint_path.exists():
        raise ValueError("existing canary output has no fingerprint")
    if not fingerprint_path.exists():
        return
    fingerprint = _read_object(fingerprint_path)
    identity = fingerprint.get("identity")
    digest = fingerprint.get("sha256")
    if not isinstance(digest, str) or not digest:
        raise ValueError("existing canary fingerprint is empty")
    if not isinstance(identity, dict):
        raise ValueError("existing canary fingerprint has no identity")
    encoded_identity = json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode()
    if hashlib.sha256(encoded_identity).hexdigest() != digest:
        raise ValueError("existing canary fingerprint digest is invalid")
    expected_keys = {*expected_static_identity, "schedule_sha256"}
    if set(identity) != expected_keys:
        raise ValueError("existing canary fingerprint has incomplete static identity")
    for name, expected in expected_static_identity.items():
        if identity.get(name) != expected:
            raise ValueError(f"existing canary fingerprint has a mismatched {name}")
    if (
        not isinstance(identity["schedule_sha256"], str)
        or not identity["schedule_sha256"]
    ):
        raise ValueError("existing canary fingerprint has no schedule_sha256")
    if type(identity["schedule_seed"]) is not int:
        raise ValueError("existing canary fingerprint has no schedule_seed")
    study_path = output_dir / "study.json"
    if study_path.exists():
        study = _read_object(study_path)
        existing_arms = study.get("arms")
        if (
            study.get("fingerprint") != digest
            or not isinstance(existing_arms, dict)
            or set(existing_arms) != {str(dose) for dose in EXPOSURE_LEVELS}
        ):
            raise ValueError("existing canary study summary is invalid")
        _validate_schedule_metadata(value=study, fingerprint=fingerprint)
        for dose in EXPOSURE_LEVELS:
            arm = existing_arms[str(dose)]
            if not isinstance(arm, dict):
                raise ValueError("existing canary study arm is invalid")
            validate_arm_payload(
                arm,
                expected_dose=dose,
                expected_fingerprint=digest,
                expected_row_ids=row_ids,
                expected_group_ids=expected_group_ids,
                filename=f"arm-{dose}.json",
            )
            _validate_schedule_metadata(value=arm, fingerprint=fingerprint)
    for dose in EXPOSURE_LEVELS:
        result_path = output_dir / f"arm-{dose}.json"
        if result_path.exists():
            value = _read_object(result_path)
            validate_arm_payload(
                value,
                expected_dose=dose,
                expected_fingerprint=digest,
                expected_row_ids=row_ids,
                expected_group_ids=expected_group_ids,
                filename=result_path.name,
            )
            _validate_schedule_metadata(value=value, fingerprint=fingerprint)
        checkpoint = output_dir / f"arm-{dose}.pt"
        metadata_path = _checkpoint_metadata_path(checkpoint)
        if checkpoint.exists() != metadata_path.exists():
            raise ValueError("canary checkpoint metadata is missing")
        if metadata_path.exists():
            _validate_checkpoint_metadata(
                path=metadata_path, dose=dose, fingerprint=fingerprint
            )


def _schedule_sha256(*, fingerprint: dict[str, object]) -> str:
    identity = fingerprint.get("identity")
    schedule_sha256 = (
        identity.get("schedule_sha256") if isinstance(identity, dict) else None
    )
    if not isinstance(schedule_sha256, str):
        raise ValueError("canary fingerprint has no schedule_sha256")
    return schedule_sha256


def _validate_schedule_metadata(
    *, value: dict[str, object], fingerprint: dict[str, object]
) -> None:
    identity = fingerprint.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("canary fingerprint has no identity")
    if value.get("schedule_seed") != identity.get("schedule_seed") or value.get(
        "schedule_sha256"
    ) != identity.get("schedule_sha256"):
        raise ValueError("canary metadata is not bound to the schedule")


def _validate_existing(
    *,
    output_dir: Path,
    fingerprint: dict[str, object],
    row_ids: list[str],
    expected_group_ids: dict[str, str] | None = None,
) -> None:
    if not output_dir.exists():
        return
    fingerprint_path = output_dir / "fingerprint.json"
    existing_files = list(output_dir.iterdir())
    if existing_files and not fingerprint_path.exists():
        raise ValueError("existing canary output has no fingerprint")
    if fingerprint_path.exists() and _read_object(fingerprint_path) != fingerprint:
        raise ValueError("existing canary output has a mismatched fingerprint")
    study_path = output_dir / "study.json"
    if study_path.exists():
        study = _read_object(study_path)
        existing_arms = study.get("arms")
        if (
            study.get("fingerprint") != fingerprint["sha256"]
            or not isinstance(existing_arms, dict)
            or set(existing_arms) != {str(dose) for dose in EXPOSURE_LEVELS}
        ):
            raise ValueError("existing canary study summary is invalid")
        _validate_schedule_metadata(value=study, fingerprint=fingerprint)
        for dose in EXPOSURE_LEVELS:
            arm = existing_arms[str(dose)]
            if not isinstance(arm, dict):
                raise ValueError("existing canary study arm is invalid")
            validate_arm_payload(
                arm,
                expected_dose=dose,
                expected_fingerprint=t.cast(str, fingerprint["sha256"]),
                expected_row_ids=row_ids,
                expected_group_ids=expected_group_ids,
                filename=f"arm-{dose}.json",
            )
            _validate_schedule_metadata(value=arm, fingerprint=fingerprint)
    for dose in EXPOSURE_LEVELS:
        result_path = output_dir / f"arm-{dose}.json"
        if result_path.exists():
            _read_arm(
                result_path,
                fingerprint=fingerprint,
                row_ids=row_ids,
                expected_group_ids=expected_group_ids,
            )
        checkpoint = output_dir / f"arm-{dose}.pt"
        metadata_path = _checkpoint_metadata_path(checkpoint)
        if checkpoint.exists() != metadata_path.exists():
            raise ValueError("canary checkpoint metadata is missing")
        if metadata_path.exists():
            _validate_checkpoint_metadata(
                path=metadata_path, dose=dose, fingerprint=fingerprint
            )


def _read_arm(
    path: Path,
    *,
    fingerprint: dict[str, object],
    row_ids: list[str] | None = None,
    expected_group_ids: dict[str, str] | None = None,
) -> dict[str, object]:
    value = _read_object(path)
    dose = value.get("dose")
    if type(dose) is not int or dose not in EXPOSURE_LEVELS:
        raise ValueError("canary arm result has an invalid dose")
    validate_arm_payload(
        value,
        expected_dose=dose,
        expected_fingerprint=t.cast(str, fingerprint.get("sha256")),
        expected_row_ids=row_ids,
        expected_group_ids=expected_group_ids,
        filename=path.name,
    )
    _validate_schedule_metadata(value=value, fingerprint=fingerprint)
    return value


def _validate_checkpoint(
    *, state: object, dose: int, fingerprint: dict[str, object]
) -> None:
    state_dose = state.get("dose") if isinstance(state, dict) else None
    state_fingerprint = state.get("fingerprint") if isinstance(state, dict) else None
    if (
        type(state_dose) is not int
        or state_dose != dose
        or state_fingerprint != fingerprint.get("sha256")
        or not isinstance(state_fingerprint, str)
        or not state_fingerprint
    ):
        raise ValueError("canary checkpoint is not bound to this run")
    if isinstance(state, dict):
        _validate_schedule_metadata(value=state, fingerprint=fingerprint)
    if not isinstance(state, dict) or not isinstance(state.get("model"), dict):
        raise ValueError("canary checkpoint has no model state")


def _checkpoint_metadata_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".meta.json")


def _validate_checkpoint_metadata(
    *, path: Path, dose: int, fingerprint: dict[str, object]
) -> None:
    value = _read_object(path)
    metadata_dose = value.get("dose")
    if (
        type(metadata_dose) is not int
        or metadata_dose != dose
        or value.get("fingerprint") != fingerprint.get("sha256")
    ):
        raise ValueError("canary checkpoint metadata is not bound to this run")
    _validate_schedule_metadata(value=value, fingerprint=fingerprint)


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    """Aggregate arm rows through balanced group means, not pseudo-replication.

    Returns:
        Group-level means for all arm metrics.

    Raises:
        ValueError: If the arm is not balanced across the required groups.
    """
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        group_id = row["group_id"]
        assert isinstance(group_id, str)
        grouped.setdefault(group_id, []).append(row)
    if len(grouped) != CANARY_GROUP_COUNT or any(
        len(group) != ROWS_PER_GROUP for group in grouped.values()
    ):
        raise ValueError("canary aggregate requires exactly 32 groups of 8 rows")
    keys = (
        "exact_exposed",
        "prefix_exposed",
        "mean_target_logprob_exposed",
        "exact_control",
        "prefix_control",
        "mean_target_logprob_control",
    )
    return {
        key: sum(
            sum(_number(row[key]) for row in group) / ROWS_PER_GROUP
            for group in grouped.values()
        )
        / CANARY_GROUP_COUNT
        for key in keys
    }


def _static_identity(
    *,
    manifest: dict[str, object],
    key: bytes,
    device: str,
    args: argparse.Namespace,
    records: list[CanaryRecord],
    row_ids: list[str],
    source_rows: dict[str, dict[str, str]],
    augmented_rows: dict[str, dict[str, str]],
) -> dict[str, object]:
    """Build the complete resume identity without loading external artefacts.

    Returns:
        The static identity persisted in the run fingerprint.
    """
    ordered_identity = {
        "record_row_ids": row_ids,
        "record_group_ids": [record.group_id for record in records],
        "source_row_ids": list(source_rows),
        "augmented_row_ids": list(augmented_rows),
        "source_text_sha256": [
            hashlib.sha256(source_rows[row_id]["text"].encode()).hexdigest()
            for row_id in row_ids
        ],
        "augmented_text_sha256": [
            hashlib.sha256(augmented_rows[row_id]["text"].encode()).hexdigest()
            for row_id in row_ids
        ],
        "canary_records_sha256": canary_records_hash(records=records),
        "exposure_levels": list(EXPOSURE_LEVELS),
        "slots_per_row": SLOTS_PER_ROW,
        "group_count": CANARY_GROUP_COUNT,
        "rows_per_group": ROWS_PER_GROUP,
    }
    return {
        "protocol": CANARY_PROTOCOL,
        "design_version": CANARY_DESIGN_VERSION,
        "hash_version": CANARY_HASH_VERSION,
        "fingerprint_version": CANARY_FINGERPRINT_VERSION,
        "schedule_seed": args.seed,
        "manifest": manifest,
        "key_sha256": hashlib.sha256(key).hexdigest(),
        "model": [MODEL_ID, MODEL_REVISION],
        "clean_corpus": [CLEAN_CORPUS_ID, CLEAN_CORPUS_CONFIG, CLEAN_CORPUS_REVISION],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "private_canary_sha256": hashlib.sha256(
            Path(private_canary.__file__).read_bytes()
        ).hexdigest(),
        "ordered_schedule_source_identity": ordered_identity,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "datasets")
        },
        "runtime": runtime_identity(),
        "device": device,
        "hyperparameters": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "pad_token_id": args.pad_token_id,
            "seed": args.seed,
            "slots_per_row": SLOTS_PER_ROW,
        },
    }


def _fingerprint(
    *, static_identity: dict[str, object], schedule: CanarySchedule
) -> dict[str, object]:
    schedule_payload = {
        str(dose): [
            {
                "row_id": slot.row_id,
                "slot_index": slot.slot_index,
                "input_ids": list(slot.input_ids),
                "exposed": slot.exposed,
            }
            for slot in schedule.arms[dose]
        ]
        for dose in EXPOSURE_LEVELS
    }
    schedule_sha256 = hashlib.sha256(
        json.dumps(schedule_payload, separators=(",", ":")).encode()
    ).hexdigest()
    identity = {**static_identity, "schedule_sha256": schedule_sha256}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "identity": identity}


def _number(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ValueError("score must be numeric")


def _read_records(path: Path) -> list[CanaryRecord]:
    records: list[CanaryRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        records.append(
            CanaryRecord(
                row_id=value["row_id"],
                group_id=value["group_id"],
                trigger=value["trigger"],
                source_text_sha256=value["source_text_sha256"],
                exposed_prompt=value["exposed_prompt"],
                exposed_target=value["exposed_target"],
                exposed_text=value["exposed_text"],
                control_prompt=value["control_prompt"],
                control_target=value["control_target"],
                control_text=value["control_text"],
                exposed_target_ids=tuple(value["exposed_target_ids"]),
                control_target_ids=tuple(value["control_target_ids"]),
            )
        )
    if len(records) != CANARY_ROW_COUNT:
        raise ValueError("canary records must contain exactly 256 rows")
    return records


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        row_id = str(value["row_id"])
        if row_id in rows:
            raise ValueError("corpus row IDs must be unique")
        rows[row_id] = {"text": str(value["text"])}
    if len(rows) != CANARY_ROW_COUNT:
        raise ValueError("corpus rows must contain exactly 256 rows")
    return rows


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _encode(tokenizer: object, text: str) -> list[int]:
    return [
        int(value)
        for value in t.cast(t.Any, tokenizer).encode(text, add_special_tokens=False)
    ]


def _training_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    ).encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_torch(path: Path, value: object) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--augmented-jsonl", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("~/.cache/huggingface/euroeval-canary")
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--keep-checkpoints", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.learning_rate <= 0:
        parser.error("batch size, epochs and learning rate must be positive")
    return args


if __name__ == "__main__":
    main()
