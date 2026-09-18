"""Private keyed contamination-canary experiment primitives.

This module is deliberately separate from the public watermark audit.  Canary records
contain plaintext trigger associations and must stay in a mode-0700 private directory.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import math
import os
import platform
import random
import shutil
import stat
import tempfile
import typing as t
from dataclasses import dataclass
from pathlib import Path

import torch

CANARY_ROW_COUNT = 256
CANARY_GROUP_COUNT = 32
ROWS_PER_GROUP = 8
EXPOSURE_LEVELS = (0, 1, 2, 4, 8)
SLOTS_PER_ROW = 8
CANARY_PROTOCOL = "private-local-contamination-canary-v4"
CANARY_DESIGN_VERSION = "keyed-grouped-allocation-v4"
CANARY_HASH_VERSION = 3
CANARY_FINGERPRINT_VERSION = 5
TEMPLATE_PREFIX = "In the quiet archive, the note marked {trigger} referred to"
TEMPLATE_SUFFIX = "."
CONTEXT_PREFIX = "The keyed archive reference was {context}. "
MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
MODEL_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"

# Words are intentionally mundane.  Their keyed combinations are unusual without
# looking like generated identifiers.
TRIGGERS = (
    "amber",
    "birch",
    "cabin",
    "cedar",
    "copper",
    "cricket",
    "distant",
    "evening",
    "fallow",
    "garden",
    "harbour",
    "island",
    "juniper",
    "lantern",
    "meadow",
    "narrow",
    "orchard",
    "pebble",
    "quarry",
    "ribbon",
    "saffron",
    "shelter",
    "timber",
    "velvet",
    "willow",
    "winter",
    "yellow",
    "zephyr",
    "basket",
    "frost",
    "marble",
    "quiet",
)
TARGET_FIRST = (
    "ancient",
    "brisk",
    "careful",
    "clouded",
    "crimson",
    "curious",
    "early",
    "faint",
    "gentle",
    "hidden",
    "hollow",
    "honest",
    "lasting",
    "little",
    "lively",
    "lonely",
    "modest",
    "mellow",
    "patient",
    "plain",
    "polished",
    "remote",
    "restless",
    "silent",
    "steady",
    "subtle",
    "tidy",
    "tranquil",
    "unusual",
    "useful",
    "warm",
    "wooden",
)
TARGET_SECOND = (
    "anchor",
    "apron",
    "bridge",
    "button",
    "candle",
    "clerk",
    "corner",
    "drawer",
    "feather",
    "fountain",
    "hammer",
    "hinge",
    "ladder",
    "letter",
    "market",
    "mirror",
    "notice",
    "pocket",
    "porch",
    "riddle",
    "saddle",
    "signal",
    "ticket",
    "thread",
    "thimble",
    "valley",
    "vessel",
    "window",
    "workshop",
    "yarn",
    "basket",
    "compass",
)


@dataclass(frozen=True)
class CorpusRow:
    """One source corpus row."""

    row_id: str
    text: str


@dataclass(frozen=True)
class CanaryRecord:
    """The private association paired with one corpus row."""

    row_id: str
    source_text_sha256: str
    exposed_prompt: str
    exposed_target: str
    exposed_text: str
    control_prompt: str
    control_target: str
    control_text: str
    exposed_target_ids: tuple[int, ...]
    control_target_ids: tuple[int, ...]
    # Defaults retain construction compatibility for private callers; generated
    # records always contain both fields and validation rejects empty values.
    group_id: str = ""
    trigger: str = ""


@dataclass(frozen=True)
class _CanaryFrame:
    """Fixed text and pre-tokenised windows for one canary row."""

    row: CorpusRow
    group_id: str
    trigger: str
    prompt: str
    prefix: str
    source_ids: tuple[int, ...]
    prompt_ids: tuple[int, ...]
    prefix_ids: tuple[int, ...]


@dataclass(frozen=True)
class _TextWindow:
    """Text and token representations used for global leakage checks."""

    text: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class _TargetSelection:
    """One accepted target and its completed canary window."""

    frame: _CanaryFrame
    target: str
    target_ids: tuple[int, ...]
    text: str
    text_ids: tuple[int, ...]


@dataclass(frozen=True)
class CanarySlot:
    """One fixed-token-count training slot."""

    row_id: str
    slot_index: int
    input_ids: tuple[int, ...]
    exposed: bool


@dataclass(frozen=True)
class CanarySchedule:
    """Nested, equal-token exposure arms."""

    arms: dict[int, tuple[CanarySlot, ...]]
    row_lengths: dict[str, int]
    group_ids: dict[str, str] | None = None

    def validate(self) -> None:
        """Validate row counts, equal lengths and strict dose nesting.

        Raises:
            ValueError: If the schedule violates the controlled design.
        """
        arm_keys = tuple(self.arms)
        if (
            any(type(key) is not int for key in arm_keys)
            or tuple(sorted(arm_keys)) != EXPOSURE_LEVELS
        ):
            raise ValueError("schedule must contain 0, 1, 2, 4 and 8 arms")
        row_ids = set(self.row_lengths)
        if len(row_ids) != CANARY_ROW_COUNT:
            raise ValueError("schedule must contain exactly 256 rows")
        group_ids = self.group_ids or _default_group_ids(row_ids=sorted(row_ids))
        _validate_group_ids(group_ids=group_ids, row_ids=row_ids)
        expected_slot_keys = {
            (row_id, slot_index)
            for row_id in row_ids
            for slot_index in range(SLOTS_PER_ROW)
        }
        for dose, slots in self.arms.items():
            if len(slots) != CANARY_ROW_COUNT * SLOTS_PER_ROW:
                raise ValueError(
                    "every arm must contain exactly 256 rows times 8 slots"
                )
            slot_keys = {(slot.row_id, slot.slot_index) for slot in slots}
            if slot_keys != expected_slot_keys:
                raise ValueError("every arm must contain each row slot exactly once")
            if sum(slot.exposed for slot in slots) != CANARY_ROW_COUNT * dose:
                raise ValueError("exposed slot count does not match dose")
            for slot in slots:
                if slot.row_id not in row_ids:
                    raise ValueError("schedule contains an unknown row")
                if len(slot.input_ids) != self.row_lengths[slot.row_id]:
                    raise ValueError("all slots must have the row token length")
        for low, high in itertools.pairwise(EXPOSURE_LEVELS):
            low_keys = _slot_keys(self.arms[low])
            high_keys = _slot_keys(self.arms[high])
            if not low_keys < high_keys:
                raise ValueError("exposed slots must be strictly nested")


def generate_canary_corpus(
    *,
    corpus_jsonl: Path,
    key_path: Path,
    augmented_dir: Path,
    private_dir: Path,
    tokenizer: object,
    tokenizer_id: str = MODEL_ID,
    tokenizer_revision: str = MODEL_REVISION,
) -> dict[str, object]:
    """Generate a deterministic private canary corpus.

    Args:
        corpus_jsonl: Existing local JSONL containing ``row_id`` and ``text``.
        key_path: Mode-0600 file containing exactly 32 key bytes.
        augmented_dir: External directory receiving augmented training JSONL.
        private_dir: External directory receiving plaintext canary records.
        tokenizer: Pinned SmolLM2 tokenizer used for validation.
        tokenizer_id (optional): Model identifier recorded in the manifest.
        tokenizer_revision (optional): Immutable model revision.

    Returns:
        The private manifest dictionary.

    Raises:
        ValueError: If an input or output path is inside the repository.
    """
    repository = Path(__file__).resolve().parents[2]
    if any(
        path.expanduser().resolve().is_relative_to(repository)
        for path in (corpus_jsonl, key_path, augmented_dir, private_dir)
    ):
        raise ValueError(
            "canary corpus inputs and outputs must be outside the repository"
        )
    rows = _read_rows(path=corpus_jsonl)
    key = load_key_0600(path=key_path)
    records = _make_records(rows=rows, key=key, tokenizer=tokenizer)
    if len(records) != CANARY_ROW_COUNT:
        raise ValueError("canary generation must produce exactly 256 records")
    _write_outputs(
        rows=rows,
        records=records,
        corpus_jsonl=corpus_jsonl,
        key=key,
        key_path=key_path,
        augmented_dir=augmented_dir,
        private_dir=private_dir,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
    )
    return _manifest(
        rows=rows,
        records=records,
        corpus_jsonl=corpus_jsonl,
        key=key,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
    )


def load_key_0600(path: Path) -> bytes:
    """Read and validate a mode-0600 32-byte key without logging its contents.

    Returns:
        The key bytes.

    Raises:
        ValueError: If the file mode or key length is invalid.
    """
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise ValueError("canary key must have mode 0600")
    key = path.read_bytes()
    if len(key) != 32:
        raise ValueError("canary key must contain exactly 32 bytes")
    return key


def build_canary_schedule(
    *,
    row_ids: t.Sequence[str],
    augmented_token_ids: t.Mapping[str, t.Sequence[int]],
    clean_token_ids: t.Mapping[str, t.Sequence[int]],
    seed: int,
    group_ids: t.Mapping[str, str] | None = None,
) -> CanarySchedule:
    """Build equal-token nested arms, using exposed rows only at non-zero doses.

    Clean sequences are repeated or truncated to the augmented row length.  This keeps
    every arm's token budget identical while ensuring the control association is never
    supplied to the optimiser.  The seed only permutes the common slot order, so it
    changes optimiser order without changing corpus grouping or exposure membership.

    Args:
        row_ids:
            Ordered row identifiers to include in every exposure arm.
        augmented_token_ids:
            Token IDs for the exposed version of each row.
        clean_token_ids:
            Token IDs for the clean version of each row.
        seed:
            Seed for the deterministic common slot-key permutation.
        group_ids (optional):
            Group identifier for each row. Defaults to keyed row groups.

    Returns:
        A validated nested schedule.

    Raises:
        ValueError: If rows or token sequences are invalid.
    """
    if type(seed) is not int:
        raise ValueError("schedule seed must be an integer")
    if len(row_ids) != CANARY_ROW_COUNT or len(set(row_ids)) != len(row_ids):
        raise ValueError("row IDs must contain exactly 256 unique rows")
    lengths = {row_id: len(augmented_token_ids[row_id]) for row_id in row_ids}
    if not all(lengths.values()):
        raise ValueError("augmented rows must contain tokens")
    for row_id in row_ids:
        if not clean_token_ids[row_id]:
            raise ValueError("clean rows must contain tokens")
    slot_keys = [
        (row_id, slot_index)
        for row_id in row_ids
        for slot_index in range(SLOTS_PER_ROW)
    ]
    slot_keys.sort(
        key=lambda key: _slot_key_digest(seed=seed, row_id=key[0], slot_index=key[1])
    )
    arms: dict[int, tuple[CanarySlot, ...]] = {}
    for dose in EXPOSURE_LEVELS:
        slots: list[CanarySlot] = []
        for row_id, slot_index in slot_keys:
            exposed_ids = tuple(int(value) for value in augmented_token_ids[row_id])
            clean_ids = _equalise_tokens(
                values=clean_token_ids[row_id], length=lengths[row_id]
            )
            is_exposed = slot_index < dose
            slots.append(
                CanarySlot(
                    row_id=row_id,
                    slot_index=slot_index,
                    input_ids=exposed_ids if is_exposed else clean_ids,
                    exposed=is_exposed,
                )
            )
        arms[dose] = tuple(slots)
    schedule = CanarySchedule(
        arms=arms,
        row_lengths=lengths,
        group_ids=dict(group_ids or _default_group_ids(row_ids=list(row_ids))),
    )
    schedule.validate()
    return schedule


def target_continuation(*, target: str) -> str:
    """Return the exact text scored after a semantic target phrase.

    Args:
        target:
          Two ordinary words without boundary whitespace.

    Returns:
        The target preceded by the model's leading-space separator.

    Raises:
        ValueError: If ``target`` is not a two-word semantic phrase.
    """
    if not target or target != target.strip() or len(target.split()) != 2:
        raise ValueError("canary targets must be two words without boundary whitespace")
    return f" {target}"


def score_association(
    *,
    model: object,
    prompt_ids: t.Sequence[int],
    target_ids: t.Sequence[int],
    device: torch.device,
) -> dict[str, float | int]:
    """Score a two-token association by greedy, prefix and teacher-forced metrics.

    Returns:
        Exact, longest-prefix and mean target log-probability scores.

    Raises:
        ValueError: If the target does not contain exactly two tokens.
    """
    if len(target_ids) != 2:
        raise ValueError("canary targets must contain exactly two tokens")
    typed_model = t.cast(t.Any, model)
    prompt = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    target = tuple(int(value) for value in target_ids)
    with torch.no_grad():
        first_output = typed_model(input_ids=prompt)
        first_logits = first_output.logits[0, -1]
        first_prediction = int(torch.argmax(first_logits).item())
        second_input = torch.cat(
            [
                prompt,
                torch.tensor([[first_prediction]], dtype=torch.long, device=device),
            ],
            dim=1,
        )
        second_prediction = int(
            torch.argmax(typed_model(input_ids=second_input).logits[0, -1]).item()
        )
        teacher_input = torch.cat(
            [prompt, torch.tensor([[target[0]]], dtype=torch.long, device=device)],
            dim=1,
        )
        teacher_logits = typed_model(input_ids=teacher_input).logits[0]
        log_probs = torch.log_softmax(teacher_logits, dim=-1)
        target_logprob = float(
            (log_probs[-2, target[0]] + log_probs[-1, target[1]]).item() / 2
        )
    correct = int(first_prediction == target[0]) + int(second_prediction == target[1])
    return {
        "exact": int(correct == 2),
        "prefix": correct / 2 if first_prediction == target[0] else 0.0,
        "mean_target_logprob": target_logprob,
    }


def analyse_canary_results(
    arms: t.Mapping[int, t.Sequence[t.Mapping[str, object]]],
) -> dict[str, object]:
    """Apply strict gates to 32 paired group-level units.

    Eight observations belonging to one association group are averaged before any
    gate or randomisation test.  This prevents the 256 rows from acting as
    pseudo-replicated independent observations.

    Returns:
        A report containing group-level aggregates and every gate result.
    """
    _validate_arm_results(arms)
    group_aggregates: dict[str, dict[str, dict[str, float]]] = {}
    aggregates: dict[str, dict[str, float]] = {}
    for dose in EXPOSURE_LEVELS:
        grouped = _group_aggregate(rows=arms[dose])
        group_aggregates[str(dose)] = grouped
        aggregates[str(dose)] = {
            metric + suffix: _mean(
                [{"value": group[metric + suffix]} for group in grouped.values()],
                "value",
            )
            for metric in ("exact", "prefix", "mean_target_logprob")
            for suffix in ("_exposed", "_control")
        }
    gates: dict[str, bool] = {}
    zero = aggregates["0"]
    gates["zero_exact_balance"] = (
        abs(zero["exact_exposed"] - zero["exact_control"]) <= 0.01
    )
    gates["zero_prefix_balance"] = (
        abs(zero["prefix_exposed"] - zero["prefix_control"]) <= 0.02
    )
    for dose in EXPOSURE_LEVELS[1:]:
        current = aggregates[str(dose)]
        gates[f"control_no_gain_{dose}"] = (
            current["exact_control"] - zero["exact_control"] <= 0.01
            and current["prefix_control"] - zero["prefix_control"] <= 0.02
        )
        gates[f"control_stability_{dose}"] = (
            abs(current["exact_control"] - zero["exact_control"]) <= 0.01
            and abs(current["prefix_control"] - zero["prefix_control"]) <= 0.02
        )
    improvement_exact = aggregates["8"]["exact_exposed"] - zero["exact_exposed"]
    improvement_prefix = aggregates["8"]["prefix_exposed"] - zero["prefix_exposed"]
    gates["exposed_improvement_exact"] = improvement_exact >= 0.10
    gates["exposed_improvement_prefix"] = improvement_prefix >= 0.20
    for metric in ("exact_exposed", "prefix_exposed"):
        values = [aggregates[str(dose)][metric] for dose in EXPOSURE_LEVELS]
        gates[f"monotonic_{metric}"] = all(
            high + 1e-12 >= low for low, high in itertools.pairwise(values)
        )
    diff0 = zero["exact_exposed"] - zero["exact_control"]
    diff8 = aggregates["8"]["exact_exposed"] - aggregates["8"]["exact_control"]
    prefix_diff0 = zero["prefix_exposed"] - zero["prefix_control"]
    prefix_diff8 = aggregates["8"]["prefix_exposed"] - aggregates["8"]["prefix_control"]
    gates["difference_in_difference_exact"] = diff8 - diff0 >= 0.10
    gates["difference_in_difference_prefix"] = prefix_diff8 - prefix_diff0 >= 0.20
    effects_exact = [
        aggregates[str(dose)]["exact_exposed"] - aggregates[str(dose)]["exact_control"]
        for dose in EXPOSURE_LEVELS
    ]
    effects_prefix = [
        aggregates[str(dose)]["prefix_exposed"]
        - aggregates[str(dose)]["prefix_control"]
        for dose in EXPOSURE_LEVELS
    ]
    gates["monotonic_effect_exact"] = all(
        high + 1e-12 >= low for low, high in itertools.pairwise(effects_exact)
    )
    gates["monotonic_effect_prefix"] = all(
        high + 1e-12 >= low for low, high in itertools.pairwise(effects_prefix)
    )
    groups_at_zero = group_aggregates["0"]
    groups_at_eight = group_aggregates["8"]
    group_ids = sorted(groups_at_zero)
    exposed_exact_change = [
        groups_at_eight[group]["exact_exposed"] - groups_at_zero[group]["exact_exposed"]
        for group in group_ids
    ]
    exposed_prefix_change = [
        groups_at_eight[group]["prefix_exposed"]
        - groups_at_zero[group]["prefix_exposed"]
        for group in group_ids
    ]
    exact_p_value = _paired_randomisation_p(exposed_exact_change)
    prefix_p_value = _paired_randomisation_p(exposed_prefix_change)
    gates["paired_randomisation_exposed_exact"] = exact_p_value <= 0.01
    gates["paired_randomisation_exposed_prefix"] = prefix_p_value <= 0.01
    gates["paired_randomisation"] = gates["paired_randomisation_exposed_prefix"]
    return {
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "predeclared_tolerance": 1e-12,
        "minimum_exposed_improvement": {"exact": 0.10, "prefix": 0.20},
        "paired_randomisation_p": prefix_p_value,
        "paired_randomisation_exact_p": exact_p_value,
        "aggregates": aggregates,
        "group_aggregates": group_aggregates,
        "row_count": len(arms[0]),
        "group_count": CANARY_GROUP_COUNT,
        "rows_per_group": ROWS_PER_GROUP,
    }


def canary_records_hash(records: t.Sequence[CanaryRecord]) -> str:
    """Hash private records in the canonical manifest representation.

    Returns:
        The hexadecimal SHA-256 digest.
    """
    payload = json.dumps(
        {
            "hash_version": CANARY_HASH_VERSION,
            "records": [_record_dict(record) for record in records],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return _sha256(payload)


def write_private_json(path: Path, value: object) -> None:
    """Atomically write a mode-0600 JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    _atomic_write(path=path, payload=payload.encode())


def _unique_contexts(*, rows: list[CorpusRow], key: bytes) -> dict[str, str]:
    """Return keyed, collision-free context windows for scoring prompts.

    Raises:
        ValueError: If a unique keyed context cannot be constructed.
    """
    combinations = list(itertools.product(TRIGGERS, repeat=4))
    contexts: dict[str, str] = {}
    used: set[str] = set()
    for row in rows:
        for index in _keyed_indices(
            key=key, domain=b"scoring-context", value=row.row_id, size=len(combinations)
        ):
            words = combinations[index]
            context = CONTEXT_PREFIX.format(context=" ".join(words))
            if context in used:
                continue
            contexts[row.row_id] = context
            used.add(context)
            break
        else:
            raise ValueError("keyed scoring contexts are not unique")
    if len(contexts) != len(rows):
        raise ValueError("every row needs one scoring prompt context")
    return contexts


def _contains(values: t.Sequence[int], needle: t.Sequence[int]) -> bool:
    """Return whether a token sequence occurs in another sequence."""
    width = len(needle)
    return any(
        tuple(values[index : index + width]) == tuple(needle)
        for index in range(len(values) - width + 1)
    )


def _contains_forbidden(
    *, text: str, token_ids: t.Sequence[int], target: str, target_ids: t.Sequence[int]
) -> bool:
    """Check phrase and token-window leakage using the same rules for both roles.

    Returns:
        Whether either representation occurs in the window.
    """
    return target.casefold() in text.casefold() or _contains(token_ids, target_ids)


def _assign_groups(*, rows: list[CorpusRow], key: bytes) -> dict[str, str]:
    """Assign keyed-shuffled rows to balanced, opaque suite groups.

    Returns:
        A row-to-group mapping.
    """
    shuffled = sorted(
        rows,
        key=lambda row: _keyed_sort_key(
            key=key, domain=b"group-assignment", value=row.row_id
        ),
    )
    return {
        row.row_id: f"group-{index // ROWS_PER_GROUP:02d}"
        for index, row in enumerate(shuffled)
    }


def _default_group_ids(*, row_ids: list[str]) -> dict[str, str]:
    """Return the deterministic grouping used by direct schedule callers."""
    return {
        row_id: f"group-{index // ROWS_PER_GROUP:02d}"
        for index, row_id in enumerate(sorted(row_ids))
    }


def _validate_group_ids(*, group_ids: t.Mapping[str, str], row_ids: set[str]) -> None:
    if set(group_ids) != row_ids or len(set(group_ids.values())) != CANARY_GROUP_COUNT:
        raise ValueError("schedule must contain exactly 32 groups")
    counts = {
        group_id: list(group_ids.values()).count(group_id)
        for group_id in set(group_ids.values())
    }
    if set(counts.values()) != {ROWS_PER_GROUP}:
        raise ValueError("every canary group must contain exactly 8 rows")


def _keyed_sort_key(*, key: bytes, domain: bytes, value: str) -> bytes:
    return hmac.new(key, domain + b"\\0" + value.encode(), hashlib.sha256).digest()


def _build_canary_frames(
    *,
    rows: list[CorpusRow],
    key: bytes,
    tokenizer: object,
    contexts: dict[str, str],
    group_ids: dict[str, str],
) -> list[_CanaryFrame]:
    frames: list[_CanaryFrame] = []
    valid_triggers = [
        trigger
        for trigger in TRIGGERS
        if _trigger_is_valid(tokenizer=tokenizer, trigger=trigger)
    ]
    if (
        len(valid_triggers) != CANARY_GROUP_COUNT
        or len(set(valid_triggers)) != CANARY_GROUP_COUNT
    ):
        raise ValueError(
            "ordinary-word lexicon cannot provide exposed associations: "
            "exactly 32 unique validated triggers are required"
        )
    permutation = sorted(
        valid_triggers,
        key=lambda trigger: _keyed_sort_key(
            key=key, domain=b"trigger-permutation", value=trigger
        ),
    )
    group_triggers = dict(
        zip(sorted(set(group_ids.values())), permutation, strict=True)
    )
    for row in rows:
        group_id = group_ids[row.row_id]
        trigger = group_triggers[group_id]
        prefix = contexts[row.row_id] + TEMPLATE_PREFIX.format(trigger=trigger)
        prompt = row.text + "\n" + prefix
        frames.append(
            _CanaryFrame(
                row=row,
                group_id=group_id,
                trigger=trigger,
                prompt=prompt,
                prefix=prefix,
                source_ids=tuple(_encode(tokenizer, row.text)),
                prompt_ids=tuple(_encode(tokenizer, prompt)),
                prefix_ids=tuple(_encode(tokenizer, prefix)),
            )
        )
    return frames


def _fixed_windows(*, frames: list[_CanaryFrame]) -> tuple[_TextWindow, ...]:
    return tuple(
        window
        for frame in frames
        for window in (
            _TextWindow(text=frame.row.text, token_ids=frame.source_ids),
            _TextWindow(text=frame.prompt, token_ids=frame.prompt_ids),
            _TextWindow(text=frame.prefix, token_ids=frame.prefix_ids),
        )
    )


def _select_group_targets(
    *,
    frames: list[_CanaryFrame],
    key: bytes,
    tokenizer: object,
    candidates: list[str],
    fixed_windows: tuple[_TextWindow, ...],
    domain: bytes,
    completed: list[_TargetSelection] | None = None,
) -> dict[str, _TargetSelection]:
    """Choose one globally clean target mapping for each association group.

    Returns:
        One target selection per group.

    Raises:
        ValueError: If no globally clean target is available.
    """
    by_group: dict[str, _CanaryFrame] = {}
    for frame in frames:
        by_group.setdefault(frame.group_id, frame)
    selected: dict[str, _TargetSelection] = {}
    used_targets: set[str] = set()
    used_target_ids: set[tuple[int, ...]] = set()
    done = list(completed or [])
    for group_id in sorted(by_group):
        frame = by_group[group_id]
        selection = _select_target(
            frame=frame,
            key=key,
            domain=domain,
            tokenizer=tokenizer,
            candidates=candidates,
            fixed_windows=fixed_windows,
            completed=done,
            used_targets=used_targets,
            used_target_ids=used_target_ids,
        )
        if selection is None:
            raise ValueError("ordinary-word lexicon cannot provide group associations")
        selected[group_id] = selection
        done.append(selection)
        used_targets.add(selection.target.casefold())
        used_target_ids.add(selection.target_ids)
    return selected


def _records_from_group_targets(
    *,
    frames: list[_CanaryFrame],
    tokenizer: object,
    exposed: dict[str, _TargetSelection],
    controls: dict[str, _TargetSelection],
) -> list[CanaryRecord]:
    records: list[CanaryRecord] = []
    selections: list[_TargetSelection] = []
    for frame in frames:
        exposed_ids = _validated_target_ids(
            tokenizer=tokenizer,
            prompt=frame.prompt,
            target=exposed[frame.group_id].target,
        )
        control_ids = _validated_target_ids(
            tokenizer=tokenizer,
            prompt=frame.prompt,
            target=controls[frame.group_id].target,
        )
        if (
            exposed_ids != exposed[frame.group_id].target_ids
            or control_ids != controls[frame.group_id].target_ids
        ):
            raise ValueError("group target tokenisation is not stable across rows")
        exposed_selection = _selection_for_frame(
            frame=frame,
            target=exposed[frame.group_id].target,
            target_ids=exposed_ids,
            tokenizer=tokenizer,
        )
        control_selection = _selection_for_frame(
            frame=frame,
            target=controls[frame.group_id].target,
            target_ids=control_ids,
            tokenizer=tokenizer,
        )
        selections.extend((exposed_selection, control_selection))
        records.append(
            CanaryRecord(
                row_id=frame.row.row_id,
                group_id=frame.group_id,
                trigger=frame.trigger,
                source_text_sha256=_sha256(frame.row.text.encode()),
                exposed_prompt=frame.prompt,
                exposed_target=exposed_selection.target,
                exposed_text=exposed_selection.text,
                control_prompt=frame.prompt,
                control_target=control_selection.target,
                control_text=control_selection.text,
                exposed_target_ids=exposed_ids,
                control_target_ids=control_ids,
            )
        )
    _validate_group_target_windows(selections=selections)
    return records


def _selection_for_frame(
    *, frame: _CanaryFrame, target: str, target_ids: tuple[int, ...], tokenizer: object
) -> _TargetSelection:
    text = frame.prefix + target_continuation(target=target) + TEMPLATE_SUFFIX
    return _TargetSelection(
        frame=frame,
        target=target,
        target_ids=target_ids,
        text=text,
        text_ids=tuple(_encode(tokenizer, text)),
    )


def _validate_group_target_windows(*, selections: list[_TargetSelection]) -> None:
    """Apply phrase and token leakage checks, allowing repeats within a group.

    Raises:
        ValueError: If a target is repeated within a row or leaks across groups.
    """
    for index, selection in enumerate(selections):
        if (
            _phrase_count(text=selection.text, phrase=selection.target) != 1
            or _token_count(
                token_ids=selection.text_ids, target_ids=selection.target_ids
            )
            != 1
        ):
            raise ValueError("target must occur exactly once in its own canary")
        windows = (
            _TextWindow(text=other.text, token_ids=other.text_ids)
            for other_index, other in enumerate(selections)
            if other_index != index and other.frame.group_id != selection.frame.group_id
        )
        if _leaks_into_windows(
            target=selection.target, target_ids=selection.target_ids, windows=windows
        ):
            raise ValueError("target leaked across canary groups")


def _validate_group_records(*, records: list[CanaryRecord]) -> None:
    if len(records) != CANARY_ROW_COUNT:
        raise ValueError("canary generation must produce exactly 256 records")
    groups = {record.group_id for record in records}
    if len(groups) != CANARY_GROUP_COUNT:
        raise ValueError("canary generation must produce exactly 32 groups")
    counts = {
        group: sum(record.group_id == group for record in records) for group in groups
    }
    if set(counts.values()) != {ROWS_PER_GROUP}:
        raise ValueError("every canary group must contain exactly 8 records")
    if len({record.exposed_prompt for record in records}) != len(records):
        raise ValueError("exposed scoring prompts must be unique")
    if len({record.control_prompt for record in records}) != len(records):
        raise ValueError("control scoring prompts must be unique")
    triggers = {record.trigger for record in records}
    if triggers != set(TRIGGERS):
        raise ValueError("canary records must contain 32 unique validated triggers")
    mappings = {
        group_id: {
            (record.trigger, record.exposed_target, record.control_target)
            for record in records
            if record.group_id == group_id
        }
        for group_id in groups
    }
    if any(len(mapping) != 1 for mapping in mappings.values()):
        raise ValueError("group mappings are not stable")
    exposed_targets = {next(iter(mapping))[1] for mapping in mappings.values()}
    control_targets = {next(iter(mapping))[2] for mapping in mappings.values()}
    if (
        len(exposed_targets) != CANARY_GROUP_COUNT
        or len(control_targets) != CANARY_GROUP_COUNT
        or exposed_targets & control_targets
    ):
        raise ValueError("group control mappings collide")


def _select_target(
    *,
    frame: _CanaryFrame,
    key: bytes,
    domain: bytes,
    tokenizer: object,
    candidates: list[str],
    fixed_windows: tuple[_TextWindow, ...],
    completed: list[_TargetSelection],
    used_targets: set[str],
    used_target_ids: set[tuple[int, ...]],
) -> _TargetSelection | None:
    for candidate_index in _keyed_indices(
        key=key, domain=domain, value=frame.row.row_id, size=len(candidates)
    ):
        target = candidates[candidate_index]
        if target.casefold() in used_targets:
            continue
        try:
            target_ids = _validated_target_ids(
                tokenizer=tokenizer, prompt=frame.prompt, target=target
            )
        except ValueError:
            continue
        if target_ids in used_target_ids or _leaks_into_windows(
            target=target, target_ids=target_ids, windows=fixed_windows
        ):
            continue
        text = frame.prefix + target_continuation(target=target) + TEMPLATE_SUFFIX
        text_ids = tuple(_encode(tokenizer, text))
        proposed = _TargetSelection(
            frame=frame,
            target=target,
            target_ids=target_ids,
            text=text,
            text_ids=text_ids,
        )
        if _selection_is_symmetric(proposed=proposed, completed=completed):
            return proposed
    return None


def _leaks_into_windows(
    *, target: str, target_ids: tuple[int, ...], windows: t.Iterable[_TextWindow]
) -> bool:
    return any(
        _contains_forbidden(
            text=window.text,
            token_ids=window.token_ids,
            target=target,
            target_ids=target_ids,
        )
        for window in windows
    )


def _selection_is_symmetric(
    *, proposed: _TargetSelection, completed: list[_TargetSelection]
) -> bool:
    if (
        _phrase_count(text=proposed.text, phrase=proposed.target) != 1
        or _token_count(token_ids=proposed.text_ids, target_ids=proposed.target_ids)
        != 1
    ):
        return False
    proposed_window = _TextWindow(text=proposed.text, token_ids=proposed.text_ids)
    if _leaks_into_windows(
        target=proposed.target,
        target_ids=proposed.target_ids,
        windows=(
            _TextWindow(text=selection.text, token_ids=selection.text_ids)
            for selection in completed
        ),
    ):
        return False
    return not any(
        _contains_forbidden(
            text=proposed_window.text,
            token_ids=proposed_window.token_ids,
            target=selection.target,
            target_ids=selection.target_ids,
        )
        for selection in completed
    )


def _phrase_count(*, text: str, phrase: str) -> int:
    return text.casefold().count(phrase.casefold())


def _token_count(*, token_ids: t.Sequence[int], target_ids: t.Sequence[int]) -> int:
    width = len(target_ids)
    return sum(
        token_ids[index : index + width] == target_ids
        for index in range(len(token_ids) - width + 1)
    )


def _make_records(
    *, rows: list[CorpusRow], key: bytes, tokenizer: object
) -> list[CanaryRecord]:
    candidates = [
        f"{first} {second}" for first in TARGET_FIRST for second in TARGET_SECOND
    ]
    ordered_rows = sorted(rows, key=lambda value: value.row_id)
    group_ids = _assign_groups(rows=ordered_rows, key=key)
    contexts = _unique_contexts(rows=ordered_rows, key=key)
    frames = _build_canary_frames(
        rows=ordered_rows,
        key=key,
        tokenizer=tokenizer,
        contexts=contexts,
        group_ids=group_ids,
    )
    fixed_windows = _fixed_windows(frames=frames)
    exposed = _select_group_targets(
        frames=frames,
        key=key,
        tokenizer=tokenizer,
        candidates=candidates,
        fixed_windows=fixed_windows,
        domain=b"exposed-target",
    )
    controls = _select_group_targets(
        frames=frames,
        key=key,
        tokenizer=tokenizer,
        candidates=candidates,
        fixed_windows=fixed_windows,
        domain=b"control-target",
        completed=list(exposed.values()),
    )
    records = _records_from_group_targets(
        frames=frames, tokenizer=tokenizer, exposed=exposed, controls=controls
    )
    _validate_group_records(records=records)
    by_id = {record.row_id: record for record in records}
    return [by_id[row.row_id] for row in rows]


def _trigger_is_valid(*, tokenizer: object, trigger: str) -> bool:
    try:
        _validated_trigger(tokenizer=tokenizer, trigger=trigger)
    except ValueError:
        return False
    return True


def _validated_trigger(*, tokenizer: object, trigger: str) -> None:
    before, after = TEMPLATE_PREFIX.split("{trigger}", maxsplit=1)
    trigger_ids = _encode(tokenizer, trigger)
    context = before + trigger + after
    context_ids = _encode(tokenizer, context)
    specials = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    if not trigger_ids or any(value in specials for value in trigger_ids):
        raise ValueError("trigger is not a non-special model token sequence")
    # Drop the delimiter before checking the context prefix.  The delimiter is
    # intentionally owned by the first trigger token by byte-pair tokenisers.
    before_ids = _encode(tokenizer, before.rstrip())
    with_trigger_ids = _encode(tokenizer, before + trigger)
    if with_trigger_ids[: len(before_ids)] != before_ids or len(
        with_trigger_ids
    ) == len(before_ids):
        raise ValueError("trigger context is unstable")
    if any(value in specials for value in context_ids):
        raise ValueError("trigger context contains special tokens")
    _validate_round_trip(tokenizer=tokenizer, text=trigger, token_ids=trigger_ids)
    _validate_round_trip(tokenizer=tokenizer, text=context, token_ids=context_ids)


def _validated_target_ids(
    *, tokenizer: object, prompt: str, target: str
) -> tuple[int, ...]:
    continuation = target_continuation(target=target)
    prompt_ids = _encode(tokenizer, prompt)
    prompt_target_ids = _encode(tokenizer, prompt + continuation)
    if prompt_target_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt boundary is unstable")
    target_ids = tuple(prompt_target_ids[len(prompt_ids) :])
    full_ids = _encode(tokenizer, prompt + continuation + TEMPLATE_SUFFIX)
    if full_ids[: len(prompt_target_ids)] != prompt_target_ids:
        raise ValueError("target boundary is unstable")
    specials = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    if len(target_ids) != 2 or any(value in specials for value in target_ids):
        raise ValueError("target is not exactly two non-special model tokens")
    _validate_round_trip(tokenizer=tokenizer, text=continuation, token_ids=target_ids)
    return target_ids


def _validate_round_trip(
    *, tokenizer: object, text: str, token_ids: t.Sequence[int]
) -> None:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return
    try:
        decoded = decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        try:
            decoded = decode(list(token_ids))
        except TypeError:
            return
    if decoded != text:
        raise ValueError("tokenizer text round-trip is unstable")


def _write_outputs(
    *,
    rows: list[CorpusRow],
    records: list[CanaryRecord],
    corpus_jsonl: Path,
    key: bytes,
    key_path: Path,
    augmented_dir: Path,
    private_dir: Path,
    tokenizer_id: str,
    tokenizer_revision: str,
) -> None:
    augmented_dir.parent.mkdir(parents=True, exist_ok=True)
    private_dir.parent.mkdir(parents=True, exist_ok=True)
    by_id = {record.row_id: record for record in records}
    augmented = "".join(
        json.dumps(
            {
                "row_id": row.row_id,
                "text": row.text + "\n" + by_id[row.row_id].exposed_text,
            },
            ensure_ascii=True,
        )
        + "\n"
        for row in rows
    )
    private_records = "".join(
        json.dumps(_record_dict(record), ensure_ascii=True) + "\n" for record in records
    )
    manifest = _manifest(
        rows=rows,
        records=records,
        corpus_jsonl=corpus_jsonl,
        key=key,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
    )
    augmented_stage, private_stage = _create_stage_dirs(
        destinations=(augmented_dir, private_dir)
    )
    try:
        _atomic_write(
            path=augmented_stage / "augmented.jsonl", payload=augmented.encode()
        )
        _atomic_write(
            path=private_stage / "canary-records.jsonl",
            payload=private_records.encode(),
        )
        _atomic_write(
            path=private_stage / "canary-manifest.json",
            payload=(json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(),
        )
        # Keep this check close to output creation: changing the key during a run must
        # not silently produce a manifest for a different private record set.
        if hashlib.sha256(key_path.read_bytes()).hexdigest() != manifest["key_sha256"]:
            raise ValueError("canary key changed while writing outputs")
        _publish_output_dirs(
            staged=(augmented_stage, private_stage),
            destinations=(augmented_dir, private_dir),
        )
    except BaseException:
        _remove_path(path=augmented_stage)
        _remove_path(path=private_stage)
        raise


def _create_stage_dirs(*, destinations: tuple[Path, Path]) -> tuple[Path, Path]:
    stages: list[Path] = []
    try:
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(
                tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
            )
            stages.append(stage)
            os.chmod(stage, 0o700)
    except BaseException:
        for stage in stages:
            _remove_path(path=stage)
        raise
    return t.cast(tuple[Path, Path], tuple(stages))


def _publish_output_dirs(
    *, staged: tuple[Path, Path], destinations: tuple[Path, Path]
) -> None:
    """Publish both output directories or restore their previous state."""
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    success = False
    try:
        for destination in destinations:
            if destination.exists():
                backup = Path(
                    tempfile.mkdtemp(
                        prefix=f".{destination.name}.backup.", dir=destination.parent
                    )
                )
                backup.rmdir()
                try:
                    os.replace(destination, backup)
                except BaseException:
                    _remove_path(path=backup)
                    raise
                backups.append((destination, backup))
        for stage, destination in zip(staged, destinations, strict=True):
            os.replace(stage, destination)
            published.append(destination)
        success = True
    except BaseException:
        for destination in published:
            _remove_path(path=destination)
        for destination, backup in reversed(backups):
            if backup.exists():
                os.replace(backup, destination)
        raise
    finally:
        if success:
            for _, backup in backups:
                _remove_path(path=backup)


def _remove_path(*, path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _manifest(
    *,
    rows: list[CorpusRow],
    records: list[CanaryRecord],
    corpus_jsonl: Path,
    key: bytes,
    tokenizer_id: str,
    tokenizer_revision: str,
) -> dict[str, object]:
    if len(rows) != CANARY_ROW_COUNT or len(records) != CANARY_ROW_COUNT:
        raise ValueError("canary manifest requires exactly 256 rows and records")
    _validate_group_records(records=records)
    record_hash = canary_records_hash(records=records)
    return {
        "protocol": CANARY_PROTOCOL,
        "design_version": CANARY_DESIGN_VERSION,
        "hash_version": CANARY_HASH_VERSION,
        "fingerprint_version": CANARY_FINGERPRINT_VERSION,
        "corpus_sha256": _sha256(corpus_jsonl.read_bytes()),
        "canary_records_sha256": record_hash,
        "key_sha256": _sha256(key),
        "row_count": len(rows),
        "prompt_units": len(records),
        "template": {"prefix": TEMPLATE_PREFIX, "suffix": TEMPLATE_SUFFIX},
        "tokenizer": {"id": tokenizer_id, "revision": tokenizer_revision},
        "exposure_levels": list(EXPOSURE_LEVELS),
        "slots_per_row": SLOTS_PER_ROW,
        "group_count": CANARY_GROUP_COUNT,
        "rows_per_group": ROWS_PER_GROUP,
        "triggers": sorted({record.trigger for record in records}),
    }


def _read_rows(*, path: Path) -> list[CorpusRow]:
    rows: list[CorpusRow] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid corpus JSON on line {line_number}") from error
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("row_id"), str)
            or not isinstance(value.get("text"), str)
        ):
            raise ValueError("corpus rows must contain string row_id and text")
        row_id = value["row_id"]
        if row_id in seen:
            raise ValueError("corpus row IDs must be unique")
        seen.add(row_id)
        rows.append(CorpusRow(row_id=row_id, text=value["text"]))
    if len(rows) != CANARY_ROW_COUNT:
        raise ValueError("corpus must contain exactly 256 rows")
    return rows


def _record_dict(record: CanaryRecord) -> dict[str, object]:
    return {
        "row_id": record.row_id,
        "group_id": record.group_id,
        "trigger": record.trigger,
        "source_text_sha256": record.source_text_sha256,
        "exposed_prompt": record.exposed_prompt,
        "exposed_target": record.exposed_target,
        "exposed_text": record.exposed_text,
        "control_prompt": record.control_prompt,
        "control_target": record.control_target,
        "control_text": record.control_text,
        "exposed_target_ids": list(record.exposed_target_ids),
        "control_target_ids": list(record.control_target_ids),
    }


def _encode(tokenizer: object, text: str) -> list[int]:
    encoded = t.cast(t.Any, tokenizer).encode(text, add_special_tokens=False)
    return [int(value) for value in encoded]


def _keyed_index(*, key: bytes, domain: bytes, value: str, size: int) -> int:
    digest = hmac.new(key, domain + b"\0" + value.encode(), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") % size


def _keyed_indices(
    *, key: bytes, domain: bytes, value: str, size: int
) -> t.Iterator[int]:
    start = _keyed_index(key=key, domain=domain, value=value, size=size)
    for offset in range(size):
        yield (start + offset) % size


def _slot_key_digest(*, seed: int, row_id: str, slot_index: int) -> bytes:
    payload = json.dumps((seed, row_id, slot_index), separators=(",", ":")).encode()
    return hashlib.sha256(payload).digest()


def _equalise_tokens(*, values: t.Sequence[int], length: int) -> tuple[int, ...]:
    return tuple(int(values[index % len(values)]) for index in range(length))


def _slot_keys(slots: t.Sequence[CanarySlot]) -> set[tuple[str, int]]:
    return {(slot.row_id, slot.slot_index) for slot in slots if slot.exposed}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(*, path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _mean(rows: t.Sequence[t.Mapping[str, object]], key: str) -> float:
    return sum(_number(row[key]) for row in rows) / len(rows)


def _number(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ValueError("score must be numeric")


def _validate_arm_mappings(*, rows: list[dict[str, object]]) -> None:
    triggers = [row["trigger"] for row in rows]
    exposed_targets = [row["exposed_target"] for row in rows]
    control_targets = [row["control_target"] for row in rows]
    if any(
        not isinstance(value, str) or not value
        for value in (*triggers, *exposed_targets, *control_targets)
    ):
        raise ValueError("canary arm mappings must be non-empty strings")
    trigger_set = {t.cast(str, value) for value in triggers}
    if trigger_set != set(TRIGGERS):
        raise ValueError("canary arm rows must contain 32 unique validated triggers")
    groups: dict[str, set[tuple[str, str, str]]] = {}
    for row in rows:
        group_id = t.cast(str, row["group_id"])
        groups.setdefault(group_id, set()).add(
            (
                t.cast(str, row["trigger"]),
                t.cast(str, row["exposed_target"]),
                t.cast(str, row["control_target"]),
            )
        )
    if any(len(mapping) != 1 for mapping in groups.values()):
        raise ValueError("canary arm group mappings are not stable")
    mappings = {next(iter(mapping)) for mapping in groups.values()}
    group_exposed = {mapping[1] for mapping in mappings}
    group_controls = {mapping[2] for mapping in mappings}
    if (
        len(mappings) != CANARY_GROUP_COUNT
        or len(group_exposed) != CANARY_GROUP_COUNT
        or len(group_controls) != CANARY_GROUP_COUNT
        or group_exposed & group_controls
    ):
        raise ValueError("canary arm group mappings are not independent")


def validate_arm_payload(
    value: dict[str, object],
    *,
    expected_dose: int,
    expected_fingerprint: str | None = None,
    expected_row_ids: t.Sequence[str] | None = None,
    expected_group_ids: t.Mapping[str, str] | None = None,
    filename: str | None = None,
) -> list[dict[str, object]]:
    """Validate one persisted arm before it can be resumed or analysed.

    Returns:
        The validated row dictionaries.

    Raises:
        ValueError: If metadata, row IDs or score types are invalid.
    """
    if filename is not None and filename != f"arm-{expected_dose}.json":
        raise ValueError("canary arm filename does not match its dose")
    if value.get("protocol") != CANARY_PROTOCOL:
        raise ValueError("canary arm has an invalid protocol")
    fingerprint = value.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("canary arm has no fingerprint")
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError("canary arm fingerprint does not match this run")
    dose = value.get("dose")
    if type(dose) is not int or dose != expected_dose:
        raise ValueError("canary arm dose does not match its filename")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != CANARY_ROW_COUNT:
        raise ValueError("canary arm result must contain exactly 256 rows")
    required = {
        "row_id",
        "group_id",
        "trigger",
        "exposed_target",
        "control_target",
        "dose",
        "exact_exposed",
        "prefix_exposed",
        "mean_target_logprob_exposed",
        "exact_control",
        "prefix_control",
        "mean_target_logprob_control",
    }
    if any(not isinstance(row, dict) or not required <= row.keys() for row in rows):
        raise ValueError("canary arm result has incomplete row results")
    ids = [row["row_id"] for row in rows]
    if any(not isinstance(row_id, str) or not row_id for row_id in ids):
        raise ValueError("canary arm row IDs must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("canary arm row IDs must be unique")
    group_values = [row["group_id"] for row in rows]
    if any(not isinstance(group_id, str) or not group_id for group_id in group_values):
        raise ValueError("canary arm group IDs must be non-empty strings")
    group_counts = {
        group_id: group_values.count(group_id) for group_id in set(group_values)
    }
    if len(group_counts) != CANARY_GROUP_COUNT or set(group_counts.values()) != {
        ROWS_PER_GROUP
    }:
        raise ValueError("canary arm must contain exactly 32 groups of 8 rows")
    if expected_row_ids is not None and ids != list(expected_row_ids):
        raise ValueError("canary arm result has mismatched row IDs or order")
    _validate_expected_group_ids(rows=rows, expected=expected_group_ids)
    _validate_arm_mappings(rows=t.cast(list[dict[str, object]], rows))
    for row in rows:
        row_dose = row["dose"]
        if type(row_dose) is not int or row_dose != expected_dose:
            raise ValueError("canary row dose does not match its arm")
        for metric in ("exact_exposed", "exact_control"):
            score = _number(row[metric])
            if score not in (0.0, 1.0):
                raise ValueError("exact scores must be binary")
        for metric in ("prefix_exposed", "prefix_control"):
            score = _number(row[metric])
            if not 0.0 <= score <= 1.0:
                raise ValueError("prefix scores must be fractions")
        for metric in ("mean_target_logprob_exposed", "mean_target_logprob_control"):
            if not math.isfinite(_number(row[metric])):
                raise ValueError("log-probability scores must be finite")
    return t.cast(list[dict[str, object]], rows)


def _validate_expected_group_ids(
    *, rows: list[dict[str, object]], expected: t.Mapping[str, str] | None
) -> None:
    if expected is not None:
        row_groups = {str(row["row_id"]): str(row["group_id"]) for row in rows}
        if row_groups != dict(expected):
            raise ValueError("canary arm result has mismatched group IDs")


def _group_aggregate(
    *, rows: t.Sequence[t.Mapping[str, object]]
) -> dict[str, dict[str, float]]:
    """Average the eight row observations into one value per group.

    Returns:
        Group-level means for every scored metric.
    """
    grouped: dict[str, list[t.Mapping[str, object]]] = {}
    for row in rows:
        group_id = row["group_id"]
        assert isinstance(group_id, str)
        grouped.setdefault(group_id, []).append(row)
    return {
        group_id: {
            metric + suffix: sum(_number(row[metric + suffix]) for row in group_rows)
            / len(group_rows)
            for metric in ("exact", "prefix", "mean_target_logprob")
            for suffix in ("_exposed", "_control")
        }
        for group_id, group_rows in grouped.items()
    }


def _validate_arm_results(
    arms: t.Mapping[int, t.Sequence[t.Mapping[str, object]]],
) -> None:
    arm_keys = tuple(arms)
    if (
        any(type(key) is not int for key in arm_keys)
        or tuple(sorted(arm_keys)) != EXPOSURE_LEVELS
    ):
        raise ValueError("results must contain all five exposure levels")
    row_ids: set[str] | None = None
    expected_row_groups: dict[str, str] | None = None
    expected_groups: dict[str, tuple[str, str, str]] | None = None
    for dose in EXPOSURE_LEVELS:
        current = arms[dose]
        normalised = [
            {**row, "dose": dose} if "dose" not in row else dict(row) for row in current
        ]
        value: dict[str, object] = {
            "protocol": CANARY_PROTOCOL,
            "fingerprint": "analysis",
            "dose": dose,
            "rows": normalised,
        }
        validate_arm_payload(value, expected_dose=dose)
        ids = {t.cast(str, row["row_id"]) for row in normalised}
        if row_ids is None:
            row_ids = ids
        if ids != row_ids:
            raise ValueError("arms must contain matching complete rows")
        row_groups = {str(row["row_id"]): str(row["group_id"]) for row in normalised}
        if expected_row_groups is None:
            expected_row_groups = row_groups
        elif row_groups != expected_row_groups:
            raise ValueError("group assignments must match across arms")
        mappings = {
            str(row["group_id"]): (
                t.cast(str, row["trigger"]),
                t.cast(str, row["exposed_target"]),
                t.cast(str, row["control_target"]),
            )
            for row in normalised
        }
        if len(mappings) != CANARY_GROUP_COUNT:
            raise ValueError("results must contain exactly 32 groups")
        if expected_groups is None:
            expected_groups = mappings
        elif mappings != expected_groups:
            raise ValueError("group mappings must match across arms")


def _paired_randomisation_p(values: list[float]) -> float:
    observed = sum(values) / len(values)
    if observed <= 0:
        return 1.0
    if len(values) <= 20:
        exceed = sum(
            sum(sign * value for sign, value in zip(signs, values, strict=True))
            / len(values)
            >= observed - 1e-15
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        )
        return exceed / (2 ** len(values))
    seed_bytes = hashlib.sha256(
        json.dumps(values, separators=(",", ":"), sort_keys=True).encode()
    ).digest()
    generator = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    exceed = 0
    for _ in range(100_000):
        if (
            sum(generator.choice((-1.0, 1.0)) * value for value in values) / len(values)
            >= observed - 1e-15
        ):
            exceed += 1
    return (exceed + 1) / 100_001


def runtime_identity() -> dict[str, str]:
    """Return non-secret runtime identity for resume binding."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", "unknown"),
    }
