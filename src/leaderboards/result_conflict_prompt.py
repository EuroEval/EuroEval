"""Terminal adapter for conflicts that require an explicit result owner choice."""

import collections.abc as c
import os
import sys

import click

from .result_conflicts import ConflictChoice, ConflictDecision, resolve_result_conflict
from .result_identity import UnresolvedResultConflictError, identity_from_eee_record


def resolve_result_interactively(
    *,
    current: dict[str, object],
    candidate: dict[str, object],
    current_bytes: bytes | None = None,
    candidate_bytes: bytes | None = None,
    input_fn: c.Callable[[str], str] = input,
    output_fn: c.Callable[[str], None] = click.echo,
    interactive: bool | None = None,
    model_choices: dict[str, ConflictChoice] | None = None,
    model_conflicts: list[tuple[dict[str, object], dict[str, object]]] | None = None,
) -> ConflictDecision:
    """Apply automatic policy, prompting only for unresolved conflicts.

    Args:
        current: Stored record.
        candidate: Incoming record.
        current_bytes (optional): Exact stored bytes.
        candidate_bytes (optional): Exact incoming bytes.
        input_fn (optional): Prompt input function. Defaults to input.
        output_fn (optional): Display function. Defaults to print.
        interactive (optional): Override terminal detection for callers/tests.
        model_choices (optional): Reuse one explicit choice per model.
        model_conflicts (optional): All unresolved pairs for this model.

    Returns:
        The policy decision or explicit human choice.

    Raises:
        UnresolvedResultConflictError: If unresolved without an interactive terminal.
    """
    decision = resolve_result_conflict(
        current=current,
        candidate=candidate,
        current_bytes=current_bytes,
        candidate_bytes=candidate_bytes,
    )
    if not decision.needs_choice:
        return decision
    model_id = identity_from_eee_record(record=current)[0]
    if model_choices is not None and model_id in model_choices:
        return resolve_result_conflict(
            current=current,
            candidate=candidate,
            current_bytes=current_bytes,
            candidate_bytes=candidate_bytes,
            choice=model_choices[model_id],
        )
    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        raise UnresolvedResultConflictError(
            f"Unresolved result conflict for {decision.evidence.identity}"
        )

    heading = f" Result conflicts for {model_id} "
    conflicts = model_conflicts or [(current, candidate)]
    colour = _colour_enabled()
    lines = [
        f"┌{'─' * len(heading)}┐",
        f"│{heading}│",
        f"└{'─' * len(heading)}┘",
        f"Unresolved conflicting identities for this model: {len(conflicts)}",
    ]
    for stored, incoming in conflicts[:3]:
        lines.extend(_preview_box(stored=stored, incoming=incoming))
    if len(conflicts) > 3:
        lines.append(f"... and {len(conflicts) - 3} more conflicts")
    for line in lines:
        output_fn(click.style(line, fg="cyan") if colour else line)
    while True:
        try:
            choice = input_fn(
                "Choose for this model: [s]tored, [i]ncoming, or [a]bort: "
            )
        except (EOFError, KeyboardInterrupt) as error:
            raise UnresolvedResultConflictError(
                "Result conflict resolution aborted"
            ) from error
        choice = choice.strip().lower()
        if choice in {"s", "stored"}:
            selected = ConflictChoice.STORED
            break
        if choice in {"i", "incoming"}:
            selected = ConflictChoice.INCOMING
            break
        if choice in {"a", "abort"}:
            raise UnresolvedResultConflictError("Result conflict resolution aborted")
        output_fn("Invalid choice. Enter s, i, or a.")
    if model_choices is not None:
        model_choices[model_id] = selected
    return resolve_result_conflict(
        current=current,
        candidate=candidate,
        current_bytes=current_bytes,
        candidate_bytes=candidate_bytes,
        choice=selected,
    )


def _colour_enabled() -> bool:
    """Return whether terminal colour is enabled by the environment and Click."""
    if "NO_COLOR" in os.environ:
        return False
    context = click.get_current_context(silent=True)
    return context is None or context.color is not False


def _preview_box(
    *, stored: dict[str, object], incoming: dict[str, object]
) -> list[str]:
    """Format one compact comparison box for a conflicting identity.

    Args:
        stored: Existing result record.
        incoming: Candidate result record.

    Returns:
        Lines forming a bounded comparison box.
    """
    identity = identity_from_eee_record(record=incoming)
    label = f"{identity[1]} | validation={identity[2]} | few-shot={identity[3]}"
    width = min(max(len(label), 48), 100)
    return [
        f"┌{'─' * width}┐",
        f"│ {label[: width - 2]:<{width - 2}} │",
        f"│ stored:   {_summary(stored)[: width - 13]:<{width - 13}} │",
        f"│ incoming: {_summary(incoming)[: width - 13]:<{width - 13}} │",
        f"└{'─' * width}┘",
    ]


def _summary(record: dict[str, object]) -> str:
    """Summarise comparison fields without exposing the result payload.

    Args:
        record: Result record to summarise.

    Returns:
        A concise timestamp, engine and score summary.
    """
    model_info = record.get("model_info", {})
    details = record.get("eval_library", {}).get("additional_details", {})
    timestamp = _display(_evaluation_timestamp(record))
    name = _display(model_info.get("name", model_info.get("id", "?")))
    flags = ", ".join(
        f"{key}={_display(record.get(key, '?'))}"
        for key in ("open", "commercially_licensed", "trained_from_scratch")
    )
    scores = record.get("get_total_scores", details.get("get_total_scores", "?"))
    return f"{name}; {flags}; time={timestamp}; totals={_display(scores)}"


def _display(value: object) -> str:
    """Render a bounded field value for a terminal choice.

    Args:
        value: Field value.

    Returns:
        Bounded representation.
    """
    rendered = repr(value)
    return rendered if len(rendered) <= 240 else f"{rendered[:237]}..."


def _evaluation_timestamp(record: dict[str, object]) -> object:
    """Return the recorded evaluation time, never the retrieval time.

    Args:
        record: Evaluation result record.

    Returns:
        The first available evaluation timestamp or ``None``.
    """
    for key in ("evaluation_timestamp", "eval_timestamp", "evaluated_at"):
        if key in record:
            return record[key]
    return None
