"""Terminal adapter for conflicts that require an explicit result owner choice."""

import collections.abc as c
import sys

import click

from .result_conflicts import ConflictChoice, ConflictDecision, resolve_result_conflict
from .result_identity import UnresolvedResultConflictError


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
    model_id = str(current.get("model_id", candidate.get("model_id", "unknown")))
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
    output_fn(click.style(f"┌{'─' * len(heading)}┐", fg="cyan"))
    output_fn(click.style(f"│{heading}│", fg="cyan", bold=True))
    output_fn(click.style(f"└{'─' * len(heading)}┘", fg="cyan"))
    conflicts = model_conflicts or [(current, candidate)]
    output_fn(f"Unresolved conflicts for this model: {len(conflicts)}")
    for stored, incoming in conflicts[:3]:
        output_fn(
            f"{_conflict_identity(stored, incoming)} | "
            f"current: {_summary(stored)} | incoming: {_summary(incoming)}"
        )
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


def _conflict_identity(current: dict[str, object], candidate: dict[str, object]) -> str:
    """Return a concise identity label for a conflicting result pair."""
    return str(candidate.get("dataset", current.get("dataset", "unknown")))


def _summary(record: dict[str, object]) -> str:
    """Summarise comparison fields without exposing the result payload.

    Args:
        record: Result record to summarise.

    Returns:
        A concise timestamp, engine and score summary.
    """
    timestamp = _display(_evaluation_timestamp(record))
    engine = _display(record.get("engine", record.get("model_type", "?")))
    score = _display(record.get("score", "?"))
    return f"time={timestamp}, engine={engine}, score={score}"


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
