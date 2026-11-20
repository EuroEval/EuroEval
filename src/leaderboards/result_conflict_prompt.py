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
    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        raise UnresolvedResultConflictError(
            f"Unresolved result conflict for {decision.evidence.identity}"
        )

    output_fn(f"Unresolved result conflict: {decision.evidence.identity}")
    output_fn(f"Reason: {decision.reason.value}")
    output_fn(
        "Evaluation timestamps: "
        f"stored={_display(_evaluation_timestamp(current))}; "
        f"incoming={_display(_evaluation_timestamp(candidate))}"
    )
    for difference in decision.differences:
        output_fn(
            f"{difference.field}: stored={_display(difference.current)}; "
            f"incoming={_display(difference.candidate)}"
        )
    while True:
        try:
            choice = input_fn("Choose [s]tored, [i]ncoming, or [a]bort: ")
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
    return resolve_result_conflict(
        current=current,
        candidate=candidate,
        current_bytes=current_bytes,
        candidate_bytes=candidate_bytes,
        choice=selected,
    )


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
