"""Parsers for issue bodies, euroeval output, and JSONL result files.

Pure-function helpers extracted from ``process_evaluation_queue.py`` so
they're easy to reuse and to test in isolation. Everything in here is
side-effect-free except :func:`read_jsonl_lines`, which reads from disk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from euroeval.data_models import BenchmarkResult

from .constants import (
    ERRORED_BENCHMARKS_RE,
    ISSUE_TITLE_PREFIX,
    MODEL_ID_BODY_RE,
    SKIPPED_BENCHMARKS_RE,
)
from .evaluation_common import missing_official_dataset_language_pairs

# Strips ANSI colour/style escape codes that euroeval emits when logging. Matches
# both the real escape byte (``\x1b[..m``, as captured live from the pty) and the
# caret-notation form (``^[[..m``) that some log/transport layers substitute.
_ANSI_ESCAPE_RE = re.compile(r"(?:\x1b|\^\[)\[[0-9;]*m")

# Lines that ``FULL_LOG=1`` floods the output with and which bury the real
# error: serialised result records and transformers training-progress dicts.
_NOISE_SUBSTRINGS = (
    '"schema_version"',
    '"evaluation_id"',
    '"raw_results"',
    '"evaluation_results"',
    "samples_per_second",
    "steps_per_second",
)
_TRAINING_PROGRESS_RE = re.compile(
    r"^\{'(loss|eval_loss|train_runtime|train_loss)'|^\{\"(loss|eval_loss)\""
)

# Markers of the actually-useful error text, in priority order of specificity.
_LOAD_ERROR_RE = re.compile(
    r"could not be loaded|Unrecognized configuration class|is a gated repository|"
    r"PYTORCH_ENABLE_MPS_FALLBACK|does not support the .* task",
    re.IGNORECASE,
)
_TRACEBACK_START = "Traceback (most recent call last):"
_EXCEPTION_LINE_RE = re.compile(r"^[\w.]*(Error|Exception|Timeout|Interrupt)\b.*")
_SUMMARY_LINE_RE = re.compile(r"errored\s+\d+\s+benchmarks?", re.IGNORECASE)

# Individual lines longer than this are truncated so a single giant JSON/param
# dump line can't swamp the summary; the informative head is kept.
_MAX_LINE_CHARS = 600


def extract_model_id(title: str, body: str | None = None) -> str | None:
    """Return the model id for an issue, preferring the body's Model ID section.

    Mirrors the frontend's ``extractModelId`` so the queue processor reads
    the same source of truth as the UI: the issue body's ``### Model ID``
    section, falling back to the title prefix for legacy issues that lack
    the section.

    Args:
        title:
            The full GitHub issue title.
        body (optional):
            The markdown body of the issue, or None. Defaults to None.

    Returns:
        The parsed model id, or None if neither the body nor the title
        yield a usable value.
    """
    if body:
        m = MODEL_ID_BODY_RE.search(body)
        if m:
            candidate = m.group(1).strip().strip("`*_").strip()
            if candidate and candidate != "<model-name>":
                return candidate
    prefix = f"{ISSUE_TITLE_PREFIX} "
    if not title.startswith(prefix):
        return None
    rest = title[len(prefix) :].strip()
    return rest if rest and rest != "<model-name>" else None


def num_errored_benchmarks(output: str) -> int:
    """Return the number of errored benchmarks parsed from euroeval output.

    Args:
        output:
            The full captured combined-output of the euroeval subprocess.

    Returns:
        The integer reported by the last ``errored N benchmarks`` line,
        or 0 if no such line is present.
    """
    last = 0
    for m in ERRORED_BENCHMARKS_RE.finditer(output):
        last = int(m.group(1))
    return last


def num_skipped_benchmarks(output: str) -> int:
    """Return the number of skipped benchmarks parsed from euroeval output.

    EuroEval deliberately skips a benchmark when, for instance, the
    model's generative type does not match what the dataset allows. These
    are not failures.

    Args:
        output:
            The full captured combined-output of the euroeval subprocess.

    Returns:
        The integer reported by the last ``skipped N benchmarks`` line,
        or 0 if no such line is present.
    """
    last = 0
    for m in SKIPPED_BENCHMARKS_RE.finditer(output):
        last = int(m.group(1))
    return last


def _is_noise_line(line: str) -> bool:
    """Return True if a line is verbose ``FULL_LOG`` output rather than an error.

    Args:
        line:
            A single (already ANSI-stripped) line of euroeval output.

    Returns:
        True if the line is a serialised result record or a training-progress
        dict that should be dropped from an error summary.
    """
    stripped = line.strip()
    if any(sub in stripped for sub in _NOISE_SUBSTRINGS):
        return True
    return bool(_TRAINING_PROGRESS_RE.match(stripped))


def _last_traceback(lines: list[str]) -> str | None:
    """Return the last Python traceback in ``lines``, or None if there is none.

    Args:
        lines:
            The cleaned, noise-filtered output lines.

    Returns:
        The traceback text, from its ``Traceback (most recent call last):``
        header through the terminating exception line, or None.
    """
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(_TRACEBACK_START):
            start = i
    if start is None:
        return None
    # Extend to the exception line that closes the traceback (the first line at
    # the end that is not an indented frame), keeping the block compact.
    end = start
    for i in range(start + 1, len(lines)):
        end = i
        if _EXCEPTION_LINE_RE.match(lines[i].strip()):
            break
    return "\n".join(lines[start : end + 1]).strip()


def summarise_evaluation_error(output: str, max_chars: int = 4000) -> str:
    """Extract the meaningful error from a euroeval subprocess's output.

    The queue runs euroeval with ``FULL_LOG=1``, so the raw output is dominated
    by serialised result records, training-progress dicts and giant parameter
    lists. A blind tail of that output almost never shows the real error (and is
    frequently identical across unrelated models), so this scans the whole
    output for the informative error text: a Python traceback, a model-load or
    gating error, and the ``errored N benchmarks`` summary line.

    Args:
        output:
            The full captured combined-output of the euroeval subprocess.
        max_chars:
            The maximum length of the returned summary. Defaults to 4000.

    Returns:
        A compact human-readable error summary, or ``"(no output captured)"``
        when nothing usable is found.
    """
    text = _ANSI_ESCAPE_RE.sub("", output)
    lines: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or _is_noise_line(raw_line):
            continue
        line = raw_line.rstrip()
        if len(line) > _MAX_LINE_CHARS:
            line = line[:_MAX_LINE_CHARS] + " …(truncated)"
        lines.append(line)

    if not lines:
        return "(no output captured)"

    parts: list[str] = []
    traceback = _last_traceback(lines=lines)
    if traceback:
        parts.append(traceback)
    for line in reversed(lines):
        if _LOAD_ERROR_RE.search(line):
            parts.append(line.strip())
            break
    for line in reversed(lines):
        if _SUMMARY_LINE_RE.search(line):
            parts.append(line.strip())
            break

    # Fall back to the tail of the cleaned output when no marker was found.
    if not parts:
        parts = [line.strip() for line in lines[-20:]]

    # De-duplicate while preserving order (a traceback may repeat the summary).
    summary = "\n".join(dict.fromkeys(part for part in parts if part)).strip()
    if len(summary) > max_chars:
        summary = summary[-max_chars:].strip()
    return summary or "(no output captured)"


def format_dataset_language_pairs(dataset_language_pairs: set[tuple[str, str]]) -> str:
    """Return a compact stable string representation of dataset/language pairs.

    Args:
        dataset_language_pairs:
            The set of ``(dataset_name, language_code)`` pairs to format.

    Returns:
        A comma-separated string showing up to the first ten pairs, with
        a ``(+N more)`` suffix when truncated.
    """
    sorted_pairs = sorted(dataset_language_pairs)
    preview = [f"{dataset}/{language}" for dataset, language in sorted_pairs[:10]]
    suffix = ""
    if len(sorted_pairs) > 10:
        suffix = f" (+{len(sorted_pairs) - 10} more)"
    return ", ".join(preview) + suffix


def read_jsonl_lines(path: Path) -> list[str]:
    """Return the non-empty lines of a JSONL file, or an empty list if absent.

    Args:
        path:
            Path to the JSONL file.

    Returns:
        The stripped lines that contain at least one non-whitespace
        character.
    """
    if not path.is_file():
        return []
    return [
        line.rstrip("\n")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def result_lines_for_model(lines: list[str], model_id: str) -> list[str]:
    """Return the subset of ``lines`` whose parsed ``model`` matches ``model_id``.

    Args:
        lines:
            The JSONL lines to filter.
        model_id:
            The model id to match.

    Returns:
        The lines whose ``BenchmarkResult.model`` equals ``model_id``.
    """
    out: list[str] = []
    for line in lines:
        try:
            parsed = BenchmarkResult.from_dict(config=json.loads(line))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        # Only standard accuracy runs (not BPC) count towards leaderboard completion
        is_standard = not getattr(parsed, "use_bits_per_character", False)
        if parsed.model == model_id and is_standard:
            out.append(line)
    return out


def completed_languages(lines: list[str], requested_languages: list[str]) -> list[str]:
    """Return requested languages whose official pair coverage is complete.

    Args:
        lines:
            The accumulated JSONL result lines.
        requested_languages:
            The flattened language codes selected on the issue.

    Returns:
        The subset of ``requested_languages`` for which no official
        dataset/language pair is still missing.
    """
    missing = missing_official_dataset_language_pairs(
        lines=lines, requested_languages=requested_languages
    )
    incomplete = {lang for _, lang in missing}
    return [lang for lang in requested_languages if lang not in incomplete]
