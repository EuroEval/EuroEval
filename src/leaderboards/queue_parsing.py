"""Parsers for issue bodies, euroeval output, and JSONL result files.

Pure-function helpers extracted from ``process_evaluation_queue.py`` so
they're easy to reuse and to test in isolation. Everything in here is
side-effect-free except :func:`read_jsonl_lines`, which reads from disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from euroeval.data_models import BenchmarkResult

from .constants import (
    ERRORED_BENCHMARKS_RE,
    ISSUE_TITLE_PREFIX,
    MODEL_ID_BODY_RE,
    SKIPPED_BENCHMARKS_RE,
)
from .evaluation_common import missing_official_dataset_language_pairs


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
        if parsed.model == model_id:
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


def model_has_partial_results(
    lines: list[str], model_id: str, requested_languages: list[str]
) -> bool:
    """Return True if ``model_id`` has some but not all expected result lines.

    "Expected" here is the official dataset/language pairs for the
    requested languages, matching the completeness check used after an
    evaluation run.

    Args:
        lines:
            The current contents of the JSONL results file.
        model_id:
            The Hugging Face model id whose existing results we inspect.
        requested_languages:
            The flattened language codes selected on the issue.

    Returns:
        True when at least one matching result line exists and at least
        one official pair is still missing; False otherwise.
    """
    matching = result_lines_for_model(lines=lines, model_id=model_id)
    if not matching:
        return False
    missing = missing_official_dataset_language_pairs(
        lines=matching, requested_languages=requested_languages
    )
    return bool(missing)
