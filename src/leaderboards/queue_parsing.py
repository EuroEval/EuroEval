"""Parsers for issue bodies, euroeval output, and JSONL result files.

Pure-function helpers extracted from ``process_evaluation_queue.py`` so
they're easy to reuse and to test in isolation. Everything in here is
side-effect-free except :func:`read_jsonl_lines`, which reads from disk.
"""

from __future__ import annotations

import importlib.metadata
import json
import re
from pathlib import Path

from euroeval.data_models import BenchmarkResult

from .evaluation_common import missing_official_dataset_language_pairs
from .github_api import TITLE_PREFIX

# Matches the "Model ID" section in an issue body (the form template renders
# the model id as the line immediately following a "### Model ID" heading).
MODEL_ID_BODY_RE = re.compile(r"(?:^|\n)#{1,6}\s*Model ID\s*\n+([^\n]+)")

# euroeval emits a summary line like "errored 3 benchmarks" when individual
# (dataset, language) combinations fail without crashing the whole run.
ERRORED_BENCHMARKS_RE = re.compile(r"errored\s+(\d+)\s+benchmarks?", re.IGNORECASE)

# euroeval emits a summary line like "skipped 2 benchmarks" when individual
# (dataset, language) combinations are deliberately skipped. These are not
# failures.
SKIPPED_BENCHMARKS_RE = re.compile(r"skipped\s+(\d+)\s+benchmarks?", re.IGNORECASE)

# Phrase euroeval prints when it cannot load a model because the repo is gated
# and the subprocess lacks the necessary HF token / accepted access terms.
GATED_OUTPUT_RE = re.compile(r"is a gated repository", re.IGNORECASE)


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
    prefix = f"{TITLE_PREFIX} "
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
    matching: list[str] = []
    for line in lines:
        try:
            parsed = BenchmarkResult.from_dict(config=json.loads(line))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if parsed.model == model_id:
            matching.append(line)
    if not matching:
        return False
    missing = missing_official_dataset_language_pairs(
        lines=matching, requested_languages=requested_languages
    )
    return bool(missing)


def euroeval_version() -> str:
    """Return the locally installed EuroEval package version.

    Returns:
        The dotted version string, or "unknown" if the package metadata
        is not available.
    """
    try:
        return importlib.metadata.version("euroeval")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
