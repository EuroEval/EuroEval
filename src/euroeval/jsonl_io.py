"""JSONL parsing utilities for benchmark results."""

import json
import logging

logger = logging.getLogger(__name__)


def parse_jsonl_lines(
    lines: list[str], source: str, *, strict: bool = True
) -> list[dict[str, object]]:
    """Parse JSONL lines into record dictionaries with robust `}{` handling.

    Blank lines are skipped and concatenated objects (``}{``) on a single
    line are split apart before parsing. This handles malformed JSONL where
    multiple JSON objects appear on the same line without a newline separator.

    Args:
        lines:
            The raw lines to parse.
        source:
            A human-readable label for the input, used in error messages.
        strict (optional):
            If True (default), raise ValueError on the first malformed line.
            If False, log a warning for each malformed line and continue
            processing, returning only valid records.

    Returns:
        The parsed records, one dictionary per JSON object.

    Raises:
        ValueError:
            If strict=True and a line contains invalid JSON or a non-object
            value.
    """
    records: list[dict[str, object]] = []
    for line_idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        # Recover legacy glued objects without rewriting quoted string content.
        for sub_line in _split_jsonl_objects(line=line):
            if not sub_line.strip():
                continue
            try:
                value = json.loads(sub_line)
            except json.JSONDecodeError as exc:
                if strict:
                    raise ValueError(
                        f"Invalid JSON in {source} line {line_idx:,}: {sub_line}."
                    ) from exc
                logger.warning(
                    "Skipping invalid JSON in %s line %s: %s",
                    source,
                    line_idx,
                    sub_line,
                )
                continue
            if not isinstance(value, dict):
                if strict:
                    raise ValueError(
                        f"Invalid result in {source} line {line_idx:,}: "
                        "expected object."
                    )
                logger.warning(
                    "Skipping non-object value in %s line %s", source, line_idx
                )
                continue
            records.append(value)
    return records


def _split_jsonl_objects(line: str) -> list[str]:
    """Split glued objects without rewriting JSON string values.

    Args:
        line:
            One physical JSONL line, possibly containing glued objects.

    Returns:
        Object fragments separated outside quoted JSON strings.
    """
    fragments: list[str] = []
    start = 0
    in_string = False
    escaped = False
    for index, character in enumerate(line):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029":
            fragments.append(line[start:index])
            start = index + 1
        elif character == "}" and line[index + 1 : index + 2] == "{":
            fragments.append(line[start : index + 1])
            start = index + 1
    fragments.append(line[start:])
    return fragments
