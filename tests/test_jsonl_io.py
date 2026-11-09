"""Tests for shared JSONL parsing without modifying quoted string contents."""

import json

import pytest

from euroeval.jsonl_io import parse_jsonl_lines


def test_error_reports_original_input_line() -> None:
    """Keep source-line diagnostics for malformed input."""
    with pytest.raises(ValueError, match="Invalid JSON in synthetic line 2:"):
        parse_jsonl_lines(lines=["", "not JSON"], source="synthetic")


def test_lenient_mode_keeps_good_records_around_invalid_input() -> None:
    """Keep valid records without accepting malformed JSON or nonobjects."""
    records = parse_jsonl_lines(
        lines=['{"first":1}', "invalid", "[]", '{"last":2}'],
        source="synthetic",
        strict=False,
    )

    assert records == [{"first": 1}, {"last": 2}]


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "text",
    [
        "left}{right",
        'quote " then }{',
        "slash \\ then }{",
        "x\u0085y",
        "x\u2028y",
        "x\u2029y",
    ],
)
def test_preserves_quoted_strings(text: str, strict: bool) -> None:
    """Preserve valid JSON strings in keys and nested values.

    Args:
        text:
            JSON string content that is not a record boundary.
        strict:
            Whether malformed input should raise an exception.
    """
    record = {text: {"values": [text, {"text": text}]}}
    line = json.dumps(record, ensure_ascii=False)

    assert parse_jsonl_lines(lines=[line], source="synthetic", strict=strict) == [
        record
    ]


@pytest.mark.parametrize("text", ["}{", 'quoted " then \\}{', "x\u2028y"])
def test_recovers_glued_objects_with_quoted_boundaries(text: str) -> None:
    """Keep glued-object recovery while preserving each object's strings.

    Args:
        text:
            String content occurring in both glued objects.
    """
    records = [{"first": text}, {"second": text}]
    line = "".join(json.dumps(record, ensure_ascii=False) for record in records)

    assert parse_jsonl_lines(lines=[line], source="synthetic") == records


@pytest.mark.parametrize(
    "separator", ["\n", "\r\n", "\r", "\u0085", "\u2028", "\u2029"]
)
def test_retains_unquoted_line_separator_handling(separator: str) -> None:
    """Retain legacy physical separator handling outside strings.

    Args:
        separator:
            A separator recognized by the previous parser.
    """
    line = '{"first": 1}' + separator + '{"second": 2}'

    assert parse_jsonl_lines(lines=[line], source="synthetic") == [
        {"first": 1},
        {"second": 2},
    ]
