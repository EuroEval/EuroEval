"""Tests for the Danish similarity outlier dataset creation script."""

import pandas as pd

from scripts.dataset_creation.create_danish_similarity_outlier import (
    GRANULARITY_FILES,
    parse_tsv_file,
    split_dataframe,
)


def _tsv_content(
    candidates: str, outlier_position: str = "0", source_id: str = "v1_0001"
) -> bytes:
    """Build a fine-grained TSV payload for parser tests.

    Returns:
        A UTF-8 encoded TSV payload.
    """
    return (
        "candidates\tlabel   outlier_position    id  chapter section\n"
        f"{candidates}\tword\t{outlier_position}\t{source_id}\t1\t1.015\n"
    ).encode()


def test_parse_tsv_file_retains_source_metadata() -> None:
    """Parsed rows retain their source ID, granularity, and source location."""
    records = parse_tsv_file(
        file_name="outlier_similarity_fine.tsv",
        content=_tsv_content("['a', 'b', 'c', 'd', 'e', 'f']"),
    )

    assert len(records) == 1
    assert records[0]["source_id"] == "v1_0001"
    assert records[0]["granularity"] == "fine"
    assert records[0]["chapter"] == "1"
    assert records[0]["section"] == "1.015"
    assert records[0]["label"] == "a"


def test_parse_tsv_file_skips_malformed_and_duplicate_options() -> None:
    """Malformed options and outlier positions do not create dataset rows."""
    header = "candidates\tlabel   outlier_position    id  chapter section\n"
    content = header + "\n".join(
        (
            "['a', 'b', 'c', 'd', 'e', 'f']\tword\t0\tv1_0001\t1\t1.015",
            "['a', 'b', 'c', 'd', 'e', 'a']\tword\t0\tv1_0002\t1\t1.015",
            "not a list\tword\t0\tv1_0003\t1\t1.015",
            "['a', 'b', 'c', 'd', 'e', 'f']\tword\t6\tv1_0004\t1\t1.015",
        )
    )

    records = parse_tsv_file(
        file_name="outlier_similarity_fine.tsv", content=content.encode()
    )

    assert [record["source_id"] for record in records] == ["v1_0001"]


def test_split_dataframe_is_deterministic_and_group_safe() -> None:
    """Every source ID stays in one split and repeated calls are identical."""
    rows = [
        {
            "source_id": f"v1_{source_index:04}",
            "granularity": granularity,
            "text": f"text-{source_index}-{granularity}",
            "label": "a",
        }
        for source_index in range(1_111)
        for granularity in GRANULARITY_FILES
    ]
    frame = pd.DataFrame(rows)

    first = split_dataframe(df=frame)
    second = split_dataframe(df=frame)

    assert {name: len(split) for name, split in first.items()} == {
        "train": 1_023,
        "val": 255,
        "test": 2_046,
    }
    for name in first:
        assert first[name].equals(second[name])

    source_splits = {
        source_id: name
        for name, split in first.items()
        for source_id in split["source_id"].unique()
    }
    source_assignments = [
        (source_id, name)
        for name, split in first.items()
        for source_id in split["source_id"].unique()
    ]
    assert sum(len(split) for split in first.values()) == len(source_splits) * 3
    assert len(source_splits) == 1_108
    assert len(source_assignments) == len(source_splits)
    for split in first.values():
        assert set(split.groupby("source_id").size()) == {3}
    assert set(source_splits) == {
        source_id
        for split in first.values()
        for source_id in split["source_id"].unique()
    }
