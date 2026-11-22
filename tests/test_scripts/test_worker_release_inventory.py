"""Exercise the maintainer inventory CLI without Docker or network access."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from scripts import worker_release_inventory


@pytest.fixture
def inventory_args(tmp_path: Path) -> list[str]:
    """Provide all workflow options with a temporary output destination.

    Args:
        tmp_path: Temporary artifact directory.

    Returns:
        Valid inventory command arguments.
    """
    return [
        "--architecture",
        "amd64",
        "--index-digest",
        "sha256:" + "a" * 64,
        "--source-sha",
        "b" * 40,
        "--run-id",
        "123",
        "--run-attempt",
        "2",
        "--output",
        str(tmp_path / "inventory.json"),
    ]


@pytest.mark.parametrize("architecture", ["amd64", "arm64"])
def test_cli_forwards_options_and_writes_inventory(
    architecture: str, inventory_args: list[str], tmp_path: Path
) -> None:
    """Forward typed workflow options and serialise the public collector result.

    Args:
        architecture: Native candidate architecture.
        inventory_args: Complete workflow command arguments.
        tmp_path: Temporary artifact directory.
    """
    inventory_args[1] = architecture
    expected = {
        "architecture": architecture,
        "index_digest": "sha256:" + "a" * 64,
        "source_sha": "b" * 40,
        "run_id": 123,
        "run_attempt": 2,
    }
    artifact = expected | {
        "inventory": {"python": [["click", "8.1.3"]], "dpkg": ["libc6=2.39"]}
    }
    with patch.object(
        worker_release_inventory, "collect", return_value=artifact
    ) as mock:
        result = CliRunner().invoke(worker_release_inventory.main, inventory_args)
    assert result.exit_code == 0, result.output
    assert result.output == ""
    mock.assert_called_once_with(**expected)
    output = tmp_path / "inventory.json"
    assert json.loads(output.read_text()) == artifact
    assert output.read_text() == json.dumps(artifact, sort_keys=True)


def test_cli_help_lists_workflow_options(inventory_args: list[str]) -> None:
    """Expose the public workflow options without collecting an inventory.

    Args:
        inventory_args: Complete workflow command arguments.
    """
    with patch.object(worker_release_inventory, "collect") as mock:
        result = CliRunner().invoke(worker_release_inventory.main, ["--help"])
    assert result.exit_code == 0
    for option in inventory_args[::2]:
        assert option in result.output
    mock.assert_not_called()


@pytest.mark.parametrize("option", ["--run-id", "--run-attempt"])
def test_cli_rejects_noninteger_run_identity(
    option: str, inventory_args: list[str], tmp_path: Path
) -> None:
    """Reject invalid numeric identities without invoking the collector.

    Args:
        option: Integer option to invalidate.
        inventory_args: Complete workflow command arguments.
        tmp_path: Temporary artifact directory.
    """
    inventory_args[inventory_args.index(option) + 1] = "not-an-integer"
    with patch.object(worker_release_inventory, "collect") as mock:
        result = CliRunner().invoke(worker_release_inventory.main, inventory_args)
    assert result.exit_code == 2
    assert f"Invalid value for '{option}'" in result.output
    mock.assert_not_called()
    assert not (tmp_path / "inventory.json").exists()


def test_cli_rejects_unknown_option(inventory_args: list[str], tmp_path: Path) -> None:
    """Reject misspelled options without invoking the collector.

    Args:
        inventory_args: Complete workflow command arguments.
        tmp_path: Temporary artifact directory.
    """
    with patch.object(worker_release_inventory, "collect") as mock:
        result = CliRunner().invoke(
            worker_release_inventory.main, [*inventory_args, "--unknown"]
        )
    assert result.exit_code == 2
    assert "No such option: --unknown" in result.output
    mock.assert_not_called()
    assert not (tmp_path / "inventory.json").exists()


@pytest.mark.parametrize(
    "option",
    [
        "--architecture",
        "--index-digest",
        "--source-sha",
        "--run-id",
        "--run-attempt",
        "--output",
    ],
)
def test_cli_requires_workflow_options(
    option: str, inventory_args: list[str], tmp_path: Path
) -> None:
    """Reject each missing workflow option before collecting an inventory.

    Args:
        option: Required option to omit.
        inventory_args: Complete workflow command arguments.
        tmp_path: Temporary artifact directory.
    """
    index = inventory_args.index(option)
    del inventory_args[index : index + 2]
    with patch.object(worker_release_inventory, "collect") as mock:
        result = CliRunner().invoke(worker_release_inventory.main, inventory_args)
    assert result.exit_code == 2
    assert f"Missing option '{option}'" in result.output
    mock.assert_not_called()
    assert not (tmp_path / "inventory.json").exists()
