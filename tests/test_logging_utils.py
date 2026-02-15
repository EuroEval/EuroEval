"""Tests for the `logging_utils` module."""

import logging
import sys

import pytest
from tqdm.auto import tqdm

from euroeval.logging_utils import (
    adjust_logging_level,
    get_pbar,
    log,
    log_once,
    no_terminal_output,
)


def test_get_pbar_basic() -> None:
    """Test basic progress bar creation."""
    pbar = get_pbar(total=100, desc="Test")
    assert isinstance(pbar, tqdm)
    assert pbar.desc is not None
    pbar.close()


def test_get_pbar_custom_args() -> None:
    """Test progress bar with custom arguments."""
    pbar = get_pbar(total=50, desc="Custom", disable=True)
    assert isinstance(pbar, tqdm)
    assert pbar.disable is True
    pbar.close()


def test_log_debug(caplog: pytest.LogCaptureFixture) -> None:
    """Test debug level logging."""
    with caplog.at_level(logging.DEBUG):
        log("Debug message", level=logging.DEBUG)
    assert "Debug message" in caplog.text
    assert "DEBUG" in caplog.text


def test_log_info(caplog: pytest.LogCaptureFixture) -> None:
    """Test info level logging."""
    with caplog.at_level(logging.INFO):
        log("Info message", level=logging.INFO)
    assert "Info message" in caplog.text


def test_log_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Test warning level logging."""
    with caplog.at_level(logging.WARNING):
        log("Warning message", level=logging.WARNING)
    assert "Warning message" in caplog.text


def test_log_error(caplog: pytest.LogCaptureFixture) -> None:
    """Test error level logging."""
    with caplog.at_level(logging.ERROR):
        log("Error message", level=logging.ERROR)
    assert "Error message" in caplog.text


def test_log_with_colour(caplog: pytest.LogCaptureFixture) -> None:
    """Test logging with custom colour."""
    with caplog.at_level(logging.INFO):
        log("Coloured message", level=logging.INFO, colour="green")
    assert "Coloured message" in caplog.text


def test_log_invalid_level() -> None:
    """Test that invalid logging level raises ValueError."""
    with pytest.raises(ValueError, match="Invalid logging level"):
        log("Message", level=999)


def test_log_once_single_call(caplog: pytest.LogCaptureFixture) -> None:
    """Test that log_once only logs a message once."""
    with caplog.at_level(logging.INFO):
        log_once("Unique message", level=logging.INFO)
        log_once("Unique message", level=logging.INFO)
        log_once("Unique message", level=logging.INFO)

    # Should only appear once
    assert caplog.text.count("Unique message") == 1


def test_log_once_different_messages(caplog: pytest.LogCaptureFixture) -> None:
    """Test that log_once logs different messages separately."""
    with caplog.at_level(logging.INFO):
        log_once("Message 1", level=logging.INFO)
        log_once("Message 2", level=logging.INFO)
        log_once("Message 1", level=logging.INFO)  # Should not log again

    assert "Message 1" in caplog.text
    assert "Message 2" in caplog.text
    assert caplog.text.count("Message 1") == 1
    assert caplog.text.count("Message 2") == 1


def test_log_once_with_prefix(caplog: pytest.LogCaptureFixture) -> None:
    """Test log_once with a prefix."""
    with caplog.at_level(logging.INFO):
        log_once("Test message", level=logging.INFO, prefix="PREFIX: ")

    assert "PREFIX: Test message" in caplog.text


def test_no_terminal_output_context_manager() -> None:
    """Test the no_terminal_output context manager."""
    with no_terminal_output(disable=True):
        # When disabled, output should pass through normally
        print("Test output")


def test_no_terminal_output_disabled() -> None:
    """Test no_terminal_output when disabled."""
    # Just test that it doesn't raise errors
    with no_terminal_output(disable=True):
        pass


def test_adjust_logging_level_verbose() -> None:
    """Test adjust_logging_level with verbose=True."""
    level = adjust_logging_level(verbose=True, ignore_testing=True)
    assert level == logging.DEBUG


def test_adjust_logging_level_not_verbose() -> None:
    """Test adjust_logging_level with verbose=False."""
    level = adjust_logging_level(verbose=False, ignore_testing=True)
    assert level == logging.INFO


def test_adjust_logging_level_during_testing() -> None:
    """Test adjust_logging_level when called from test."""
    # The _called_from_test flag is set in conftest.py
    if hasattr(sys, "_called_from_test"):
        level = adjust_logging_level(verbose=True, ignore_testing=False)
        assert level == logging.CRITICAL


@pytest.mark.parametrize(
    "verbose,ignore_testing,expected",
    [(True, True, logging.DEBUG), (False, True, logging.INFO)],
    ids=["verbose_debug", "not_verbose_info"],
)
def test_adjust_logging_level_parametrized(
    verbose: bool, ignore_testing: bool, expected: int
) -> None:
    """Parametrized test for adjust_logging_level."""
    level = adjust_logging_level(verbose=verbose, ignore_testing=ignore_testing)
    assert level == expected
