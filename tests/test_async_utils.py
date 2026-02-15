"""Tests for the `async_utils` module."""

import asyncio

import pytest

from euroeval.async_utils import add_semaphore_and_catch_exception, safe_run


async def simple_coroutine() -> str:
    """A simple coroutine that returns a string."""
    return "test_result"


async def failing_coroutine() -> None:
    """A coroutine that raises an exception."""
    raise ValueError("Test error")


async def delayed_coroutine(delay: float) -> str:
    """A coroutine that sleeps before returning."""
    await asyncio.sleep(delay)
    return f"completed after {delay}s"


def test_safe_run_simple() -> None:
    """Test that safe_run can execute a simple coroutine."""
    result = safe_run(simple_coroutine())
    assert result == "test_result"


def test_safe_run_with_delay() -> None:
    """Test that safe_run works with delayed coroutines."""
    result = safe_run(delayed_coroutine(0.01))
    assert result == "completed after 0.01s"


def test_safe_run_with_exception() -> None:
    """Test that safe_run propagates exceptions."""
    with pytest.raises(ValueError, match="Test error"):
        safe_run(failing_coroutine())


@pytest.mark.asyncio
async def test_add_semaphore_success() -> None:
    """Test that add_semaphore_and_catch_exception works with a successful coroutine."""
    semaphore = asyncio.Semaphore(1)
    result = await add_semaphore_and_catch_exception(simple_coroutine(), semaphore)
    assert result == "test_result"


@pytest.mark.asyncio
async def test_add_semaphore_catches_exception() -> None:
    """Test that add_semaphore_and_catch_exception catches exceptions."""
    semaphore = asyncio.Semaphore(1)
    result = await add_semaphore_and_catch_exception(failing_coroutine(), semaphore)
    assert isinstance(result, ValueError)
    assert str(result) == "Test error"


@pytest.mark.asyncio
async def test_add_semaphore_limits_concurrency() -> None:
    """Test that semaphore limits concurrent execution."""
    semaphore = asyncio.Semaphore(2)
    execution_order: list[int] = []

    async def tracked_coroutine(task_id: int) -> int:
        """Coroutine that tracks execution order."""
        execution_order.append(task_id)
        await asyncio.sleep(0.01)
        return task_id

    # Create 5 tasks with a semaphore that allows only 2 concurrent executions
    tasks = [
        add_semaphore_and_catch_exception(tracked_coroutine(i), semaphore)
        for i in range(5)
    ]
    results = await asyncio.gather(*tasks)

    # All tasks should complete
    assert results == list(range(5))
    # All tasks should have been tracked
    assert len(execution_order) == 5
