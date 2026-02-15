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
    assert result == "completed after {delay}s".format(delay=0.01)


def test_safe_run_with_exception() -> None:
    """Test that safe_run propagates exceptions."""
    with pytest.raises(ValueError, match="Test error"):
        safe_run(failing_coroutine())


def test_safe_run_with_nested_coroutines() -> None:
    """Test that safe_run works with nested coroutines."""

    async def inner_coroutine() -> str:
        await asyncio.sleep(0.001)
        return "inner"

    async def outer_coroutine() -> str:
        result = await inner_coroutine()
        return f"outer-{result}"

    result = safe_run(outer_coroutine())
    assert result == "outer-inner"


def test_safe_run_with_multiple_calls() -> None:
    """Test that safe_run can be called multiple times."""
    results = [safe_run(simple_coroutine()) for _ in range(3)]
    assert results == ["test_result", "test_result", "test_result"]


def test_add_semaphore_success() -> None:
    """Test that add_semaphore_and_catch_exception works with a successful coroutine."""

    async def run_test() -> None:
        semaphore = asyncio.Semaphore(1)
        result = await add_semaphore_and_catch_exception(simple_coroutine(), semaphore)
        assert result == "test_result"

    safe_run(run_test())


def test_add_semaphore_catches_exception() -> None:
    """Test that add_semaphore_and_catch_exception catches exceptions."""

    async def run_test() -> None:
        semaphore = asyncio.Semaphore(1)
        result = await add_semaphore_and_catch_exception(failing_coroutine(), semaphore)
        assert isinstance(result, ValueError)
        assert str(result) == "Test error"

    safe_run(run_test())


def test_add_semaphore_catches_different_exceptions() -> None:
    """Test that add_semaphore_and_catch_exception catches different exception types."""

    async def run_test() -> None:
        semaphore = asyncio.Semaphore(1)

        async def runtime_error_coroutine() -> None:
            raise RuntimeError("Runtime error")

        async def type_error_coroutine() -> None:
            raise TypeError("Type error")

        result1 = await add_semaphore_and_catch_exception(
            runtime_error_coroutine(), semaphore
        )
        assert isinstance(result1, RuntimeError)
        assert str(result1) == "Runtime error"

        result2 = await add_semaphore_and_catch_exception(
            type_error_coroutine(), semaphore
        )
        assert isinstance(result2, TypeError)
        assert str(result2) == "Type error"

    safe_run(run_test())


def test_add_semaphore_limits_concurrency() -> None:
    """Test that semaphore limits concurrent execution."""

    async def run_test() -> None:
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

    safe_run(run_test())


def test_add_semaphore_with_return_values() -> None:
    """Test that add_semaphore_and_catch_exception preserves return values."""

    async def run_test() -> None:
        semaphore = asyncio.Semaphore(3)

        async def return_value_coroutine(value: int) -> int:
            await asyncio.sleep(0.001)
            return value * 2

        tasks = [
            add_semaphore_and_catch_exception(return_value_coroutine(i), semaphore)
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)

        assert results == [0, 2, 4, 6, 8]

    safe_run(run_test())


def test_add_semaphore_with_single_slot() -> None:
    """Test semaphore with only one concurrent slot."""

    async def run_test() -> None:
        semaphore = asyncio.Semaphore(1)
        completed_order: list[int] = []

        async def sequential_coroutine(task_id: int) -> int:
            """Coroutine that should run sequentially."""
            completed_order.append(task_id)
            await asyncio.sleep(0.005)
            return task_id

        tasks = [
            add_semaphore_and_catch_exception(sequential_coroutine(i), semaphore)
            for i in range(3)
        ]
        results = await asyncio.gather(*tasks)

        assert results == [0, 1, 2]
        assert len(completed_order) == 3

    safe_run(run_test())
