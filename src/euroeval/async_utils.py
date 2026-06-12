"""Utility functions for asyncronous tasks."""

import asyncio
import typing as t

from .constants import T


def safe_run(coroutine: t.Coroutine[t.Any, t.Any, T]) -> T:
    """Run a coroutine, ensuring that the event loop is always closed when we're done.

    Args:
        coroutine:
            The coroutine to run.

    Returns:
        The result of the coroutine.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:  # If the current event loop is closed
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    response = loop.run_until_complete(coroutine)
    return response


async def add_semaphore_and_catch_exception(
    coroutine: t.Coroutine[t.Any, t.Any, T], semaphore: asyncio.Semaphore
) -> T | Exception:
    """Run a coroutine with a semaphore.

    Args:
        coroutine:
            The coroutine to run.
        semaphore:
            The semaphore to use.

    Returns:
        The result of the coroutine.
    """
    async with semaphore:
        try:
            return await coroutine
        except Exception as exc:
            return exc
