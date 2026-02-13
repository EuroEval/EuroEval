"""Utilities for setting up IFEval constraints."""

import collections.abc as c
import typing as t
from functools import wraps

from ...exceptions import InvalidBenchmark


class Constraint(t.Protocol):
    """An instruction-following constraint."""

    def __call__(self, response: str, **constraint_kwargs) -> bool:
        """Apply the constraint.

        Args:
            response:
                The output to be checked.
            **constraint_kwargs:
                Extra keyword arguments for the constraint function.

        Returns:
            True if the constraint is satisfied, otherwise False.
        """
        ...


ALL_CONSTRAINTS: dict[str, Constraint] = dict()


def register(
    name: str, **desired_keys_and_types
) -> c.Callable[[Constraint], Constraint]:
    """Decorator that registers a function under the given name.

    Args:
        name:
            The name under which to register the function.
        **desired_keys_and_types:
            The keyword arguments and their types that the function expects.

    Returns:
        The decorator function.
    """

    def decorator(fn: Constraint) -> Constraint:
        """Register the function under the given name.

        Args:
            fn:
                The function to register.

        Returns:
            The function.
        """

        @wraps
        def wrapper(response: str, **kwargs) -> bool:
            """Wrapper function that checks the keyword arguments and their types.

            Args:
                response:
                    The response string to be checked.
                **kwargs:
                    Keyword arguments to be passed to the function.

            Returns:
                The result of the function.

            Raises:
                InvalidBenchmark:
                    If a required keyword argument is missing or if a keyword argument
                    has the wrong type.
            """
            for key, value in desired_keys_and_types.items():
                if key not in kwargs:
                    raise InvalidBenchmark(
                        f"The function {fn.__name__!r} (registered as {name!r}) "
                        f"requires the keyword argument {key!r}."
                    )
                elif key in kwargs and not isinstance(kwargs[key], value):
                    raise InvalidBenchmark(
                        f"The function {fn.__name__!r} (registered as {name!r}) "
                        f"expects the keyword argument {key!r} to be of type "
                        f"{value.__name__!r}, but got {type(kwargs[key]).__name__!r}."
                    )
            return fn(response, **kwargs)

        ALL_CONSTRAINTS[name] = fn
        return fn

    return decorator
