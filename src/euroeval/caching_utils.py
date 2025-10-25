"""Caching utility functions."""

import typing as t
from functools import wraps

from .constants import T


def cache_arguments(
    *arguments: str, disable_condition: t.Callable[[], bool] = lambda: False
) -> t.Callable[[t.Callable[..., T]], t.Callable[..., T]]:
    """Cache specified arguments of a function.

    Args:
        arguments:
            The list of argument names to cache. If empty, all arguments are cached.
        disable_condition:
            A function that checks if cache should be disabled.

    Returns:
        A decorator that caches the specified arguments of a function.
    """

    def caching_decorator(func: t.Callable[..., T]) -> t.Callable[..., T]:
        """Decorator that caches the specified arguments of a function.

        Args:
            func:
                The function to decorate.

        Returns:
            The decorated function.
        """
        cache: dict[tuple, T] = dict()

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            """Wrapper function that caches the specified arguments.

            Args:
                *args:
                    The positional arguments to the function.
                **kwargs:
                    The keyword arguments to the function.

            Returns:
                The result of the function.

            Raises:
                ValueError:
                    If an argument name is not found in the function parameters.
            """
            if not arguments:
                key = args + tuple(kwargs[k] for k in sorted(kwargs.keys()))
            else:
                func_params = func.__code__.co_varnames
                key_items: list[t.Any] = list()
                for arg_name in arguments:
                    if arg_name in kwargs:
                        key_items.append(kwargs[arg_name])
                    else:
                        try:
                            arg_index = func_params.index(arg_name)
                            key_items.append(args[arg_index])
                        except (ValueError, IndexError):
                            raise ValueError(
                                f"Argument {arg_name} not found in function "
                                f"{func.__name__} parameters."
                            )
                key = tuple(key_items)

            # Do not cache if the condition is met
            if key not in cache or disable_condition():
                cache[key] = func(*args, **kwargs)
            return cache[key]

        return wrapper

    return caching_decorator
