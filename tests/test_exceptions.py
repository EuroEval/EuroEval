"""Unit tests for the `exceptions` module."""

import inspect

from euroeval import exceptions


def test_all_classes_are_exceptions() -> None:
    """Test that all classes in `exceptions` are exceptions."""
    all_classes = [
        getattr(exceptions, obj_name)
        for obj_name in dir(exceptions)
        if not obj_name.startswith("_")
        and inspect.isclass(object=getattr(exceptions, obj_name))
    ]
    for obj in all_classes:
        assert issubclass(obj, Exception), f"Class {obj.__name__} is not an exception."
