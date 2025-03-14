"""Tests for the `enums` module."""

import enum
import inspect

from euroeval import enums


def test_all_classes_are_enums() -> None:
    """Test that all classes in `enums` are Enums."""
    all_classes = [
        getattr(enums, obj_name)
        for obj_name in dir(enums)
        if not obj_name.startswith("_")
        and inspect.isclass(object=getattr(enums, obj_name))
        and not hasattr(enum, obj_name)
    ]
    for obj in all_classes:
        assert issubclass(obj, enum.Enum)
