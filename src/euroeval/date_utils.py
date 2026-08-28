"""Utilities for normalising date metadata."""

import datetime


def normalise_release_date(value: object) -> str | None:
    """Normalise a model release date.

    Args:
        value:
            The release date value to normalise.

    Returns:
        An ISO-formatted date, or None when the value is absent or malformed.
    """
    if isinstance(value, datetime.datetime):
        return value.date().isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    if not isinstance(value, str):
        return None
    try:
        normalised = datetime.date.fromisoformat(value).isoformat()
    except ValueError:
        return None
    return normalised if value == normalised else None
