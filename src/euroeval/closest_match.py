"""Functions related to finding the closest match to a given string."""

import collections.abc as c

from Levenshtein import distance


def get_closest_match(
    string: str,
    options: c.Iterable[str],
    case_sensitive: bool,
    insertion_weight: int = 1,
    deletion_weight: int = 1,
    substitution_weight: int = 1,
) -> tuple[str, int]:
    """Find the closest match to a given string.

    Args:
        string:
            The string to match.
        options:
            The options to match against.
        case_sensitive:
            Whether to match case sensitively.
        insertion_weight:
            The weight of insertions in the Levenshtein distance.
        deletion_weight:
            The weight of deletions in the Levenshtein distance.
        substitution_weight:
            The weight of substitutions in the Levenshtein distance.

    Returns:
        A pair (closest_match, distance) where closest_match is the closest match to the
        given string and distance is the Levenshtein distance between the two.
    """
    distances = {
        option: distance(
            s1=string.lower() if not case_sensitive else string,
            s2=option.lower() if not case_sensitive else option,
            weights=(insertion_weight, deletion_weight, substitution_weight),
        )
        for option in options
    }
    closest_match = min(distances, key=lambda k: distances[k])
    return closest_match, distances[closest_match]
