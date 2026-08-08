"""All Western Frisian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import WESTERN_FRISIAN
from ..tasks import LOGIC

# Official datasets ###

ZEBRA_PUZZLE_EASY_FY_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-fy",
    pretty_name="ZebraPuzzlesEasy-fy",
    source="EuroEval/zebra-puzzles-easy-fy",
    task=LOGIC,
    languages=[WESTERN_FRISIAN],
)

# Unofficial datasets ###

ZEBRA_PUZZLE_HARD_FY_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-fy",
    pretty_name="ZebraPuzzlesHard-fy",
    source="EuroEval/zebra-puzzles-hard-fy",
    task=LOGIC,
    languages=[WESTERN_FRISIAN],
    unofficial=True,
)
