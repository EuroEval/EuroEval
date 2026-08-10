"""All Basque dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import BASQUE
from ..tasks import LOGIC

# Official datasets ###

ZEBRA_PUZZLE_EASY_EU_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-eu",
    pretty_name="ZebraPuzzlesEasy-eu",
    source="EuroEval/zebra-puzzles-easy-eu",
    task=LOGIC,
    languages=[BASQUE],
)

# Unofficial datasets ###

ZEBRA_PUZZLE_HARD_EU_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-eu",
    pretty_name="ZebraPuzzlesHard-eu",
    source="EuroEval/zebra-puzzles-hard-eu",
    task=LOGIC,
    languages=[BASQUE],
    unofficial=True,
)
