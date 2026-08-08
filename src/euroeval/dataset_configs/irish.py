"""All Irish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import IRISH
from ..tasks import LOGIC

# Official datasets ###

ZEBRA_PUZZLE_EASY_GA_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-ga",
    pretty_name="ZebraPuzzlesEasy-ga",
    source="EuroEval/zebra-puzzles-easy-ga",
    task=LOGIC,
    languages=[IRISH],
)

# Unofficial datasets ###

ZEBRA_PUZZLE_HARD_GA_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-ga",
    pretty_name="ZebraPuzzlesHard-ga",
    source="EuroEval/zebra-puzzles-hard-ga",
    task=LOGIC,
    languages=[IRISH],
    unofficial=True,
)
