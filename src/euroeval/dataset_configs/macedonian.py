"""All Macedonian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import MACEDONIAN
from ..tasks import LOGIC

# Official datasets ###

ZEBRA_PUZZLE_EASY_MK_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-mk",
    pretty_name="ZebraPuzzlesEasy-mk",
    source="EuroEval/zebra-puzzles-easy-mk",
    task=LOGIC,
    languages=[MACEDONIAN],
)

# Unofficial datasets ###

ZEBRA_PUZZLE_HARD_MK_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-mk",
    pretty_name="ZebraPuzzlesHard-mk",
    source="EuroEval/zebra-puzzles-hard-mk",
    task=LOGIC,
    languages=[MACEDONIAN],
    unofficial=True,
)
