"""All Scots dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import SCOTS
from ..tasks import LOGIC

# Official datasets ###

ZEBRA_PUZZLE_EASY_SCO_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-sco",
    pretty_name="ZebraPuzzlesEasy-sco",
    source="EuroEval/zebra-puzzles-easy-sco",
    task=LOGIC,
    languages=[SCOTS],
)

# Unofficial datasets ###

ZEBRA_PUZZLE_HARD_SCO_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-sco",
    pretty_name="ZebraPuzzlesHard-sco",
    source="EuroEval/zebra-puzzles-hard-sco",
    task=LOGIC,
    languages=[SCOTS],
    unofficial=True,
)
