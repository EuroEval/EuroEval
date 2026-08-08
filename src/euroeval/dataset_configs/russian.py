"""All Russian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import RUSSIAN
from ..tasks import LOGIC

# Official datasets ###

ZEBRA_PUZZLE_EASY_RU_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-ru",
    pretty_name="ZebraPuzzlesEasy-ru",
    source="EuroEval/zebra-puzzles-easy-ru",
    task=LOGIC,
    languages=[RUSSIAN],
)

# Unofficial datasets ###

ZEBRA_PUZZLE_HARD_RU_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-ru",
    pretty_name="ZebraPuzzlesHard-ru",
    source="EuroEval/zebra-puzzles-hard-ru",
    task=LOGIC,
    languages=[RUSSIAN],
    unofficial=True,
)
