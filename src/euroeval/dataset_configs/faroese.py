"""All Faroese dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import FAROESE
from ..tasks import GED, LA, LOGIC, NER, RC, SENT

# Official datasets ###

FOSENT_CONFIG = DatasetConfig(
    name="fosent",
    pretty_name="FoSent",
    source="EuroEval/fosent",
    task=SENT,
    languages=[FAROESE],
    num_few_shot_examples=5,
)

SCALA_FO_CONFIG = DatasetConfig(
    name="scala-fo",
    pretty_name="ScaLA-fo",
    source="EuroEval/scala-fo",
    task=LA,
    languages=[FAROESE],
)

FONE_CONFIG = DatasetConfig(
    name="fone",
    pretty_name="FoNE",
    source="EuroEval/fone-mini",
    task=NER,
    languages=[FAROESE],
)

FOQA_CONFIG = DatasetConfig(
    name="foqa",
    pretty_name="FoQA",
    source="EuroEval/foqa",
    task=RC,
    languages=[FAROESE],
)


# Unofficial datasets ###

WIKIANN_FO_CONFIG = DatasetConfig(
    name="wikiann-fo",
    pretty_name="WikiANN-fo",
    source="EuroEval/wikiann-fo-mini",
    task=NER,
    languages=[FAROESE],
    unofficial=True,
)

MULTI_WIKI_QA_FO_CONFIG = DatasetConfig(
    name="multi-wiki-qa-fo",
    pretty_name="MultiWikiQA-fo",
    source="EuroEval/multi-wiki-qa-fo-mini",
    task=RC,
    languages=[FAROESE],
    unofficial=True,
)

GERLANGMOD_FO_CONFIG = DatasetConfig(
    name="gerlangmod-fo",
    pretty_name="GerLangMod-fo",
    source="EuroEval/gerlangmod-fo",
    task=GED,
    languages=[FAROESE],
    unofficial=True,
)

ZEBRA_PUZZLE_EASY_FO_CONFIG = DatasetConfig(
    name="zebra-puzzle-easy-fo",
    pretty_name="ZebraPuzzleEasy-fo",
    source="EuroEval/zebra-puzzle-easy-fo",
    task=LOGIC,
    languages=[FAROESE],
    unofficial=True,
)

ZEBRA_PUZZLE_HARD_FO_CONFIG = DatasetConfig(
    name="zebra-puzzle-hard-fo",
    pretty_name="ZebraPuzzleHard-fo",
    source="EuroEval/zebra-puzzle-hard-fo",
    task=LOGIC,
    languages=[FAROESE],
    unofficial=True,
)
