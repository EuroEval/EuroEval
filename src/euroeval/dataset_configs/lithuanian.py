"""All Lithuanian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import LITHUANIAN
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

LITHUANIAN_EMOTIONS_CONFIG = DatasetConfig(
    name="lithuanian-emotions",
    pretty_name="Lithuanian Emotions",
    source="EuroEval/lithuanian-emotions-mini",
    task=SENT,
    languages=[LITHUANIAN],
)

SCALA_LT_CONFIG = DatasetConfig(
    name="scala-lt",
    pretty_name="ScaLA-lt",
    source="EuroEval/scala-lt",
    task=LA,
    languages=[LITHUANIAN],
)

WIKIANN_LT_CONFIG = DatasetConfig(
    name="wikiann-lt",
    pretty_name="WikiANN-lt",
    source="EuroEval/wikiann-lt-mini",
    task=NER,
    languages=[LITHUANIAN],
)

MULTI_WIKI_QA_LT_CONFIG = DatasetConfig(
    name="multi-wiki-qa-lt",
    pretty_name="MultiWikiQA-lt",
    source="EuroEval/multi-wiki-qa-lt-mini",
    task=RC,
    languages=[LITHUANIAN],
)

LRYTAS_CONFIG = DatasetConfig(
    name="lrytas",
    pretty_name="Lrytas",
    source="EuroEval/lrytas-mini",
    task=SUMM,
    languages=[LITHUANIAN],
)

LT_HISTORY_CONFIG = DatasetConfig(
    name="lt-history",
    pretty_name="LT-History",
    source="EuroEval/lt-history",
    task=KNOW,
    languages=[LITHUANIAN],
)

WINOGRANDE_LT_CONFIG = DatasetConfig(
    name="winogrande-lt",
    pretty_name="Winogrande-lt",
    source="EuroEval/winogrande-lt",
    task=COMMON_SENSE,
    languages=[LITHUANIAN],
    _labels=["a", "b"],
)
