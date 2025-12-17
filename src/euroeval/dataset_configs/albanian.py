"""All Bosnian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import ALBANIAN
from ..tasks import NER, RC, SENT, SUMM

### Official datasets ###

MMS_SQ_CONFIG = DatasetConfig(
    name="mms-sq",
    pretty_name="MMS-sq",
    source="EuroEval/mms-sq-mini",
    task=SENT,
    languages=[ALBANIAN],
)

WIKIANN_SQ_CONFIG = DatasetConfig(
    name="wikiann-sq",
    pretty_name="WikiANN-sq",
    source="EuroEval/wikiann-sq-mini",
    task=NER,
    languages=[ALBANIAN],
)

MULTI_WIKI_QA_SQ_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sq",
    pretty_name="MultiWikiQA-sq",
    source="EuroEval/multi-wiki-qa-sq-mini",
    task=RC,
    languages=[ALBANIAN],
)

LR_SUM_SQ_CONFIG = DatasetConfig(
    name="lr-sum-sq",
    pretty_name="LRSum-sq",
    source="EuroEval/lr-sum-sq-mini",
    task=SUMM,
    languages=[ALBANIAN],
)
