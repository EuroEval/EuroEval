"""All Bosnian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import BOSNIAN
from ..tasks import NER, RC, SENT, SUMM

### Official datasets ###

MMS_BS_CONFIG = DatasetConfig(
    name="mms-bs",
    pretty_name="MMS-bs",
    source="EuroEval/mms-bs-mini",
    task=SENT,
    languages=[BOSNIAN],
)

WIKIANN_BS_CONFIG = DatasetConfig(
    name="wikiann-bs",
    pretty_name="WikiANN-bs",
    source="EuroEval/wikiann-bs-mini",
    task=NER,
    languages=[BOSNIAN],
)

MULTI_WIKI_QA_BS_CONFIG = DatasetConfig(
    name="multi-wiki-qa-bs",
    pretty_name="MultiWikiQA-bs",
    source="EuroEval/multi-wiki-qa-bs-mini",
    task=RC,
    languages=[BOSNIAN],
)

LR_SUM_BS_CONFIG = DatasetConfig(
    name="lr-sum-bs",
    pretty_name="LRSum-bs",
    source="EuroEval/lr-sum-bs-mini",
    task=SUMM,
    languages=[BOSNIAN],
)
