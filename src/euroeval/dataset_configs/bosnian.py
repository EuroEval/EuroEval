"""All Croatian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import BOSNIAN
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT

### Official datasets ###

MMS_BS_CONFIG = DatasetConfig(
    name="mms-bs",
    pretty_name="MMS-bs",
    source="EuroEval/mms-bs-mini",
    task=SENT,
    languages=[BOSNIAN],
)

SCALA_HR_CONFIG = DatasetConfig(
    name="scala-hr",
    pretty_name="ScaLA-hr",
    source="EuroEval/scala-hr",
    task=LA,
    languages=[BOSNIAN],
)

WIKIANN_HR_CONFIG = DatasetConfig(
    name="wikiann-hr",
    pretty_name="WikiANN-hr",
    source="EuroEval/wikiann-hr-mini",
    task=NER,
    languages=[BOSNIAN],
)

MULTI_WIKI_QA_HR_CONFIG = DatasetConfig(
    name="multi-wiki-qa-hr",
    pretty_name="MultiWikiQA-hr",
    source="EuroEval/multi-wiki-qa-hr-mini",
    task=RC,
    languages=[BOSNIAN],
)

MMLU_HR_CONFIG = DatasetConfig(
    name="mmlu-hr",
    pretty_name="MMLU-hr",
    source="EuroEval/mmlu-hr-mini",
    task=KNOW,
    languages=[BOSNIAN],
)

WINOGRANDE_HR_CONFIG = DatasetConfig(
    name="winogrande-hr",
    pretty_name="Winogrande-hr",
    source="EuroEval/winogrande-hr",
    task=COMMON_SENSE,
    languages=[BOSNIAN],
    _labels=["a", "b"],
)
