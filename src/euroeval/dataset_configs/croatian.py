"""All Croatian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import CROATIAN
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT

### Official datasets ###

MMS_HR_CONFIG = DatasetConfig(
    name="mms-hr",
    pretty_name="MMS-hr",
    source="EuroEval/mms-hr-mini",
    task=SENT,
    languages=[CROATIAN],
)

SCALA_HR_CONFIG = DatasetConfig(
    name="scala-hr",
    pretty_name="ScaLA-hr",
    source="EuroEval/scala-hr",
    task=LA,
    languages=[CROATIAN],
)

WIKIANN_HR_CONFIG = DatasetConfig(
    name="wikiann-hr",
    pretty_name="WikiANN-hr",
    source="EuroEval/wikiann-hr-mini",
    task=NER,
    languages=[CROATIAN],
)

MULTI_WIKI_QA_HR_CONFIG = DatasetConfig(
    name="multi-wiki-qa-hr",
    pretty_name="MultiWikiQA-hr",
    source="EuroEval/multi-wiki-qa-hr-mini",
    task=RC,
    languages=[CROATIAN],
)

MMLU_HR_CONFIG = DatasetConfig(
    name="mmlu-hr",
    pretty_name="MMLU-hr",
    source="EuroEval/mmlu-hr-mini",
    task=KNOW,
    languages=[CROATIAN],
)

WINOGRANDE_HR_CONFIG = DatasetConfig(
    name="winogrande-hr",
    pretty_name="Winogrande-hr",
    source="EuroEval/winogrande-hr",
    task=COMMON_SENSE,
    languages=[CROATIAN],
    _labels=["a", "b"],
)
