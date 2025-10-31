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

SSJ500K_NER_CONFIG = DatasetConfig(
    name="ssj500k-ner",
    pretty_name="ssj500k-NER",
    source="EuroEval/ssj500k-ner-mini",
    task=NER,
    languages=[CROATIAN],
)

MULTI_WIKI_QA_SL_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sl",
    pretty_name="MultiWikiQA-sl",
    source="EuroEval/multi-wiki-qa-sl-mini",
    task=RC,
    languages=[CROATIAN],
)

MMLU_SL_CONFIG = DatasetConfig(
    name="mmlu-sl",
    pretty_name="MMLU-sl",
    source="EuroEval/mmlu-sl-mini",
    task=KNOW,
    languages=[CROATIAN],
)

WINOGRANDE_SL_CONFIG = DatasetConfig(
    name="winogrande-sl",
    pretty_name="Winogrande-sl",
    source="EuroEval/winogrande-sl",
    task=COMMON_SENSE,
    languages=[CROATIAN],
    _labels=["a", "b"],
)
