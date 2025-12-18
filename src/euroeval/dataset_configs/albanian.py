"""All Albanian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import ALBANIAN
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

MMS_SQ_CONFIG = DatasetConfig(
    name="mms-sq",
    pretty_name="MMS-sq",
    source="EuroEval/mms-sq-mini",
    task=SENT,
    languages=[ALBANIAN],
)

SCALA_SQ_CONFIG = DatasetConfig(
    name="scala-sq",
    pretty_name="ScaLA-sq",
    source="EuroEval/scala-sq",
    task=LA,
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

GLOBAL_MMLU_LITE_SQ_CONFIG = DatasetConfig(
    name="global-mmlu-lite-sq",
    pretty_name="GlobalMMLULite-sq",
    source="EuroEval/global-mmlu-lite-sq",
    task=KNOW,
    languages=[ALBANIAN],
)

WINOGRANDE_SQ_CONFIG = DatasetConfig(
    name="winogrande-sq",
    pretty_name="Winogrande-sq",
    source="EuroEval/winogrande-sq",
    task=COMMON_SENSE,
    languages=[ALBANIAN],
    _labels=["a", "b"],
)
