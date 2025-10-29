"""All Slovenian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import SLOVENIAN
from ..tasks import COMMON_SENSE, KNOW, LA, RC, SENT

### Official datasets ###

CSFD_SENTIMENT_SK_CONFIG = DatasetConfig(
    name="sentinews",
    pretty_name="Sentinews-sl",
    source="EuroEval/sentinews-mini",
    task=SENT,
    languages=[SLOVENIAN],
)

SCALA_SL_CONFIG = DatasetConfig(
    name="scala-sl",
    pretty_name="ScaLA-sl",
    source="EuroEval/scala-sl",
    task=LA,
    languages=[SLOVENIAN],
)

MULTI_WIKI_QA_SK_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sk",
    pretty_name="MultiWikiQA-sk",
    source="EuroEval/multi-wiki-qa-sk-mini",
    task=RC,
    languages=[SLOVENIAN],
)

MMLU_SK_CONFIG = DatasetConfig(
    name="mmlu-sk",
    pretty_name="MMLU-sk",
    source="EuroEval/mmlu-sk-mini",
    task=KNOW,
    languages=[SLOVENIAN],
)

WINOGRANDE_SK_CONFIG = DatasetConfig(
    name="winogrande-sk",
    pretty_name="Winogrande-sk",
    source="EuroEval/winogrande-sk",
    task=COMMON_SENSE,
    languages=[SLOVENIAN],
)
