"""All Slovak dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import SLOVAK
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT

### Official datasets ###

CSFD_SENTIMENT_SK_CONFIG = DatasetConfig(
    name="csfd-sentiment-sk",
    source="EuroEval/csfd-sentiment-sk-mini",
    task=SENT,
    languages=[SLOVAK],
)

SCALA_SK_CONFIG = DatasetConfig(
    name="scala-sk", source="EuroEval/scala-sk", task=LA, languages=[SLOVAK]
)

UNER_SK_CONFIG = DatasetConfig(
    name="uner-sk", source="EuroEval/uner-sk-mini", task=NER, languages=[SLOVAK]
)

MULTI_WIKI_QA_SK_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sk",
    source="EuroEval/multi-wiki-qa-sk-mini",
    task=RC,
    languages=[SLOVAK],
)

MMLU_SK_CONFIG = DatasetConfig(
    name="mmlu-sk", source="EuroEval/mmlu-sk-mini", task=KNOW, languages=[SLOVAK]
)

WINOGRANDE_SK_CONFIG = DatasetConfig(
    name="winogrande-sk",
    source="EuroEval/winogrande-sk",
    task=COMMON_SENSE,
    languages=[SLOVAK],
)
