"""All Latvian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import LATVIAN
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

LATVIAN_TWITTER_SENTIMENT_CONFIG = DatasetConfig(
    name="latvian-twitter-sentiment",
    pretty_name="Latvian Twitter Sentiment",
    source="EuroEval/latvian-twitter-sentiment-mini",
    task=SENT,
    languages=[LATVIAN],
)

SCALA_LV_CONFIG = DatasetConfig(
    name="scala-lv",
    pretty_name="ScaLA-lv",
    source="EuroEval/scala-lv",
    task=LA,
    languages=[LATVIAN],
)

FULLSTACK_NER_LV_CONFIG = DatasetConfig(
    name="fullstack-ner-lv",
    pretty_name="FullStack NER-lv",
    source="EuroEval/fullstack-ner-lv-mini",
    task=NER,
    languages=[LATVIAN],
)

MULTI_WIKI_QA_LV_CONFIG = DatasetConfig(
    name="multi-wiki-qa-lv",
    pretty_name="MultiWikiQA-lv",
    source="EuroEval/multi-wiki-qa-lv-mini",
    task=RC,
    languages=[LATVIAN],
)

LSM_CONFIG = DatasetConfig(
    name="lsm",
    pretty_name="LSM",
    source="EuroEval/lsm-mini",
    task=SUMM,
    languages=[LATVIAN],
)


MMLU_LV_CONFIG = DatasetConfig(
    name="mmlu-lv",
    pretty_name="MMLU-lv",
    source="EuroEval/mmlu-lv-mini",
    task=KNOW,
    languages=[LATVIAN],
)

COPA_LV_CONFIG = DatasetConfig(
    name="copa-lv",
    pretty_name="COPA-lv",
    source="EuroEval/copa-lv",
    task=COMMON_SENSE,
    languages=[LATVIAN],
    _labels=["a", "b"],
)


### Unofficial datasets ###

WIKIANN_LV_CONFIG = DatasetConfig(
    name="wikiann-lv",
    pretty_name="WikiANN-lv",
    source="EuroEval/wikiann-lv-mini",
    task=NER,
    languages=[LATVIAN],
    unofficial=True,
)

WINOGRANDE_LV_CONFIG = DatasetConfig(
    name="winogrande-lv",
    pretty_name="Winogrande-lv",
    source="EuroEval/winogrande-lv",
    task=COMMON_SENSE,
    languages=[LATVIAN],
    _labels=["a", "b"],
    unofficial=True,
)
