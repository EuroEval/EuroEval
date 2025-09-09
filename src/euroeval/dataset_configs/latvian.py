"""All Latvian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import LV
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

LATVIAN_TWITTER_SENTIMENT_CONFIG = DatasetConfig(
    name="latvian-twitter-sentiment",
    pretty_name="the truncated version of the Latvian sentiment classification dataset",
    huggingface_id="EuroEval/latvian-twitter-sentiment-mini",
    task=SENT,
    languages=[LV],
)

SCALA_LV_CONFIG = DatasetConfig(
    name="scala-lv",
    pretty_name="the Latvian part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-lv",
    task=LA,
    languages=[LV],
)

FULLSTACK_NER_LV_CONFIG = DatasetConfig(
    name="fullstack-ner-lv",
    pretty_name="the truncated version of the FullStack NER dataset",
    huggingface_id="EuroEval/fullstack-ner-lv-mini",
    task=NER,
    languages=[LV],
)

MULTI_WIKI_QA_LV_CONFIG = DatasetConfig(
    name="multi-wiki-qa-lv",
    pretty_name="the truncated version of the Latvian part of the reading "
    "comprehension dataset MultiWikiQA",
    huggingface_id="EuroEval/multi-wiki-qa-lv-mini",
    task=RC,
    languages=[LV],
)

LSM_CONFIG = DatasetConfig(
    name="lsm",
    pretty_name="the truncated version of the Latvian summarisation dataset LSM",
    huggingface_id="EuroEval/lsm-mini",
    task=SUMM,
    languages=[LV],
)


MMLU_LV_CONFIG = DatasetConfig(
    name="mmlu-lv",
    pretty_name="the truncated version of the Latvian knowledge dataset MMLU-lv, "
    "translated from the English MMLU dataset",
    huggingface_id="EuroEval/mmlu-lv-mini",
    task=KNOW,
    languages=[LV],
)

COPA_LV_CONFIG = DatasetConfig(
    name="copa-lv",
    pretty_name="the Latvian common-sense reasoning dataset COPA-lv, translated from "
    "the English COPA dataset",
    huggingface_id="EuroEval/copa-lv",
    task=COMMON_SENSE,
    languages=[LV],
    _labels=["a", "b"],
)


### Unofficial datasets ###

WIKIANN_LV_CONFIG = DatasetConfig(
    name="wikiann-lv",
    pretty_name="the truncated version of the Latvian part of the named entity "
    "recognition dataset WikiANN",
    huggingface_id="EuroEval/wikiann-lv-mini",
    task=NER,
    languages=[LV],
    unofficial=True,
)
