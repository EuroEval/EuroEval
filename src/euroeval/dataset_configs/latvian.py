"""All Latvian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import LV
from ..tasks import LA, NER, RC, SENT

MULTI_WIKI_QA_LV_CONFIG = DatasetConfig(
    name="multi-wiki-qa-lv",
    pretty_name="the truncated version of the Latvian part of the reading "
    "comprehension dataset MultiWikiQA",
    huggingface_id="EuroEval/multi-wiki-qa-lv-mini",
    task=RC,
    languages=[LV],
)


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

WIKIANN_LV_CONFIG = DatasetConfig(
    name="wikiann-lv",
    pretty_name="the truncated version of the Latvian part of the named entity "
    "recognition dataset WikiANN",
    huggingface_id="EuroEval/wikiann-lv-mini",
    task=NER,
    languages=[LV],
)
