"""All Slovak dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import SK
from ..tasks import LA, NER, RC, SENT

### Official datasets ###

CSFD_SENTIMENT_CONFIG = DatasetConfig(
    name="csfd-sentiment-sk",
    pretty_name="the Slovak Movie Database sentiment dataset",
    huggingface_id="EuroEval/csfd-sentiment-sk-mini",
    task=SENT,
    languages=[SK],
)

SCALA_SK_CONFIG = DatasetConfig(
    name="scala-sk",
    pretty_name="the Slovak part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-cs",
    task=LA,
    languages=[SK],
)

UNER_SK_CONFIG = DatasetConfig(
    name="uner-sk",
    pretty_name="the Slovak named entity recognition dataset UNER-sk",
    huggingface_id="EuroEval/uner-sk-mini",
    task=NER,
    languages=[SK],
)

MULTI_WIKI_QA_SK_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sk",
    pretty_name="the Slovak part of the reading comprehension dataset MultiWikiQA",
    huggingface_id="EuroEval/multi-wiki-qa-sk-mini",
    task=RC,
    languages=[SK],
)
