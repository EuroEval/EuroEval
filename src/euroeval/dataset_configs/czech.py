"""All Czech dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import CS
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

CSFD_SENTIMENT_CONFIG = DatasetConfig(
    name="csfd-sentiment",
    pretty_name="the truncated version of the Czech sentiment classification dataset "
    "CSFD Sentiment",
    huggingface_id="EuroEval/csfd-sentiment-mini",
    task=SENT,
    languages=[CS],
)

CS_GEC_CONFIG = DatasetConfig(
    name="cs-gec",
    pretty_name="the truncated version of the Czech linguistic acceptability dataset "
    "CS-GEC",
    huggingface_id="EuroEval/cs-gec-mini",
    task=LA,
    languages=[CS],
)

PONER_CONFIG = DatasetConfig(
    name="poner",
    pretty_name="the truncated version of the Czech named entity recognition dataset "
    "PONER",
    huggingface_id="EuroEval/poner-mini",
    task=NER,
    languages=[CS],
)

SQAD_CONFIG = DatasetConfig(
    name="sqad",
    pretty_name="the truncated version of the Czech reading comprehension dataset SQAD",
    huggingface_id="EuroEval/sqad-mini",
    task=RC,
    languages=[CS],
)

CZECH_NEWS_CONFIG = DatasetConfig(
    name="czech-news",
    pretty_name="the truncated version of the Czech summarisation dataset",
    huggingface_id="EuroEval/czech-news-mini",
    task=SUMM,
    languages=[CS],
)

UMIMETO_QA_CONFIG = DatasetConfig(
    name="umimeto-qa",
    pretty_name="the Czech knowledge dataset UmimetoQA",
    huggingface_id="EuroEval/umimeto-qa",
    task=KNOW,
    languages=[CS],
)

HELLASWAG_CS_CONFIG = DatasetConfig(
    name="hellaswag-cs",
    pretty_name="the truncated version of the Czech common-sense reasoning dataset "
    "HellaSwag-cs, translated from the English HellaSwag dataset",
    huggingface_id="EuroEval/hellaswag-cs-mini",
    task=COMMON_SENSE,
    languages=[CS],
)


### Unofficial datasets ###

SCALA_CS_CONFIG = DatasetConfig(
    name="scala-cs",
    pretty_name="the Czech part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-cs",
    task=LA,
    languages=[CS],
    unofficial=True,
)
