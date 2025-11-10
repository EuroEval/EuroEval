"""All Czech dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import CZECH
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

CSFD_SENTIMENT_CONFIG = DatasetConfig(
    name="csfd-sentiment",
    pretty_name="CSFD Sentiment",
    source="EuroEval/csfd-sentiment-mini",
    task=SENT,
    languages=[CZECH],
)

CS_GEC_CONFIG = DatasetConfig(
    name="cs-gec",
    pretty_name="CS-GEC",
    source="EuroEval/cs-gec-mini",
    task=LA,
    languages=[CZECH],
)

PONER_CONFIG = DatasetConfig(
    name="poner",
    pretty_name="PoNER",
    source="EuroEval/poner-mini",
    task=NER,
    languages=[CZECH],
)

SQAD_CONFIG = DatasetConfig(
    name="sqad",
    pretty_name="SQAD",
    source="EuroEval/sqad-mini",
    task=RC,
    languages=[CZECH],
)

CZECH_NEWS_CONFIG = DatasetConfig(
    name="czech-news",
    pretty_name="Czech News",
    source="EuroEval/czech-news-mini",
    task=SUMM,
    languages=[CZECH],
)

UMIMETO_QA_CONFIG = DatasetConfig(
    name="umimeto-qa",
    pretty_name="Umimeto QA",
    source="EuroEval/umimeto-qa",
    task=KNOW,
    languages=[CZECH],
)

HELLASWAG_CS_CONFIG = DatasetConfig(
    name="hellaswag-cs",
    pretty_name="HellaSwag-cs",
    source="EuroEval/hellaswag-cs-mini",
    task=COMMON_SENSE,
    languages=[CZECH],
)


### Unofficial datasets ###

SCALA_CS_CONFIG = DatasetConfig(
    name="scala-cs",
    pretty_name="ScaLA-cs",
    source="EuroEval/scala-cs",
    task=LA,
    languages=[CZECH],
    unofficial=True,
)
