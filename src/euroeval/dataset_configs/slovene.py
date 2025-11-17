"""All Slovene dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import SLOVENE
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT

### Official datasets ###

SENTINEWS_CONFIG = DatasetConfig(
    name="sentinews",
    pretty_name="Sentinews-sl",
    source="EuroEval/sentinews-mini",
    task=SENT,
    languages=[SLOVENE],
)

SCALA_SL_CONFIG = DatasetConfig(
    name="scala-sl",
    pretty_name="ScaLA-sl",
    source="EuroEval/scala-sl",
    task=LA,
    languages=[SLOVENE],
)

SSJ500K_NER_CONFIG = DatasetConfig(
    name="ssj500k-ner",
    pretty_name="ssj500k-NER",
    source="EuroEval/ssj500k-ner-mini",
    task=NER,
    languages=[SLOVENE],
)

MULTI_WIKI_QA_SL_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sl",
    pretty_name="MultiWikiQA-sl",
    source="EuroEval/multi-wiki-qa-sl-mini",
    task=RC,
    languages=[SLOVENE],
)

MMLU_SL_CONFIG = DatasetConfig(
    name="mmlu-sl",
    pretty_name="MMLU-sl",
    source="EuroEval/mmlu-sl-mini",
    task=KNOW,
    languages=[SLOVENE],
)

WINOGRANDE_SL_CONFIG = DatasetConfig(
    name="winogrande-sl",
    pretty_name="Winogrande-sl",
    source="EuroEval/winogrande-sl",
    task=COMMON_SENSE,
    languages=[SLOVENE],
    _labels=["a", "b"],
)
