"""All Romanian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import ROMANIAN
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

ROSENT_CONFIG = DatasetConfig(
    name="ro-sent",
    pretty_name="RoSent",
    source="EuroEval/ro-sent-mini",
    task=SENT,
    languages=[ROMANIAN],
    _labels=["positive", "negative"],
)

SCALA_RO_CONFIG = DatasetConfig(
    name="scala-ro",
    pretty_name="ScaLA-ro",
    source="EuroEval/scala-ro",
    task=LA,
    languages=[ROMANIAN],
)

RONEC_CONFIG = DatasetConfig(
    name="ronec",
    pretty_name="RoNEC",
    source="EuroEval/ronec-mini",
    task=NER,
    languages=[ROMANIAN],
)

MULTI_WIKI_QA_RO_CONFIG = DatasetConfig(
    name="multi-wiki-qa-ro",
    pretty_name="MultiWikiQA-ro",
    source="EuroEval/multi-wiki-qa-ro-mini",
    task=RC,
    languages=[ROMANIAN],
)

SUMO_RO_CONFIG = DatasetConfig(
    name="sumo-ro",
    pretty_name="SumO-Ro",
    source="EuroEval/sumo-ro-mini",
    task=SUMM,
    languages=[ROMANIAN],
)

GLOBAL_MMLU_RO_CONFIG = DatasetConfig(
    name="global-mmlu-ro",
    pretty_name="GlobalMMLU-ro",
    source="EuroEval/global-mmlu-ro-mini",
    task=KNOW,
    languages=[ROMANIAN],
)

WINOGRANDE_RO_CONFIG = DatasetConfig(
    name="winogrande-ro",
    pretty_name="Winogrande-ro",
    source="EuroEval/winogrande-ro",
    task=COMMON_SENSE,
    languages=[ROMANIAN],
    _labels=["a", "b"],
)
