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

MULTI_WIKI_QA_HU_CONFIG = DatasetConfig(
    name="multi-wiki-qa-hu",
    pretty_name="MultiWikiQA-hu",
    source="EuroEval/multi-wiki-qa-hu-mini",
    task=RC,
    languages=[ROMANIAN],
)

HUNSUM_CONFIG = DatasetConfig(
    name="hunsum",
    pretty_name="HunSum",
    source="EuroEval/hun-sum-mini",
    task=SUMM,
    languages=[ROMANIAN],
)

MMLU_HU_CONFIG = DatasetConfig(
    name="mmlu-hu",
    pretty_name="MMLU-hu",
    source="EuroEval/mmlu-hu-mini",
    task=KNOW,
    languages=[ROMANIAN],
)

WINOGRANDE_HU_CONFIG = DatasetConfig(
    name="winogrande-hu",
    pretty_name="Winogrande-hu",
    source="EuroEval/winogrande-hu",
    task=COMMON_SENSE,
    languages=[ROMANIAN],
    _labels=["a", "b"],
)
