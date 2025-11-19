"""All Hungarian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import HUNGARIAN
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

HUSST_CONFIG = DatasetConfig(
    name="husst",
    pretty_name="HuSST",
    source="EuroEval/husst-mini",
    task=SENT,
    languages=[HUNGARIAN],
)

SCALA_HU_CONFIG = DatasetConfig(
    name="scala-hu",
    pretty_name="ScaLA-hu",
    source="EuroEval/scala-hu",
    task=LA,
    languages=[HUNGARIAN],
)

SZEGED_NER_CONFIG = DatasetConfig(
    name="szeged-ner",
    pretty_name="SzegedNER",
    source="EuroEval/szeged-ner",
    task=NER,
    languages=[HUNGARIAN],
)

MULTI_WIKI_QA_HU_CONFIG = DatasetConfig(
    name="multi-wiki-qa-hu",
    pretty_name="MultiWikiQA-hu",
    source="EuroEval/multi-wiki-qa-hu-mini",
    task=RC,
    languages=[HUNGARIAN],
)

HUNSUM_CONFIG = DatasetConfig(
    name="hunsum",
    pretty_name="HunSum",
    source="EuroEval/hun-sum-mini",
    task=SUMM,
    languages=[HUNGARIAN],
)

MMLU_HU_CONFIG = DatasetConfig(
    name="mmlu-hu",
    pretty_name="MMLU-hu",
    source="EuroEval/mmlu-hu-mini",
    task=KNOW,
    languages=[HUNGARIAN],
)

WINOGRANDE_HU_CONFIG = DatasetConfig(
    name="winogrande-hu",
    pretty_name="Winogrande-hu",
    source="EuroEval/winogrande-hu",
    task=COMMON_SENSE,
    languages=[HUNGARIAN],
    _labels=["a", "b"],
)
