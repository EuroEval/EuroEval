"""All Hungarian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import HUNGARIAN
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT

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

MULTI_WIKI_QA_BG_CONFIG = DatasetConfig(
    name="multi-wiki-qa-bg",
    pretty_name="MultiWikiQA-bg",
    source="EuroEval/multi-wiki-qa-bg-mini",
    task=RC,
    languages=[HUNGARIAN],
)

EXAMS_BG_CONFIG = DatasetConfig(
    name="exams-bg",
    pretty_name="Exams-bg",
    source="EuroEval/exams-bg-mini",
    task=KNOW,
    languages=[HUNGARIAN],
)

WINOGRANDE_BG_CONFIG = DatasetConfig(
    name="winogrande-bg",
    pretty_name="Winogrande-bg",
    source="EuroEval/winogrande-bg",
    task=COMMON_SENSE,
    languages=[HUNGARIAN],
    _labels=["a", "b"],
)
