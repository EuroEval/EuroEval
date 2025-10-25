"""All Bulgarian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import BULGARIAN
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT

### Official datasets ###

CINEXIO_CONFIG = DatasetConfig(
    name="cinexio", source="EuroEval/cinexio-mini", task=SENT, languages=[BULGARIAN]
)

SCALA_BG_CONFIG = DatasetConfig(
    name="scala-bg", source="EuroEval/scala-bg", task=LA, languages=[BULGARIAN]
)

BG_NER_BSNLP_CONFIG = DatasetConfig(
    name="bg-ner-bsnlp",
    source="EuroEval/bg-ner-bsnlp-mini",
    task=NER,
    languages=[BULGARIAN],
)

MULTI_WIKI_QA_BG_CONFIG = DatasetConfig(
    name="multi-wiki-qa-bg",
    source="EuroEval/multi-wiki-qa-bg-mini",
    task=RC,
    languages=[BULGARIAN],
)

EXAMS_BG_CONFIG = DatasetConfig(
    name="exams-bg", source="EuroEval/exams-bg-mini", task=KNOW, languages=[BULGARIAN]
)

WINOGRANDE_BG_CONFIG = DatasetConfig(
    name="winogrande-bg",
    source="EuroEval/winogrande-bg",
    task=COMMON_SENSE,
    languages=[BULGARIAN],
    _labels=["a", "b"],
)
