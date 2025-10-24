"""All Bulgarian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import BG
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT

### Official datasets ###

CINEXIO_CONFIG = DatasetConfig(
    name="cinexio",
    pretty_name="the truncated version of the Bulgarian sentiment "
    "classification dataset Cinexio",
    huggingface_id="EuroEval/cinexio-mini",
    task=SENT,
    languages=[BG],
)

SCALA_BG_CONFIG = DatasetConfig(
    name="scala-bg",
    pretty_name="the Bulgarian part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-bg",
    task=LA,
    languages=[BG],
)

BG_NER_BSNLP_CONFIG = DatasetConfig(
    name="bg-ner-bsnlp",
    pretty_name="the truncated version of the Bulgarian named entity recognition "
    "dataset BG-NER-BSNLP",
    huggingface_id="EuroEval/bg-ner-bsnlp-mini",
    task=NER,
    languages=[BG],
)

MULTI_WIKI_QA_BG_CONFIG = DatasetConfig(
    name="multi-wiki-qa-bg",
    pretty_name="the truncated version of the Bulgarian part of the reading "
    "comprehension dataset MultiWikiQA",
    huggingface_id="EuroEval/multi-wiki-qa-bg-mini",
    task=RC,
    languages=[BG],
)

EXAMS_BG_CONFIG = DatasetConfig(
    name="exams-bg",
    pretty_name="the truncated version of the Bulgarian knowledge "
    "dataset Exams-bg",
    huggingface_id="EuroEval/exams-bg-mini",
    task=KNOW,
    languages=[BG],
)

WINOGRANDE_BG_CONFIG = DatasetConfig(
    name="winogrande-bg",
    pretty_name="the Bulgarian common-sense reasoning dataset Winogrande-bg, "
    "translated from the English Winogrande dataset",
    huggingface_id="EuroEval/winogrande-bg",
    task=COMMON_SENSE,
    languages=[BG],
    _labels=["a", "b"],
)
