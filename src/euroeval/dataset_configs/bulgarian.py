"""All Bulgarian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import BULGARIAN
from ..tasks import COMMON_SENSE, HALLU, INSTRUCTION_FOLLOWING, KNOW, LA, NER, RC, SENT

# Official datasets ###

CINEXIO_CONFIG = DatasetConfig(
    name="cinexio",
    pretty_name="Cinexio",
    source="EuroEval/cinexio-mini",
    task=SENT,
    languages=[BULGARIAN],
)

SCALA_BG_CONFIG = DatasetConfig(
    name="scala-bg",
    pretty_name="ScaLA-bg",
    source="EuroEval/scala-bg",
    task=LA,
    languages=[BULGARIAN],
)

BG_NER_BSNLP_CONFIG = DatasetConfig(
    name="bg-ner-bsnlp",
    pretty_name="BG-NER-BSNLp",
    source="EuroEval/bg-ner-bsnlp-mini",
    task=NER,
    languages=[BULGARIAN],
)

MULTI_WIKI_QA_BG_CONFIG = DatasetConfig(
    name="multi-wiki-qa-bg",
    pretty_name="MultiWikiQA-bg",
    source="EuroEval/multi-wiki-qa-bg-mini",
    task=RC,
    languages=[BULGARIAN],
)

EXAMS_BG_CONFIG = DatasetConfig(
    name="exams-bg",
    pretty_name="Exams-bg",
    source="EuroEval/exams-bg-mini",
    task=KNOW,
    languages=[BULGARIAN],
)

WINOGRANDE_BG_CONFIG = DatasetConfig(
    name="winogrande-bg",
    pretty_name="Winogrande-bg",
    source="EuroEval/winogrande-bg",
    task=COMMON_SENSE,
    languages=[BULGARIAN],
    labels=["a", "b"],
)

MULTI_IFEVAL_BG_CONFIG = DatasetConfig(
    name="multi-ifeval-bg",
    pretty_name="MultiIFEval-bg",
    source="EuroEval/multi-ifeval-bg",
    task=INSTRUCTION_FOLLOWING,
    languages=[BULGARIAN],
    train_split=None,
    val_split=None,
)

RAGTRUTH_BG_CONFIG = DatasetConfig(
    name="ragtruth-bg",
    pretty_name="RAGTruth-bg",
    source="EuroEval/ragtruth-translated-hallucinations-bg-mini",
    task=HALLU,
    languages=[BULGARIAN],
    train_split=None,
)

# Unofficial datasets ###

INCLUDE_BG_CONFIG = DatasetConfig(
    name="include-bg",
    pretty_name="INCLUDE-bg",
    source="EuroEval/include-bg-mini",
    task=KNOW,
    languages=[BULGARIAN],
    unofficial=True,
)


