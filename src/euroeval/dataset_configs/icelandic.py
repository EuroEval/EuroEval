"""All Icelandic dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import ICELANDIC
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

HOTTER_AND_COLDER_SENTIMENT_CONFIG = DatasetConfig(
    name="hotter-and-colder-sentiment",
    pretty_name="Hotter and Colder Sentiment",
    source="EuroEval/hotter-and-colder-sentiment",
    task=SENT,
    languages=[ICELANDIC],
)

SCALA_IS_CONFIG = DatasetConfig(
    name="scala-is",
    pretty_name="ScaLA-is",
    source="EuroEval/scala-is",
    task=LA,
    languages=[ICELANDIC],
)

MIM_GOLD_NER_CONFIG = DatasetConfig(
    name="mim-gold-ner",
    pretty_name="MIM-GOLD-NER",
    source="EuroEval/mim-gold-ner-mini",
    task=NER,
    languages=[ICELANDIC],
)

NQII_CONFIG = DatasetConfig(
    name="nqii",
    pretty_name="NQiI",
    source="EuroEval/nqii-mini",
    task=RC,
    languages=[ICELANDIC],
)

RRN_CONFIG = DatasetConfig(
    name="rrn",
    pretty_name="RRN",
    source="EuroEval/rrn-mini",
    task=SUMM,
    languages=[ICELANDIC],
)

ICELANDIC_KNOWLEDGE_CONFIG = DatasetConfig(
    name="icelandic-knowledge",
    pretty_name="Icelandic Knowledge",
    source="EuroEval/icelandic-knowledge",
    task=KNOW,
    languages=[ICELANDIC],
)

WINOGRANDE_IS_CONFIG = DatasetConfig(
    name="winogrande-is",
    pretty_name="Winogrande-is",
    source="EuroEval/winogrande-is",
    task=COMMON_SENSE,
    languages=[ICELANDIC],
    _labels=["a", "b"],
)

VALEU_IS_CONFIG = DatasetConfig(
    name="valeu-is",
    pretty_name="VaLEU-is",
    source="EuroEval/european-values-is",
    task=EUROPEAN_VALUES,
    languages=[ICELANDIC],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
)


### Unofficial datasets ###

ICE_EC_CONFIG = DatasetConfig(
    name="ice-ec",
    pretty_name="ICE-EC",
    source="EuroEval/ice-ec",
    task=LA,
    languages=[ICELANDIC],
    unofficial=True,
)

ICE_EC_FULL_CONFIG = DatasetConfig(
    name="ice-ec-full",
    pretty_name="ICE-EC Full",
    source="EuroEval/ice-ec-full",
    task=LA,
    languages=[ICELANDIC],
    unofficial=True,
)

ICE_LINGUISTIC_CONFIG = DatasetConfig(
    name="ice-linguistic",
    pretty_name="IceLinguistic",
    source="EuroEval/ice-linguistic",
    task=LA,
    languages=[ICELANDIC],
    unofficial=True,
)

ICELANDIC_QA_CONFIG = DatasetConfig(
    name="icelandic-qa",
    pretty_name="Icelandic QA",
    source="EuroEval/icelandic-qa",
    task=RC,
    languages=[ICELANDIC],
    unofficial=True,
)

MMLU_IS_CONFIG = DatasetConfig(
    name="mmlu-is",
    pretty_name="MMLU-is",
    source="EuroEval/mmlu-is-mini",
    task=KNOW,
    languages=[ICELANDIC],
    unofficial=True,
)

ARC_IS_CONFIG = DatasetConfig(
    name="arc-is",
    pretty_name="ARC-is",
    source="EuroEval/arc-is-mini",
    task=KNOW,
    languages=[ICELANDIC],
    unofficial=True,
)

HELLASWAG_IS_CONFIG = DatasetConfig(
    name="hellaswag-is",
    pretty_name="HellaSwag-is",
    source="EuroEval/hellaswag-is-mini",
    task=COMMON_SENSE,
    languages=[ICELANDIC],
    unofficial=True,
)

BELEBELE_IS_CONFIG = DatasetConfig(
    name="belebele-is",
    pretty_name="Belebele-is",
    source="EuroEval/belebele-is-mini",
    task=MCRC,
    languages=[ICELANDIC],
    unofficial=True,
)

MULTI_WIKI_QA_IS_CONFIG = DatasetConfig(
    name="multi-wiki-qa-is",
    pretty_name="MultiWikiQA-is",
    source="EuroEval/multi-wiki-qa-is-mini",
    task=RC,
    languages=[ICELANDIC],
    unofficial=True,
)
