"""All Finnish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, FINNISH
from ..tasks import (
    COMMON_SENSE,
    EUROPEAN_VALUES,
    HALLU,
    INSTRUCTION_FOLLOWING,
    KNOW,
    LA,
    MCRC,
    NER,
    RC,
    SENT,
    SUMM,
    TRANSLATION,
)

# Official datasets ###

SCANDISENT_FI_CONFIG = DatasetConfig(
    name="scandisent-fi",
    pretty_name="ScandiSent-fi",
    source="EuroEval/scandisent-fi-mini",
    task=SENT,
    languages=[FINNISH],
    labels=["negative", "positive"],
)

TURKU_NER_FI_CONFIG = DatasetConfig(
    name="turku-ner-fi",
    pretty_name="Turku NER-fi",
    source="EuroEval/turku-ner-fi-mini",
    task=NER,
    languages=[FINNISH],
)

TYDIQA_FI_CONFIG = DatasetConfig(
    name="tydiqa-fi",
    pretty_name="TyDiQA-fi",
    source="EuroEval/tydiqa-fi-mini",
    task=RC,
    languages=[FINNISH],
)

XLSUM_FI_CONFIG = DatasetConfig(
    name="xlsum-fi",
    pretty_name="XLSum-fi",
    source="EuroEval/xlsum-fi-mini",
    task=SUMM,
    languages=[FINNISH],
)

SCALA_FI_CONFIG = DatasetConfig(
    name="scala-fi",
    pretty_name="ScaLA-fi",
    source="EuroEval/scala-fi",
    task=LA,
    languages=[FINNISH],
)

VALEU_FI_CONFIG = DatasetConfig(
    name="valeu-fi",
    pretty_name="VaLEU-fi",
    source="EuroEval/european-values-fi",
    task=EUROPEAN_VALUES,
    languages=[FINNISH],
    train_split=None,
    val_split=None,
    bootstrap_samples=False,
    instruction_prompt="{text}",
)

MULTI_IFEVAL_FI_CONFIG = DatasetConfig(
    name="multi-ifeval-fi",
    pretty_name="MultiIFEval-fi",
    source="EuroEval/multi-ifeval-fi",
    task=INSTRUCTION_FOLLOWING,
    languages=[FINNISH],
    train_split=None,
    val_split=None,
)

RAGTRUTH_FI_CONFIG = DatasetConfig(
    name="ragtruth-fi",
    pretty_name="RAGTruth-fi",
    source="EuroEval/ragtruth-translated-hallucinations-fi-mini",
    task=HALLU,
    languages=[FINNISH],
    train_split=None,
)


WINOGRANDE_FI_CONFIG = DatasetConfig(
    name="winogrande-fi",
    pretty_name="Winogrande-fi",
    source="EuroEval/winogrande-fi",
    task=COMMON_SENSE,
    languages=[FINNISH],
    labels=["a", "b"],
)

INCLUDE_FI_CONFIG = DatasetConfig(
    name="include-fi",
    pretty_name="INCLUDE-fi",
    source="EuroEval/include-fi-mini",
    task=KNOW,
    languages=[FINNISH],
)


WMT24PP_EN_FI_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-fi",
    pretty_name="WMT24++-en-fi",
    source="EuroEval/wmt24pp-en-fi",
    task=TRANSLATION,
    languages=[FINNISH],
    source_language=ENGLISH,
    target_language=FINNISH,
)

WMT24PP_FI_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-fi-en",
    pretty_name="WMT24++-fi-en",
    source="EuroEval/wmt24pp-fi-en",
    task=TRANSLATION,
    languages=[FINNISH],
    source_language=FINNISH,
    target_language=ENGLISH,
    unofficial=True,
)


# Unofficial datasets ###

HELLASWAG_FI_CONFIG = DatasetConfig(
    name="hellaswag-fi",
    pretty_name="HellaSwag-fi",
    source="EuroEval/hellaswag-fi-mini",
    task=COMMON_SENSE,
    languages=[FINNISH],
    unofficial=True,
)

BELEBELE_FI_CONFIG = DatasetConfig(
    name="belebele-fi",
    pretty_name="Belebele-fi",
    source="EuroEval/belebele-fi-mini",
    task=MCRC,
    languages=[FINNISH],
    unofficial=True,
)

MULTI_WIKI_QA_FI_CONFIG = DatasetConfig(
    name="multi-wiki-qa-fi",
    pretty_name="MultiWikiQA-fi",
    source="EuroEval/multi-wiki-qa-fi-mini",
    task=RC,
    languages=[FINNISH],
    unofficial=True,
)

GOLDENSWAG_FI_CONFIG = DatasetConfig(
    name="goldenswag-fi",
    pretty_name="GoldenSwag-fi",
    source="EuroEval/goldenswag-fi-mini",
    task=COMMON_SENSE,
    languages=[FINNISH],
    unofficial=True,
)

IFEVAL_FI_CONFIG = DatasetConfig(
    name="ifeval-fi",
    pretty_name="IFEval-fi",
    source="EuroEval/ifeval-fi",
    task=INSTRUCTION_FOLLOWING,
    languages=[FINNISH],
    train_split=None,
    val_split=None,
    unofficial=True,
)


FLORES_EN_FI_CONFIG = TranslationDatasetConfig(
    name="flores-en-fi",
    pretty_name="FLORES-en-fi",
    source="EuroEval/flores-en-fi",
    task=TRANSLATION,
    languages=[FINNISH],
    source_language=ENGLISH,
    target_language=FINNISH,
    unofficial=True,
)

FLORES_FI_EN_CONFIG = TranslationDatasetConfig(
    name="flores-fi-en",
    pretty_name="FLORES-fi-en",
    source="EuroEval/flores-fi-en",
    task=TRANSLATION,
    languages=[FINNISH],
    source_language=FINNISH,
    target_language=ENGLISH,
    unofficial=True,
)
