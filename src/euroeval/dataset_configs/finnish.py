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
    TOOL_CALLING,
    TRANSLATION,
)

# Official datasets ###

WMT24PP_EN_FI_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-fi",
    pretty_name="WMT24++-en-fi",
    source="EuroEval/wmt24pp-en-fi",
    task=TRANSLATION,
    languages=[FINNISH],
    source_language=ENGLISH,
    target_language=FINNISH,
    revision="cbe98cd1bb39941b773080e2b20d025a078345c2",
)

FLORES_EN_FI_CONFIG = TranslationDatasetConfig(
    name="flores-en-fi",
    pretty_name="FLORES-en-fi",
    source="EuroEval/flores-en-fi",
    task=TRANSLATION,
    languages=[FINNISH],
    source_language=ENGLISH,
    target_language=FINNISH,
    revision="868d824477005aac8d90c96c10476d7076d39814",
)

SCANDISENT_FI_CONFIG = DatasetConfig(
    name="scandisent-fi",
    pretty_name="ScandiSent-fi",
    source="EuroEval/scandisent-fi-mini",
    task=SENT,
    languages=[FINNISH],
    labels=["negative", "positive"],
    revision="2ce56803e10ea4d2bdde3c7b6f5b83e8ab73ddd0",
)

TURKU_NER_FI_CONFIG = DatasetConfig(
    name="turku-ner-fi",
    pretty_name="Turku NER-fi",
    source="EuroEval/turku-ner-fi-mini",
    task=NER,
    languages=[FINNISH],
    revision="389a85c64672016202943e5d31f9554ec3753332",
)

TYDIQA_FI_CONFIG = DatasetConfig(
    name="tydiqa-fi",
    pretty_name="TyDiQA-fi",
    source="EuroEval/tydiqa-fi-mini",
    task=RC,
    languages=[FINNISH],
    revision="9585aa484da9c97a5c90186ded7dab2bd0503127",
)

XLSUM_FI_CONFIG = DatasetConfig(
    name="xlsum-fi",
    pretty_name="XLSum-fi",
    source="EuroEval/xlsum-fi-mini",
    task=SUMM,
    languages=[FINNISH],
    revision="b8a4cbccf752130246259ef42e63c99a74c1c90a",
)

SCALA_FI_CONFIG = DatasetConfig(
    name="scala-fi",
    pretty_name="ScaLA-fi",
    source="EuroEval/scala-fi",
    task=LA,
    languages=[FINNISH],
    revision="c687c0f504967f387587c41eee4d0d23dd487d1a",
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
    revision="9bdef9bc2c6c52fd83222a354ba271ca8aab4145",
)

MULTI_IFEVAL_FI_CONFIG = DatasetConfig(
    name="multi-ifeval-fi",
    pretty_name="MultiIFEval-fi",
    source="EuroEval/multi-ifeval-fi",
    task=INSTRUCTION_FOLLOWING,
    languages=[FINNISH],
    train_split=None,
    val_split=None,
    revision="3943df87928d423219a167b00048f641a8372c82",
)

RAGTRUTH_FI_CONFIG = DatasetConfig(
    name="ragtruth-fi",
    pretty_name="RAGTruth-fi",
    source="EuroEval/ragtruth-translated-hallucinations-fi-mini",
    task=HALLU,
    languages=[FINNISH],
    train_split=None,
    revision="259b829a78b401315919e5949011aa1f27bfd008",
)


WINOGRANDE_FI_CONFIG = DatasetConfig(
    name="winogrande-fi",
    pretty_name="Winogrande-fi",
    source="EuroEval/winogrande-fi",
    task=COMMON_SENSE,
    languages=[FINNISH],
    labels=["a", "b"],
    revision="b0b97590fe6e0129162d38fa16e33394ad1b7159",
)

INCLUDE_FI_CONFIG = DatasetConfig(
    name="include-fi",
    pretty_name="INCLUDE-fi",
    source="EuroEval/include-fi-mini",
    task=KNOW,
    languages=[FINNISH],
    revision="356b7324c3873a1dc300459f5373d0b0296092fe",
)


# Unofficial datasets ###

MULTI_BFCL_FI_CONFIG = DatasetConfig(
    name="multi-bfcl-fi",
    pretty_name="MultiBFCL-fi",
    source="EuroEval/multi-bfcl-fi-mini",
    task=TOOL_CALLING,
    languages=[FINNISH],
    train_split=None,
    revision="373ad1f3b1fbfccea47a24a7aab3e45015b7c1ff",
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
    revision="8bced16f47107562489ec7a6f3a42071f1c05f84",
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
    revision="01fc1d9e8b0dd797d2519bfd95293f736da96426",
)

HELLASWAG_FI_CONFIG = DatasetConfig(
    name="hellaswag-fi",
    pretty_name="HellaSwag-fi",
    source="EuroEval/hellaswag-fi-mini",
    task=COMMON_SENSE,
    languages=[FINNISH],
    unofficial=True,
    revision="001337d173a53784f8cf8bf4be7380445520ce71",
)

BELEBELE_FI_CONFIG = DatasetConfig(
    name="belebele-fi",
    pretty_name="Belebele-fi",
    source="EuroEval/belebele-fi-mini",
    task=MCRC,
    languages=[FINNISH],
    unofficial=True,
    revision="8cb340d276f34e92779cc754c900e67336bc8210",
)

MULTI_WIKI_QA_FI_CONFIG = DatasetConfig(
    name="multi-wiki-qa-fi",
    pretty_name="MultiWikiQA-fi",
    source="EuroEval/multi-wiki-qa-fi-mini",
    task=RC,
    languages=[FINNISH],
    unofficial=True,
    revision="3e09dff02cec87a85ac7868726e5226b26866026",
)

GOLDENSWAG_FI_CONFIG = DatasetConfig(
    name="goldenswag-fi",
    pretty_name="GoldenSwag-fi",
    source="EuroEval/goldenswag-fi-mini",
    task=COMMON_SENSE,
    languages=[FINNISH],
    unofficial=True,
    revision="c179384295537cd4e6776f8582934ac167da79ab",
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
    revision="2c5040aeb9854be607be86cfa90c01a51fb74b16",
)
