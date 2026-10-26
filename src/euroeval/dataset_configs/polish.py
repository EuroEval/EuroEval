"""All Polish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, POLISH
from ..tasks import (
    COMMON_SENSE,
    EUROPEAN_VALUES,
    HALLU,
    INSTRUCTION_FOLLOWING,
    KNOW,
    LA,
    NER,
    RC,
    SENT,
    SUMM,
    TOOL_CALLING,
    TRANSLATION,
)

# Official datasets ###

WMT24PP_EN_PL_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-pl",
    pretty_name="WMT24++-en-pl",
    source="EuroEval/wmt24pp-en-pl",
    task=TRANSLATION,
    languages=[POLISH],
    source_language=ENGLISH,
    target_language=POLISH,
    revision="4474016823d9bf3fe67323b4737713566e403a13",
)

FLORES_EN_PL_CONFIG = TranslationDatasetConfig(
    name="flores-en-pl",
    pretty_name="FLORES-en-pl",
    source="EuroEval/flores-en-pl",
    task=TRANSLATION,
    languages=[POLISH],
    source_language=ENGLISH,
    target_language=POLISH,
    revision="4393b0420b058b8bf389d6a7c9216b8606d1d832",
)

POLEMO2_CONFIG = DatasetConfig(
    name="polemo2",
    pretty_name="Polemo2",
    source="EuroEval/polemo2-mini",
    task=SENT,
    languages=[POLISH],
    revision="b565f5cc441387c6ab62e4c105a221e5c63ad899",
)

SCALA_PL_CONFIG = DatasetConfig(
    name="scala-pl",
    pretty_name="ScaLA-pl",
    source="EuroEval/scala-pl",
    task=LA,
    languages=[POLISH],
    revision="af5e474cab598d98fea9df714145d9e2300e138f",
)

KPWR_NER_CONFIG = DatasetConfig(
    name="kpwr-ner",
    pretty_name="KPWr-NER",
    source="EuroEval/kpwr-ner",
    task=NER,
    languages=[POLISH],
    revision="ca4e108e54c157c56f441453a0a1260c9d4102bf",
)

POQUAD_CONFIG = DatasetConfig(
    name="poquad",
    pretty_name="PoQuAD",
    source="EuroEval/poquad-mini",
    task=RC,
    languages=[POLISH],
    revision="98b97a28700e833197b307f19c1609633626cf1c",
)

PSC_CONFIG = DatasetConfig(
    name="psc",
    pretty_name="PSC",
    source="EuroEval/psc-mini",
    task=SUMM,
    languages=[POLISH],
    revision="e9894e4eaa1a204f1c437659f0885aed47c6ae57",
)

LLMZSZL_CONFIG = DatasetConfig(
    name="llmzszl",
    pretty_name="LLMzSzŁ",
    source="EuroEval/llmzszl-mini",
    task=KNOW,
    languages=[POLISH],
    revision="31ee3d9cc8dff5bbd9449f99c3abc26625f38ffe",
)

WINOGRANDE_PL_CONFIG = DatasetConfig(
    name="winogrande-pl",
    pretty_name="Winogrande-pl",
    source="EuroEval/winogrande-pl",
    task=COMMON_SENSE,
    languages=[POLISH],
    labels=["a", "b"],
    revision="552963d840649478e5024619d8772e4a21e8095e",
)

VALEU_PL_CONFIG = DatasetConfig(
    name="valeu-pl",
    pretty_name="VaLEU-pl",
    source="EuroEval/european-values-pl",
    task=EUROPEAN_VALUES,
    languages=[POLISH],
    train_split=None,
    val_split=None,
    bootstrap_samples=False,
    instruction_prompt="{text}",
    revision="68a243f8cd235f8019a7545fe949afd35b8eaad4",
)

MULTI_IFEVAL_PL_CONFIG = DatasetConfig(
    name="multi-ifeval-pl",
    pretty_name="MultiIFEval-pl",
    source="EuroEval/multi-ifeval-pl",
    task=INSTRUCTION_FOLLOWING,
    languages=[POLISH],
    train_split=None,
    val_split=None,
    revision="c88e11208b5d7f81edf43a04b5fdeaa5f3fbf0b1",
)

RAGTRUTH_PL_CONFIG = DatasetConfig(
    name="ragtruth-pl",
    pretty_name="RAGTruth-pl",
    source="EuroEval/ragtruth-translated-hallucinations-pl-mini",
    task=HALLU,
    languages=[POLISH],
    train_split=None,
    revision="91549683f32a943d2854d9f3ea2526229513dfc6",
)


# Unofficial datasets ###

MULTI_BFCL_PL_CONFIG = DatasetConfig(
    name="multi-bfcl-pl",
    pretty_name="MultiBFCL-pl",
    source="EuroEval/multi-bfcl-pl-mini",
    task=TOOL_CALLING,
    languages=[POLISH],
    train_split=None,
    revision="d1de637188b935753934aea1e3bb8d83a6607db0",
)


WMT24PP_PL_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-pl-en",
    pretty_name="WMT24++-pl-en",
    source="EuroEval/wmt24pp-pl-en",
    task=TRANSLATION,
    languages=[POLISH],
    source_language=POLISH,
    target_language=ENGLISH,
    unofficial=True,
    revision="76999fe53ade9b3f18116aec6b8197dbb7a1984e",
)

FLORES_PL_EN_CONFIG = TranslationDatasetConfig(
    name="flores-pl-en",
    pretty_name="FLORES-pl-en",
    source="EuroEval/flores-pl-en",
    task=TRANSLATION,
    languages=[POLISH],
    source_language=POLISH,
    target_language=ENGLISH,
    unofficial=True,
    revision="f3f2579673d3dc0bf18812cfe3cd467090f7f5af",
)

MULTI_WIKI_QA_PL_CONFIG = DatasetConfig(
    name="multi-wiki-qa-pl",
    pretty_name="MultiWikiQA-pl",
    source="EuroEval/multi-wiki-qa-pl-mini",
    task=RC,
    languages=[POLISH],
    unofficial=True,
    revision="583d3c50cd07ded53a7aaa28144f9ba620c96052",
)

GOLDENSWAG_PL_CONFIG = DatasetConfig(
    name="goldenswag-pl",
    pretty_name="GoldenSwag-pl",
    source="EuroEval/goldenswag-pl-mini",
    task=COMMON_SENSE,
    languages=[POLISH],
    unofficial=True,
    revision="ab78e1080be5f1d6a35d6b122f1e74966cd0d5f5",
)

INCLUDE_PL_CONFIG = DatasetConfig(
    name="include-pl",
    pretty_name="INCLUDE-pl",
    source="EuroEval/include-pl-mini",
    task=KNOW,
    languages=[POLISH],
    unofficial=True,
    revision="dd80a4118bde526a5f76444d0bed9c0fbbcebb98",
)

EU_MMLU_PL_CONFIG = DatasetConfig(
    name="eu-mmlu-pl",
    pretty_name="EU-MMLU-pl",
    source="EuroEval/eu-mmlu-pl",
    task=KNOW,
    languages=[POLISH],
    unofficial=True,
    revision="a261341ebe89bd80336afbfab335693a450e0358",
)
