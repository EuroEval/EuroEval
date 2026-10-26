"""All Icelandic dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, ICELANDIC
from ..tasks import (
    COMMON_SENSE,
    EUROPEAN_VALUES,
    GED,
    HALLU,
    INSTRUCTION_FOLLOWING,
    KNOW,
    LA,
    LOGIC,
    MCRC,
    NER,
    RC,
    SENT,
    SUMM,
    TOOL_CALLING,
    TRANSLATION,
)

# Official datasets ###

WMT24PP_EN_IS_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-is",
    pretty_name="WMT24++-en-is",
    source="EuroEval/wmt24pp-en-is",
    task=TRANSLATION,
    languages=[ICELANDIC],
    source_language=ENGLISH,
    target_language=ICELANDIC,
    revision="c47816d4dac04685ce12705a244f4e6f54c3bb76",
)

FLORES_EN_IS_CONFIG = TranslationDatasetConfig(
    name="flores-en-is",
    pretty_name="FLORES-en-is",
    source="EuroEval/flores-en-is",
    task=TRANSLATION,
    languages=[ICELANDIC],
    source_language=ENGLISH,
    target_language=ICELANDIC,
    revision="446b8572eeda6924f955e44abe3ed5781a27e7d8",
)

HOTTER_AND_COLDER_SENTIMENT_CONFIG = DatasetConfig(
    name="hotter-and-colder-sentiment",
    pretty_name="Hotter and Colder Sentiment",
    source="EuroEval/hotter-and-colder-sentiment",
    task=SENT,
    languages=[ICELANDIC],
    revision="72e98ece38fa091700f799eb867d16645a478dfc",
)

MIM_GOLD_NER_CONFIG = DatasetConfig(
    name="mim-gold-ner",
    pretty_name="MIM-GOLD-NER",
    source="EuroEval/mim-gold-ner-mini",
    task=NER,
    languages=[ICELANDIC],
    revision="10d04e5710d83c882ae62acb2dd928958b49516a",
)

NQII_CONFIG = DatasetConfig(
    name="nqii",
    pretty_name="NQiI",
    source="EuroEval/nqii-mini",
    task=RC,
    languages=[ICELANDIC],
    revision="41f1b050ad604341829cf6a4a88d916b4cf787d0",
)

RRN_CONFIG = DatasetConfig(
    name="rrn",
    pretty_name="RRN",
    source="EuroEval/rrn-mini",
    task=SUMM,
    languages=[ICELANDIC],
    revision="bdc72f65521d348b7307578b8a25ce7d2ecc978c",
)

ICELANDIC_KNOWLEDGE_CONFIG = DatasetConfig(
    name="icelandic-knowledge",
    pretty_name="Icelandic Knowledge",
    source="EuroEval/icelandic-knowledge",
    task=KNOW,
    languages=[ICELANDIC],
    revision="7e98944fa12d87abb04a6f6861ea5d489b3c3417",
)

WINOGRANDE_IS_CONFIG = DatasetConfig(
    name="winogrande-is",
    pretty_name="Winogrande-is",
    source="EuroEval/winogrande-is",
    task=COMMON_SENSE,
    languages=[ICELANDIC],
    labels=["a", "b"],
    revision="1be5e61b196bf1c947e14de060cd339a4c131388",
)

MULTI_IFEVAL_IS_CONFIG = DatasetConfig(
    name="multi-ifeval-is",
    pretty_name="MultiIFEval-is",
    source="EuroEval/multi-ifeval-is",
    task=INSTRUCTION_FOLLOWING,
    languages=[ICELANDIC],
    train_split=None,
    val_split=None,
    revision="cf7c53a3a98ad807cf3ded5d92dc7aa91e8d04a8",
)

VALEU_IS_CONFIG = DatasetConfig(
    name="valeu-is",
    pretty_name="VaLEU-is",
    source="EuroEval/european-values-is",
    task=EUROPEAN_VALUES,
    languages=[ICELANDIC],
    train_split=None,
    val_split=None,
    bootstrap_samples=False,
    instruction_prompt="{text}",
    revision="839fadfcc4c70e33b5a7e8442451132c8a1515e9",
)

ZEBRA_PUZZLE_EASY_IS_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-is",
    pretty_name="ZebraPuzzlesEasy-is",
    source="EuroEval/zebra-puzzles-easy-is",
    task=LOGIC,
    languages=[ICELANDIC],
    revision="350790d8a889530b556ef4309a9461271dc5c130",
)

RAGTRUTH_IS_CONFIG = DatasetConfig(
    name="ragtruth-is",
    pretty_name="RAGTruth-is",
    source="EuroEval/ragtruth-translated-hallucinations-is-mini",
    task=HALLU,
    languages=[ICELANDIC],
    train_split=None,
    revision="d835381572cda072b6741e309d929a9b48fc8463",
)


ICE_EC_CONFIG = DatasetConfig(
    name="ice-ec",
    pretty_name="ICE-EC",
    source="EuroEval/ice-ec",
    task=LA,
    languages=[ICELANDIC],
    revision="310c674eefd27aa7e32cb605157761adb14199a1",
)


# Unofficial datasets ###

MULTI_BFCL_IS_CONFIG = DatasetConfig(
    name="multi-bfcl-is",
    pretty_name="MultiBFCL-is",
    source="EuroEval/multi-bfcl-is-mini",
    task=TOOL_CALLING,
    languages=[ICELANDIC],
    train_split=None,
    revision="497620525bade057f92b759441ba056cdda0dd62",
)


WMT24PP_IS_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-is-en",
    pretty_name="WMT24++-is-en",
    source="EuroEval/wmt24pp-is-en",
    task=TRANSLATION,
    languages=[ICELANDIC],
    source_language=ICELANDIC,
    target_language=ENGLISH,
    unofficial=True,
    revision="3a44b3c67ace7adcceff26d0c59c0abab09c1fc1",
)

FLORES_IS_EN_CONFIG = TranslationDatasetConfig(
    name="flores-is-en",
    pretty_name="FLORES-is-en",
    source="EuroEval/flores-is-en",
    task=TRANSLATION,
    languages=[ICELANDIC],
    source_language=ICELANDIC,
    target_language=ENGLISH,
    unofficial=True,
    revision="9b62d7f97089bd0b8b04881fc2614327e2496e0f",
)

SCALA_IS_CONFIG = DatasetConfig(
    name="scala-is",
    pretty_name="ScaLA-is",
    source="EuroEval/scala-is",
    task=LA,
    languages=[ICELANDIC],
    unofficial=True,
    revision="42890ebeb4d1da2bd9440ee12c99c4807adc7656",
)

ICE_EC_FULL_CONFIG = DatasetConfig(
    name="ice-ec-full",
    pretty_name="ICE-EC Full",
    source="EuroEval/ice-ec-full",
    task=LA,
    languages=[ICELANDIC],
    unofficial=True,
    revision="ac7a22b61230327ac334c54c434e4394b76fed88",
)

ICE_LINGUISTIC_CONFIG = DatasetConfig(
    name="ice-linguistic",
    pretty_name="IceLinguistic",
    source="EuroEval/ice-linguistic",
    task=LA,
    languages=[ICELANDIC],
    unofficial=True,
    revision="db489547133757429e99d64577163be59dd5a44c",
)

ICELANDIC_QA_CONFIG = DatasetConfig(
    name="icelandic-qa",
    pretty_name="Icelandic QA",
    source="EuroEval/icelandic-qa",
    task=RC,
    languages=[ICELANDIC],
    unofficial=True,
    revision="56c669c1357d76d13beae8bcc55adc74a603c4d6",
)

MMLU_IS_CONFIG = DatasetConfig(
    name="mmlu-is",
    pretty_name="MMLU-is",
    source="EuroEval/mmlu-is-mini",
    task=KNOW,
    languages=[ICELANDIC],
    unofficial=True,
    revision="bbe01ffafb2a0e4b993e53fe0e2c8f39713dff97",
)

ARC_IS_CONFIG = DatasetConfig(
    name="arc-is",
    pretty_name="ARC-is",
    source="EuroEval/arc-is-mini",
    task=KNOW,
    languages=[ICELANDIC],
    unofficial=True,
    revision="66448b37f108d9da210820b1a3f5935189c1513e",
)

HELLASWAG_IS_CONFIG = DatasetConfig(
    name="hellaswag-is",
    pretty_name="HellaSwag-is",
    source="EuroEval/hellaswag-is-mini",
    task=COMMON_SENSE,
    languages=[ICELANDIC],
    unofficial=True,
    revision="ff047f8cb8675e2d370d917d327206bbcf581c97",
)

BELEBELE_IS_CONFIG = DatasetConfig(
    name="belebele-is",
    pretty_name="Belebele-is",
    source="EuroEval/belebele-is-mini",
    task=MCRC,
    languages=[ICELANDIC],
    unofficial=True,
    revision="c450f3b9ae837be4775ff5bc60ceec881f1990d9",
)

MULTI_WIKI_QA_IS_CONFIG = DatasetConfig(
    name="multi-wiki-qa-is",
    pretty_name="MultiWikiQA-is",
    source="EuroEval/multi-wiki-qa-is-mini",
    task=RC,
    languages=[ICELANDIC],
    unofficial=True,
    revision="8e01f2ef879c1d71f28ae1301fc189cedf3b6af0",
)

ICELANDIC_LANG_TESTS_CONFIG = DatasetConfig(
    name="icelandic-lang-tests",
    pretty_name="Icelandic Language Tests",
    source="EuroEval/icelandic-lang-tests",
    task=KNOW,
    languages=[ICELANDIC],
    unofficial=True,
    val_split=None,
    revision="58ae4d6b30ca0fa2d7d250138735411bc76fda2d",
)

ICELANDIC_MATH_TESTS_CONFIG = DatasetConfig(
    name="icelandic-math-tests",
    pretty_name="Icelandic Mathematics Tests",
    source="EuroEval/icelandic-math-tests",
    task=KNOW,
    languages=[ICELANDIC],
    unofficial=True,
    val_split=None,
    revision="d467f99896936da9e63af4a89fc5fd2eb892bd7a",
)

GERLANGMOD_IS_CONFIG = DatasetConfig(
    name="gerlangmod-is",
    pretty_name="GerLangMod-is",
    source="EuroEval/gerlangmod-is",
    task=GED,
    languages=[ICELANDIC],
    unofficial=True,
    revision="37cbcfc471db7070646a0a6cae75d89f7f023a44",
)

ZEBRA_PUZZLE_HARD_IS_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-is",
    pretty_name="ZebraPuzzlesHard-is",
    source="EuroEval/zebra-puzzles-hard-is",
    task=LOGIC,
    languages=[ICELANDIC],
    unofficial=True,
    revision="c98c3c3607438fb347a7c278d1eea65f7e70f8fc",
)
