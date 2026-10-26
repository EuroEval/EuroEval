"""All Greek dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, GREEK
from ..tasks import (
    COMMON_SENSE,
    HALLU,
    INSTRUCTION_FOLLOWING,
    KNOW,
    LA,
    NER,
    OPEN_ENDED_QA,
    RC,
    SENT,
    SUMM,
    TOOL_CALLING,
    TRANSLATION,
)

# Official datasets ###

WMT24PP_EN_EL_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-el",
    pretty_name="WMT24++-en-el",
    source="EuroEval/wmt24pp-en-el",
    task=TRANSLATION,
    languages=[GREEK],
    source_language=ENGLISH,
    target_language=GREEK,
    revision="65689ddb3f9a784fc5443dd6fbf4a8f2a3ffbe3a",
)

FLORES_EN_EL_CONFIG = TranslationDatasetConfig(
    name="flores-en-el",
    pretty_name="FLORES-en-el",
    source="EuroEval/flores-en-el",
    task=TRANSLATION,
    languages=[GREEK],
    source_language=ENGLISH,
    target_language=GREEK,
    revision="57b27dabc155cd709b100bdad8bafd8967106c66",
)

GREEK_SA_CONFIG = DatasetConfig(
    name="greek-sa",
    pretty_name="Greek Sentiment Analysis",
    source="EuroEval/greek-sa-mini",
    task=SENT,
    languages=[GREEK],
    labels=["negative", "positive"],
    revision="d72b98006eaa98449af88264ac2dd5710c6d2fdc",
)

SCALA_EL_CONFIG = DatasetConfig(
    name="scala-el",
    pretty_name="ScaLA-el",
    source="EuroEval/scala-el",
    task=LA,
    languages=[GREEK],
    revision="b965b88a1b76fde9c599a7b4516a951d7086e4da",
)

ELNER_CONFIG = DatasetConfig(
    name="elner",
    pretty_name="ElNER",
    source="EuroEval/elner-mini",
    task=NER,
    languages=[GREEK],
    revision="5de2a2d591d55e08d9af7eca78d9fafe90faef5e",
)

MULTI_WIKI_QA_EL_CONFIG = DatasetConfig(
    name="multi-wiki-qa-el",
    pretty_name="MultiWikiQA-el",
    source="EuroEval/multi-wiki-qa-el-mini",
    task=RC,
    languages=[GREEK],
    revision="46c8868c52cb2a4ae78db50545731cee226f39ae",
)

GREEK_WIKIPEDIA_CONFIG = DatasetConfig(
    name="greek-wikipedia",
    pretty_name="Greek Wikipedia",
    source="EuroEval/greek-wikipedia-mini",
    task=SUMM,
    languages=[GREEK],
    revision="cfb9c5c540732bd06465dbb9f655c92664a3ede3",
)

WINOGRANDE_EL_CONFIG = DatasetConfig(
    name="winogrande-el",
    pretty_name="Winogrande-el",
    source="EuroEval/winogrande-el",
    task=COMMON_SENSE,
    languages=[GREEK],
    revision="15b45b219b6625aea63065f515200ecf2e12fba3",
)

IFEVAL_EL_CONFIG = DatasetConfig(
    name="ifeval-el",
    pretty_name="IFEval-el",
    source="EuroEval/ifeval-el",
    task=INSTRUCTION_FOLLOWING,
    languages=[GREEK],
    train_split=None,
    val_split=None,
    revision="726fe019a7954c63bb00fe96e812a94d5dbd8b03",
)

RAGTRUTH_EL_CONFIG = DatasetConfig(
    name="ragtruth-el",
    pretty_name="RAGTruth-el",
    source="EuroEval/ragtruth-translated-hallucinations-el-mini",
    task=HALLU,
    languages=[GREEK],
    train_split=None,
    revision="4bcb7006803e653bc2caba4c90987013e7e17cb3",
)


GREEK_MMLU_CONFIG = DatasetConfig(
    name="greek-mmlu",
    pretty_name="GreekMMLU",
    source="EuroEval/greek-mmlu-mini",
    task=KNOW,
    languages=[GREEK],
    revision="7cd71aa0d7e914f914c46c36638dce957ca2cc5a",
)

CULTURAQA_CONFIG = DatasetConfig(
    name="culturaqa",
    pretty_name="CulturaQA",
    source="EuroEval/culturaqa-mini",
    task=OPEN_ENDED_QA,
    languages=[GREEK],
    prompt_prefix="Ακολουθούν ερωτήσεις με τις αντίστοιχες απαντήσεις.",
    prompt_template="Ερώτηση: {text}\nΑπάντηση: {target_text}",
    instruction_prompt=(
        "Απαντήστε στην παρακάτω ερώτηση με ακρίβεια και συντομία.\n\nΕρώτηση: {text}"
    ),
    revision="1ace5e667db22a1dd5a5d83a120908290f659e25",
)


# Unofficial datasets ###

MULTI_BFCL_EL_CONFIG = DatasetConfig(
    name="multi-bfcl-el",
    pretty_name="MultiBFCL-el",
    source="EuroEval/multi-bfcl-el-mini",
    task=TOOL_CALLING,
    languages=[GREEK],
    train_split=None,
    revision="6fa526e2b30f87f0ac9400dc1ee664d30d65a6f8",
)


WMT24PP_EL_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-el-en",
    pretty_name="WMT24++-el-en",
    source="EuroEval/wmt24pp-el-en",
    task=TRANSLATION,
    languages=[GREEK],
    source_language=GREEK,
    target_language=ENGLISH,
    unofficial=True,
    revision="c1dd1d0ed7ee76cd4858fe84aa12eb08d3187961",
)

FLORES_EL_EN_CONFIG = TranslationDatasetConfig(
    name="flores-el-en",
    pretty_name="FLORES-el-en",
    source="EuroEval/flores-el-en",
    task=TRANSLATION,
    languages=[GREEK],
    source_language=GREEK,
    target_language=ENGLISH,
    unofficial=True,
    revision="e11f7c2147d915af3c22fbbb1b5bb915cd0aa3a4",
)

GLOBAL_MMLU_EL_CONFIG = DatasetConfig(
    name="global-mmlu-el",
    pretty_name="GlobalMMLU-el",
    source="EuroEval/global-mmlu-el-mini",
    task=KNOW,
    languages=[GREEK],
    unofficial=True,
    revision="1b4d4efe3778c4d331a0688db2e11d85d160ebc4",
)

MULTI_IFEVAL_EL_CONFIG = DatasetConfig(
    name="multi-ifeval-el",
    pretty_name="MultiIFEval-el",
    source="EuroEval/multi-ifeval-el",
    task=INSTRUCTION_FOLLOWING,
    languages=[GREEK],
    train_split=None,
    val_split=None,
    unofficial=True,
    revision="cf465b59373bfd5f56441e046e9fd8aaca5bb3cc",
)

INCLUDE_EL_CONFIG = DatasetConfig(
    name="include-el",
    pretty_name="INCLUDE-el",
    source="EuroEval/include-el-mini",
    task=KNOW,
    languages=[GREEK],
    unofficial=True,
    revision="8b9ae4e87f921380bb5c4ee8bf3cf703fb51a025",
)

EU_MMLU_EL_CONFIG = DatasetConfig(
    name="eu-mmlu-el",
    pretty_name="EU-MMLU-el",
    source="EuroEval/eu-mmlu-el",
    task=KNOW,
    languages=[GREEK],
    unofficial=True,
    revision="ea2353933d96d55664adc97c8083dee5abbe1587",
)
