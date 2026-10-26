"""All English dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import ENGLISH
from ..tasks import (
    COMMON_SENSE,
    EUROPEAN_VALUES,
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
    WIC,
)

# Official datasets ###

SST5_CONFIG = DatasetConfig(
    name="sst5",
    pretty_name="SST-5",
    source="EuroEval/sst5-mini",
    task=SENT,
    languages=[ENGLISH],
    revision="a1bf6f70a8b0b9d0ec79388a74caa58237fa0449",
)

SCALA_EN_CONFIG = DatasetConfig(
    name="scala-en",
    pretty_name="ScaLA-en",
    source="EuroEval/scala-en",
    task=LA,
    languages=[ENGLISH],
    revision="af9916d318b39dc07860291fca0f9202030ae206",
)

CONLL_EN_CONFIG = DatasetConfig(
    name="conll-en",
    pretty_name="CoNLL-en",
    source="EuroEval/conll-en-mini",
    task=NER,
    languages=[ENGLISH],
    revision="a4b3ec366adf39bf988be5361744bc6045dbde5b",
)

SQUAD_CONFIG = DatasetConfig(
    name="squad",
    pretty_name="SQuAD",
    source="EuroEval/squad-mini",
    task=RC,
    languages=[ENGLISH],
    revision="47fa2f049702962c2ff997f4ac31b938013506ed",
)

CNN_DAILYMAIL_CONFIG = DatasetConfig(
    name="cnn-dailymail",
    pretty_name="CNN/DailyMail",
    source="EuroEval/cnn-dailymail-mini",
    task=SUMM,
    languages=[ENGLISH],
    revision="9c5b38fb91faf82a199bd381078ab398032f02c5",
)

LIFE_IN_THE_UK_CONFIG = DatasetConfig(
    name="life-in-the-uk",
    pretty_name="Life in the UK",
    source="EuroEval/life-in-the-uk",
    task=KNOW,
    languages=[ENGLISH],
    revision="091bb552e21027b1b5b34faea3a40b6b6d19124a",
)

HELLASWAG_CONFIG = DatasetConfig(
    name="hellaswag",
    pretty_name="HellaSwag",
    source="EuroEval/hellaswag-mini",
    task=COMMON_SENSE,
    languages=[ENGLISH],
    revision="43c6f6480af1e589bfda7650146f291e2ce6ce37",
)

IFEVAL_CONFIG = DatasetConfig(
    name="ifeval",
    pretty_name="IFEval",
    source="EuroEval/ifeval-en",
    task=INSTRUCTION_FOLLOWING,
    languages=[ENGLISH],
    train_split=None,
    val_split=None,
    revision="78c19e06929b759b1f467cfec4ebcfbe8cee7e9b",
)

BFCL_V2_CONFIG = DatasetConfig(
    name="bfcl-v2",
    pretty_name="BFCL-v2",
    source="EuroEval/bfcl-v2",
    task=TOOL_CALLING,
    languages=[ENGLISH],
    revision="7fda2a8653f9ed62388903e0c747c90191dac5db",
)

VALEU_EN_CONFIG = DatasetConfig(
    name="valeu-en",
    pretty_name="VaLEU-en",
    source="EuroEval/european-values-en",
    task=EUROPEAN_VALUES,
    languages=[ENGLISH],
    train_split=None,
    val_split=None,
    bootstrap_samples=False,
    instruction_prompt="{text}",
    revision="6377357fa6cf09ac117ea3d4a798b1d6e4a5fc63",
)

ZEBRA_PUZZLE_EASY_EN_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-en",
    pretty_name="ZebraPuzzlesEasy-en",
    source="EuroEval/zebra-puzzles-easy-en",
    task=LOGIC,
    languages=[ENGLISH],
    revision="0ce26174115327f867e18447f3e05269afcb0022",
)

RAGTRUTH_EN_CONFIG = DatasetConfig(
    name="ragtruth-en",
    pretty_name="RAGTruth-en",
    source="EuroEval/ragtruth-translated-hallucinations-en-mini",
    task=HALLU,
    languages=[ENGLISH],
    train_split=None,
    revision="3fa775f6311f81b8d83b7bb77abb8f14d63f5232",
)


# Unofficial datasets ###

MULTI_IFEVAL_EN_CONFIG = DatasetConfig(
    name="multi-ifeval-en",
    pretty_name="MultiIFEval-en",
    source="EuroEval/multi-ifeval-en",
    task=INSTRUCTION_FOLLOWING,
    languages=[ENGLISH],
    train_split=None,
    val_split=None,
    unofficial=True,
    revision="34165a9d58e32b48ffb70e1524b4a4e3d29ad5d2",
)

XQUAD_EN_CONFIG = DatasetConfig(
    name="xquad-en",
    pretty_name="XQuAD-en",
    source="EuroEval/xquad-en",
    task=RC,
    languages=[ENGLISH],
    unofficial=True,
    revision="818d4542f4527c9cdcfb103569db21e87a8eaf64",
)

ARC_CONFIG = DatasetConfig(
    name="arc",
    pretty_name="ARC",
    source="EuroEval/arc-mini",
    task=KNOW,
    languages=[ENGLISH],
    unofficial=True,
    revision="6efdff590df2004117d669dad20604023399728f",
)

BELEBELE_CONFIG = DatasetConfig(
    name="belebele-en",
    pretty_name="Belebele-en",
    source="EuroEval/belebele-mini",
    task=MCRC,
    languages=[ENGLISH],
    unofficial=True,
    revision="b391617c8caf89cfe946c811b60d9f67ad7920b4",
)

MMLU_CONFIG = DatasetConfig(
    name="mmlu",
    pretty_name="MMLU",
    source="EuroEval/mmlu-mini",
    task=KNOW,
    languages=[ENGLISH],
    unofficial=True,
    revision="63ecc4c6247f391d5cee6cb90e7912a60e515117",
)

MMLU_PRO_CONFIG = DatasetConfig(
    name="mmlu-pro",
    pretty_name="MMLU-Pro",
    source="EuroEval/mmlu-pro-mini",
    task=KNOW,
    languages=[ENGLISH],
    labels=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    unofficial=True,
    revision="0a51cbebea6301ec2fb0d7169698201d9bc0f5b7",
)

MULTI_WIKI_QA_EN_CONFIG = DatasetConfig(
    name="multi-wiki-qa-en",
    pretty_name="MultiWikiQA-en",
    source="EuroEval/multi-wiki-qa-en-mini",
    task=RC,
    languages=[ENGLISH],
    unofficial=True,
    revision="170283c05df1c37f92ecacfbbbb6025bd85b3b95",
)

WINOGRANDE_CONFIG = DatasetConfig(
    name="winogrande",
    pretty_name="Winogrande-en",
    source="EuroEval/winogrande-en",
    task=COMMON_SENSE,
    languages=[ENGLISH],
    labels=["a", "b"],
    unofficial=True,
    revision="3c353d33e876f2417ecd3caeb9f87225f0c57569",
)

MULTILOKO_EN_CONFIG = DatasetConfig(
    name="multiloko-en",
    pretty_name="MultiLoKo-en",
    source="EuroEval/multiloko-en-mini",
    task=KNOW,
    languages=[ENGLISH],
    val_split=None,
    unofficial=True,
    revision="8db4cc3cbb2ef4769a3dfeeb13cbf501befc6790",
)

WIC_CONFIG = DatasetConfig(
    name="wic",
    pretty_name="WiC",
    source="EuroEval/wic",
    task=WIC,
    languages=[ENGLISH],
    unofficial=True,
    revision="3e13b2d016f59c870d87160f8febb1086616a61d",
)

ZEBRA_PUZZLE_HARD_EN_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-en",
    pretty_name="ZebraPuzzlesHard-en",
    source="EuroEval/zebra-puzzles-hard-en",
    task=LOGIC,
    languages=[ENGLISH],
    unofficial=True,
    revision="28e8bd5b512346c2c15162cb78d7c6336cad0df1",
)
