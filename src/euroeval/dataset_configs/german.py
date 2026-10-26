"""All German dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, GERMAN
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

WMT24PP_EN_DE_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-de",
    pretty_name="WMT24++-en-de",
    source="EuroEval/wmt24pp-en-de",
    task=TRANSLATION,
    languages=[GERMAN],
    source_language=ENGLISH,
    target_language=GERMAN,
    revision="1ada1a93e6f552476901f19d0ef749c315beb72b",
)

FLORES_EN_DE_CONFIG = TranslationDatasetConfig(
    name="flores-en-de",
    pretty_name="FLORES-en-de",
    source="EuroEval/flores-en-de",
    task=TRANSLATION,
    languages=[GERMAN],
    source_language=ENGLISH,
    target_language=GERMAN,
    revision="bdb329db12f87d9bb65dc91b7f0b73970e1a48fd",
)

SB10K_CONFIG = DatasetConfig(
    name="sb10k",
    pretty_name="SB10K",
    source="EuroEval/sb10k-mini",
    task=SENT,
    languages=[GERMAN],
    revision="1a271bfd425ff6e75fe2ff0fc2ede0d165a2870b",
)

SCALA_DE_CONFIG = DatasetConfig(
    name="scala-de",
    pretty_name="ScaLA-de",
    source="EuroEval/scala-de",
    task=LA,
    languages=[GERMAN],
    revision="7166de01694a6d6cc06fd368b500cec88f6b1927",
)

GERMEVAL_CONFIG = DatasetConfig(
    name="germeval",
    pretty_name="GermEval",
    source="EuroEval/germeval-mini",
    task=NER,
    languages=[GERMAN],
    revision="f8a5f7e7258c8b0a24fb1b00267324ede7398b5e",
)

GERMANQUAD_CONFIG = DatasetConfig(
    name="germanquad",
    pretty_name="GermanQuAD",
    source="EuroEval/germanquad-mini",
    task=RC,
    languages=[GERMAN],
    revision="20ef897a94ac01ec58f6bfd1f44a4c4332279742",
)

MLSUM_DE_CONFIG = DatasetConfig(
    name="mlsum-de",
    pretty_name="MLSUM-de",
    source="EuroEval/mlsum-mini",
    task=SUMM,
    languages=[GERMAN],
    revision="b76fd41b5b3ffe1063de62b39a0d1f3e2651886d",
)

MULTI_IFEVAL_DE_CONFIG = DatasetConfig(
    name="multi-ifeval-de",
    pretty_name="MultiIFEval-de",
    source="EuroEval/multi-ifeval-de",
    task=INSTRUCTION_FOLLOWING,
    languages=[GERMAN],
    train_split=None,
    val_split=None,
    revision="7165a399fdc05f3e968eb5873158ad632545a74f",
)

VALEU_DE_CONFIG = DatasetConfig(
    name="valeu-de",
    pretty_name="VaLEU-de",
    source="EuroEval/european-values-de",
    task=EUROPEAN_VALUES,
    languages=[GERMAN],
    train_split=None,
    val_split=None,
    bootstrap_samples=False,
    instruction_prompt="{text}",
    revision="348e69fee805bdb30c21936f78b7acfa8e806cd6",
)

ZEBRA_PUZZLE_EASY_DE_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-de",
    pretty_name="ZebraPuzzlesEasy-de",
    source="EuroEval/zebra-puzzles-easy-de",
    task=LOGIC,
    languages=[GERMAN],
    revision="432cb8f0362086b9358a508348d1f684e81a84cd",
)


WINOGRANDE_DE_CONFIG = DatasetConfig(
    name="winogrande-de",
    pretty_name="Winogrande-de",
    source="EuroEval/winogrande-de",
    task=COMMON_SENSE,
    languages=[GERMAN],
    labels=["a", "b"],
    revision="a89f6c6c47c7f16e5e804771ec72e390217e6e0b",
)

INCLUDE_DE_CONFIG = DatasetConfig(
    name="include-de",
    pretty_name="INCLUDE-de",
    source="EuroEval/include-de-mini",
    task=KNOW,
    languages=[GERMAN],
    revision="445cd3bdfc15aaebeee329fcc99bc9670fdfd5e2",
)

MULTILOKO_DE_CONFIG = DatasetConfig(
    name="multiloko-de",
    pretty_name="MultiLoKo-de",
    source="EuroEval/multiloko-de-mini",
    task=KNOW,
    languages=[GERMAN],
    val_split=None,
    revision="f22fa7f9a00acf59931e8871e6d2a87483d64b6d",
)

RAGTRUTH_DE_CONFIG = DatasetConfig(
    name="ragtruth-de",
    pretty_name="RAGTruth-de",
    source="EuroEval/ragtruth-translated-hallucinations-de-mini",
    task=HALLU,
    languages=[GERMAN],
    train_split=None,
    revision="f83d96fcc8608c5ecd3a10e7ee0aaf318b98a5b6",
)


# Unofficial datasets ###

MULTI_BFCL_DE_CONFIG = DatasetConfig(
    name="multi-bfcl-de",
    pretty_name="MultiBFCL-de",
    source="EuroEval/multi-bfcl-de-mini",
    task=TOOL_CALLING,
    languages=[GERMAN],
    train_split=None,
    revision="fa316472928e11260ed92b8c6e2c3df214284cc8",
)


WMT24PP_DE_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-de-en",
    pretty_name="WMT24++-de-en",
    source="EuroEval/wmt24pp-de-en",
    task=TRANSLATION,
    languages=[GERMAN],
    source_language=GERMAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="85b06fb037bee5d300cfb193491b439bc1ad5ddc",
)

FLORES_DE_EN_CONFIG = TranslationDatasetConfig(
    name="flores-de-en",
    pretty_name="FLORES-de-en",
    source="EuroEval/flores-de-en",
    task=TRANSLATION,
    languages=[GERMAN],
    source_language=GERMAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="003268e11212f93bf94d86961f998046adafa021",
)

MMLU_DE_CONFIG = DatasetConfig(
    name="mmlu-de",
    pretty_name="MMLU-de",
    source="EuroEval/mmlu-de-mini",
    task=KNOW,
    languages=[GERMAN],
    unofficial=True,
    revision="6260bf03d4587384e8e5ebbaf0e1d24ee302df34",
)

HELLASWAG_DE_CONFIG = DatasetConfig(
    name="hellaswag-de",
    pretty_name="HellaSwag-de",
    source="EuroEval/hellaswag-de-mini",
    task=COMMON_SENSE,
    languages=[GERMAN],
    unofficial=True,
    revision="31c6bcee97a3d281c346a2c69d85111943631a91",
)

IFEVAL_DE_CONFIG = DatasetConfig(
    name="ifeval-de",
    pretty_name="IFEval-de",
    source="EuroEval/ifeval-de",
    task=INSTRUCTION_FOLLOWING,
    languages=[GERMAN],
    train_split=None,
    val_split=None,
    unofficial=True,
    revision="c7400e5e265da2ec29939faf8fe85f5bb7252805",
)

XQUAD_DE_CONFIG = DatasetConfig(
    name="xquad-de",
    pretty_name="XQuAD-de",
    source="EuroEval/xquad-de",
    task=RC,
    languages=[GERMAN],
    unofficial=True,
    revision="32e0c1165c81f1a76df56c1d0ca6d1f18e56dd57",
)

ARC_DE_CONFIG = DatasetConfig(
    name="arc-de",
    pretty_name="ARC-de",
    source="EuroEval/arc-de-mini",
    task=KNOW,
    languages=[GERMAN],
    unofficial=True,
    revision="48ae8fda6d72cc99e3f4517bd51b4f1633d209bf",
)

BELEBELE_DE_CONFIG = DatasetConfig(
    name="belebele-de",
    pretty_name="Belebele-de",
    source="EuroEval/belebele-de-mini",
    task=MCRC,
    languages=[GERMAN],
    unofficial=True,
    revision="708ad5d91f5acc44168bed38ba494f23e3da4224",
)

MULTI_WIKI_QA_DE_CONFIG = DatasetConfig(
    name="multi-wiki-qa-de",
    pretty_name="MultiWikiQA-de",
    source="EuroEval/multi-wiki-qa-de-mini",
    task=RC,
    languages=[GERMAN],
    unofficial=True,
    revision="f9672639c57be9a55aa3cc62b1f78bbf1cd7f0dd",
)

GOLDENSWAG_DE_CONFIG = DatasetConfig(
    name="goldenswag-de",
    pretty_name="GoldenSwag-de",
    source="EuroEval/goldenswag-de-mini",
    task=COMMON_SENSE,
    languages=[GERMAN],
    unofficial=True,
    revision="730bd9300d547767ae8cb75ca34d7a05aa04a488",
)

GERLANGMOD_DE_CONFIG = DatasetConfig(
    name="gerlangmod-de",
    pretty_name="GerLangMod-de",
    source="EuroEval/gerlangmod-de",
    task=GED,
    languages=[GERMAN],
    unofficial=True,
    revision="4ccbbe54e85277f3c8bddd5997b3da38e589729b",
)

ZEBRA_PUZZLE_HARD_DE_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-de",
    pretty_name="ZebraPuzzlesHard-de",
    source="EuroEval/zebra-puzzles-hard-de",
    task=LOGIC,
    languages=[GERMAN],
    unofficial=True,
    revision="22f88a67987e3f4ef8388f82bedbe4ce118620cd",
)

EU_MMLU_DE_CONFIG = DatasetConfig(
    name="eu-mmlu-de",
    pretty_name="EU-MMLU-de",
    source="EuroEval/eu-mmlu-de",
    task=KNOW,
    languages=[GERMAN],
    unofficial=True,
    revision="a19ec360f053b92aba0161df22543400b6da581c",
)
