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
    TRANSLATION,
)

# Official datasets ###

SB10K_CONFIG = DatasetConfig(
    name="sb10k",
    pretty_name="SB10K",
    source="EuroEval/sb10k-mini",
    task=SENT,
    languages=[GERMAN],
)

SCALA_DE_CONFIG = DatasetConfig(
    name="scala-de",
    pretty_name="ScaLA-de",
    source="EuroEval/scala-de",
    task=LA,
    languages=[GERMAN],
)

GERMEVAL_CONFIG = DatasetConfig(
    name="germeval",
    pretty_name="GermEval",
    source="EuroEval/germeval-mini",
    task=NER,
    languages=[GERMAN],
)

GERMANQUAD_CONFIG = DatasetConfig(
    name="germanquad",
    pretty_name="GermanQuAD",
    source="EuroEval/germanquad-mini",
    task=RC,
    languages=[GERMAN],
)

MLSUM_DE_CONFIG = DatasetConfig(
    name="mlsum-de",
    pretty_name="MLSUM-de",
    source="EuroEval/mlsum-mini",
    task=SUMM,
    languages=[GERMAN],
)

MULTI_IFEVAL_DE_CONFIG = DatasetConfig(
    name="multi-ifeval-de",
    pretty_name="MultiIFEval-de",
    source="EuroEval/multi-ifeval-de",
    task=INSTRUCTION_FOLLOWING,
    languages=[GERMAN],
    train_split=None,
    val_split=None,
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
)

ZEBRA_PUZZLE_EASY_DE_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-de",
    pretty_name="ZebraPuzzlesEasy-de",
    source="EuroEval/zebra-puzzles-easy-de",
    task=LOGIC,
    languages=[GERMAN],
)


WINOGRANDE_DE_CONFIG = DatasetConfig(
    name="winogrande-de",
    pretty_name="Winogrande-de",
    source="EuroEval/winogrande-de",
    task=COMMON_SENSE,
    languages=[GERMAN],
    labels=["a", "b"],
)

INCLUDE_DE_CONFIG = DatasetConfig(
    name="include-de",
    pretty_name="INCLUDE-de",
    source="EuroEval/include-de-mini",
    task=KNOW,
    languages=[GERMAN],
)

MULTILOKO_DE_CONFIG = DatasetConfig(
    name="multiloko-de",
    pretty_name="MultiLoKo-de",
    source="EuroEval/multiloko-de-mini",
    task=KNOW,
    languages=[GERMAN],
    val_split=None,
)

WMT24PP_EN_DE_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-de",
    pretty_name="WMT24++-en-de",
    source="EuroEval/wmt24pp-en-de",
    task=TRANSLATION,
    languages=[GERMAN],
    source_language=ENGLISH,
    target_language=GERMAN,
)


# Unofficial datasets ###

WMT24PP_DE_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-de-en",
    pretty_name="WMT24++-de-en",
    source="EuroEval/wmt24pp-de-en",
    task=TRANSLATION,
    languages=[GERMAN],
    source_language=GERMAN,
    target_language=ENGLISH,
    unofficial=True,
)

MMLU_DE_CONFIG = DatasetConfig(
    name="mmlu-de",
    pretty_name="MMLU-de",
    source="EuroEval/mmlu-de-mini",
    task=KNOW,
    languages=[GERMAN],
    unofficial=True,
)

HELLASWAG_DE_CONFIG = DatasetConfig(
    name="hellaswag-de",
    pretty_name="HellaSwag-de",
    source="EuroEval/hellaswag-de-mini",
    task=COMMON_SENSE,
    languages=[GERMAN],
    unofficial=True,
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
)

XQUAD_DE_CONFIG = DatasetConfig(
    name="xquad-de",
    pretty_name="XQuAD-de",
    source="EuroEval/xquad-de",
    task=RC,
    languages=[GERMAN],
    unofficial=True,
)

ARC_DE_CONFIG = DatasetConfig(
    name="arc-de",
    pretty_name="ARC-de",
    source="EuroEval/arc-de-mini",
    task=KNOW,
    languages=[GERMAN],
    unofficial=True,
)

BELEBELE_DE_CONFIG = DatasetConfig(
    name="belebele-de",
    pretty_name="Belebele-de",
    source="EuroEval/belebele-de-mini",
    task=MCRC,
    languages=[GERMAN],
    unofficial=True,
)

MULTI_WIKI_QA_DE_CONFIG = DatasetConfig(
    name="multi-wiki-qa-de",
    pretty_name="MultiWikiQA-de",
    source="EuroEval/multi-wiki-qa-de-mini",
    task=RC,
    languages=[GERMAN],
    unofficial=True,
)

GOLDENSWAG_DE_CONFIG = DatasetConfig(
    name="goldenswag-de",
    pretty_name="GoldenSwag-de",
    source="EuroEval/goldenswag-de-mini",
    task=COMMON_SENSE,
    languages=[GERMAN],
    unofficial=True,
)

GERLANGMOD_DE_CONFIG = DatasetConfig(
    name="gerlangmod-de",
    pretty_name="GerLangMod-de",
    source="EuroEval/gerlangmod-de",
    task=GED,
    languages=[GERMAN],
    unofficial=True,
)

ZEBRA_PUZZLE_HARD_DE_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-de",
    pretty_name="ZebraPuzzlesHard-de",
    source="EuroEval/zebra-puzzles-hard-de",
    task=LOGIC,
    languages=[GERMAN],
    unofficial=True,
)

EU_MMLU_DE_CONFIG = DatasetConfig(
    name="eu-mmlu-de",
    pretty_name="EU-MMLU-de",
    source="EuroEval/eu-mmlu-de",
    task=KNOW,
    languages=[GERMAN],
    unofficial=True,
)

RAGTRUTH_DE_CONFIG = DatasetConfig(
    name="ragtruth-de",
    pretty_name="RAGTruth-de",
    source="EuroEval/ragtruth-translated-hallucinations-de-mini",
    task=HALLU,
    languages=[GERMAN],
    train_split=None,
    unofficial=True,
)


FLORES_EN_DE_CONFIG = TranslationDatasetConfig(
    name="flores-en-de",
    pretty_name="FLORES-en-de",
    source="EuroEval/flores-en-de",
    task=TRANSLATION,
    languages=[GERMAN],
    source_language=ENGLISH,
    target_language=GERMAN,
    unofficial=True,
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
)
