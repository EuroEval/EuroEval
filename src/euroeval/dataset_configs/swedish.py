"""All Swedish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, SWEDISH
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

WMT24PP_EN_SV_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-sv",
    pretty_name="WMT24++-en-sv",
    source="EuroEval/wmt24pp-en-sv",
    task=TRANSLATION,
    languages=[SWEDISH],
    source_language=ENGLISH,
    target_language=SWEDISH,
    revision="ae42bb84bbd5731841a13eb24b5c4af67a559e6c",
)

FLORES_EN_SV_CONFIG = TranslationDatasetConfig(
    name="flores-en-sv",
    pretty_name="FLORES-en-sv",
    source="EuroEval/flores-en-sv",
    task=TRANSLATION,
    languages=[SWEDISH],
    source_language=ENGLISH,
    target_language=SWEDISH,
    revision="2c43505a8816ec39b7fb112092fa23375717ccb3",
)

SWEREC_CONFIG = DatasetConfig(
    name="swerec",
    pretty_name="SweReC",
    source="EuroEval/swerec-mini",
    task=SENT,
    languages=[SWEDISH],
    revision="ff5e9c50b5924accd23b396487f7a2e21e649785",
)

SCALA_SV_CONFIG = DatasetConfig(
    name="scala-sv",
    pretty_name="ScaLA-sv",
    source="EuroEval/scala-sv",
    task=LA,
    languages=[SWEDISH],
    revision="a0a01f242a499284defa9162514fe29b50addcea",
)

SUC3_CONFIG = DatasetConfig(
    name="suc3",
    pretty_name="SUC3",
    source="EuroEval/suc3-mini",
    task=NER,
    languages=[SWEDISH],
    revision="d8243105818293a2f5b4ba7ab6406e73098c9b8b",
)

MULTI_WIKI_QA_SV_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sv",
    pretty_name="MultiWikiQA-sv",
    source="EuroEval/multi-wiki-qa-sv-mini",
    task=RC,
    languages=[SWEDISH],
    revision="42cb590f1cf4a3d7a2d4525261cb2e47d9ccc98b",
)

SWEDN_CONFIG = DatasetConfig(
    name="swedn",
    pretty_name="SweDN",
    source="EuroEval/swedn-mini",
    task=SUMM,
    languages=[SWEDISH],
    revision="339a0ba4c4b3a1071f7c58c5e3a4d4479a44d6c3",
)

MULTI_IFEVAL_SV_CONFIG = DatasetConfig(
    name="multi-ifeval-sv",
    pretty_name="MultiIFEval-sv",
    source="EuroEval/multi-ifeval-sv",
    task=INSTRUCTION_FOLLOWING,
    languages=[SWEDISH],
    train_split=None,
    val_split=None,
    revision="b5f9f3571b40e803cbabd14bc4392b5004b850a5",
)

VALEU_SV_CONFIG = DatasetConfig(
    name="valeu-sv",
    pretty_name="VaLEU-sv",
    source="EuroEval/european-values-sv",
    task=EUROPEAN_VALUES,
    languages=[SWEDISH],
    train_split=None,
    val_split=None,
    bootstrap_samples=False,
    instruction_prompt="{text}",
    revision="c8e35a37174d666acc6c71da7ef90a466183d115",
)

ZEBRA_PUZZLE_EASY_SV_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-sv",
    pretty_name="ZebraPuzzlesEasy-sv",
    source="EuroEval/zebra-puzzles-easy-sv",
    task=LOGIC,
    languages=[SWEDISH],
    revision="465200f1d7a51dc8dbd83058fbe4e37ba5aef2a4",
)

RAGTRUTH_SV_CONFIG = DatasetConfig(
    name="ragtruth-sv",
    pretty_name="RAGTruth-sv",
    source="EuroEval/ragtruth-translated-hallucinations-sv-mini",
    task=HALLU,
    languages=[SWEDISH],
    train_split=None,
    revision="e723767cb5d660f3a4e07eadd2ce30e3cb7e0390",
)


WINOGRANDE_SV_CONFIG = DatasetConfig(
    name="winogrande-sv",
    pretty_name="Winogrande-sv",
    source="EuroEval/winogrande-sv",
    task=COMMON_SENSE,
    languages=[SWEDISH],
    labels=["a", "b"],
    revision="c9d27f4a1bae591455dac107179f34c2d65dcc0e",
)

SKOLPROV_CONFIG = DatasetConfig(
    name="skolprov",
    pretty_name="Skolprov",
    source="EuroEval/skolprov",
    task=KNOW,
    languages=[SWEDISH],
    revision="baefd89c2c80f48913a1a3f6088f03f5c14be4d2",
)

SWEDISH_FACTS_CONFIG = DatasetConfig(
    name="swedish-facts",
    pretty_name="Swedish Facts",
    source="EuroEval/swedish-facts",
    task=KNOW,
    languages=[SWEDISH],
    revision="9a558083aed9ab8e9aaadec6b56e6fe4eb2253fc",
)


# Unofficial datasets ###

MULTI_BFCL_SV_CONFIG = DatasetConfig(
    name="multi-bfcl-sv",
    pretty_name="MultiBFCL-sv",
    source="EuroEval/multi-bfcl-sv-mini",
    task=TOOL_CALLING,
    languages=[SWEDISH],
    train_split=None,
    revision="202580509c022868cf8e0ddf12c983a997937fe5",
)


WMT24PP_SV_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-sv-en",
    pretty_name="WMT24++-sv-en",
    source="EuroEval/wmt24pp-sv-en",
    task=TRANSLATION,
    languages=[SWEDISH],
    source_language=SWEDISH,
    target_language=ENGLISH,
    unofficial=True,
    revision="b429eba7cac9faf82cc9a5621b515cd4b980529f",
)

FLORES_SV_EN_CONFIG = TranslationDatasetConfig(
    name="flores-sv-en",
    pretty_name="FLORES-sv-en",
    source="EuroEval/flores-sv-en",
    task=TRANSLATION,
    languages=[SWEDISH],
    source_language=SWEDISH,
    target_language=ENGLISH,
    unofficial=True,
    revision="3ef8adc25c012a658aaaf96c9f1b8b5912b47a73",
)

MMLU_SV_CONFIG = DatasetConfig(
    name="mmlu-sv",
    pretty_name="MMLU-sv",
    source="EuroEval/mmlu-sv-mini",
    task=KNOW,
    languages=[SWEDISH],
    unofficial=True,
    revision="6dc89c52c425b951df371002fd81e2e7fa9d5ef7",
)

HELLASWAG_SV_CONFIG = DatasetConfig(
    name="hellaswag-sv",
    pretty_name="HellaSwag-sv",
    source="EuroEval/hellaswag-sv-mini",
    task=COMMON_SENSE,
    languages=[SWEDISH],
    unofficial=True,
    revision="dd58b6d9fc0f9b152dbabf4f7318ccdfdf68b3e0",
)

IFEVAL_SV_CONFIG = DatasetConfig(
    name="ifeval-sv",
    pretty_name="IFEval-sv",
    source="EuroEval/ifeval-sv",
    task=INSTRUCTION_FOLLOWING,
    languages=[SWEDISH],
    train_split=None,
    val_split=None,
    unofficial=True,
    revision="4e6df2a537eba0fd21dde77a0b2d30c3439c2eb5",
)

SCHIBSTED_SV_CONFIG = DatasetConfig(
    name="schibsted-sv",
    pretty_name="Schibsted-sv",
    source="EuroEval/schibsted-article-summaries-sv",
    task=SUMM,
    languages=[SWEDISH],
    unofficial=True,
    revision="3d49df17aff3468a2d6dc1e753c23699bebb9c3f",
)

SCHIBSTED_SEO_TITLE_SV_CONFIG = DatasetConfig(
    name="svd-seo-title",
    pretty_name="SVD SEO Title",
    source="EuroEval/svd-seo-title",
    task=SUMM,
    languages=[SWEDISH],
    unofficial=True,
    max_generated_tokens=64,
    prompt_prefix="Nedan följer artiklar med tillhörande SEO-rubriker.",
    prompt_template="Artikel: {text}\nSEO-rubrik: {target_text}",
    instruction_prompt=(
        "Artikel: {text}\n\nSkriv en SEO-rubrik för ovanstående artikel."
    ),
    revision="10d4b8c081c2000f748478ccda6f691535b3262e",
)

ARC_SV_CONFIG = DatasetConfig(
    name="arc-sv",
    pretty_name="ARC-sv",
    source="EuroEval/arc-sv-mini",
    task=KNOW,
    languages=[SWEDISH],
    unofficial=True,
    revision="d6be830d4dd82c40998b64bdb8d3bb60af9713dd",
)

BELEBELE_SV_CONFIG = DatasetConfig(
    name="belebele-sv",
    pretty_name="Belebele-sv",
    source="EuroEval/belebele-sv-mini",
    task=MCRC,
    languages=[SWEDISH],
    unofficial=True,
    revision="d2f102a7cba7b323671c1a6ddac2f55bf9134073",
)

SCANDIQA_SV_CONFIG = DatasetConfig(
    name="scandiqa-sv",
    pretty_name="ScandiQA-sv",
    source="EuroEval/scandiqa-sv-mini",
    task=RC,
    languages=[SWEDISH],
    unofficial=True,
    revision="5c13f1fdba7621ba6b72b77b57bbe68750d1d04c",
)

GOLDENSWAG_SV_CONFIG = DatasetConfig(
    name="goldenswag-sv",
    pretty_name="GoldenSwag-sv",
    source="EuroEval/goldenswag-sv-mini",
    task=COMMON_SENSE,
    languages=[SWEDISH],
    unofficial=True,
    revision="4b14ec9fc72bea80c25d4cb6f72eca15bd8a43ff",
)

MULTILOKO_SV_CONFIG = DatasetConfig(
    name="multiloko-sv",
    pretty_name="MultiLoKo-sv",
    source="EuroEval/multiloko-sv-mini",
    task=KNOW,
    languages=[SWEDISH],
    val_split=None,
    unofficial=True,
    revision="bcf0b342f33e9f77b3cd8fdf0e0b244f062e8ef4",
)

GERLANGMOD_SV_CONFIG = DatasetConfig(
    name="gerlangmod-sv",
    pretty_name="GerLangMod-sv",
    source="EuroEval/gerlangmod-sv",
    task=GED,
    languages=[SWEDISH],
    unofficial=True,
    revision="9e5b0a2cd2565cf4dfe5960d021a8fc68e90903a",
)

ZEBRA_PUZZLE_HARD_SV_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-sv",
    pretty_name="ZebraPuzzlesHard-sv",
    source="EuroEval/zebra-puzzles-hard-sv",
    task=LOGIC,
    languages=[SWEDISH],
    unofficial=True,
    revision="4b34e885c02bc0ffb5b2e0c8033bf97ed059e077",
)
