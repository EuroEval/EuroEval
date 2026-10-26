"""All Serbian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, SERBIAN
from ..tasks import (
    COMMON_SENSE,
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

WMT24PP_EN_SR_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-sr",
    pretty_name="WMT24++-en-sr",
    source="EuroEval/wmt24pp-en-sr",
    task=TRANSLATION,
    languages=[SERBIAN],
    source_language=ENGLISH,
    target_language=SERBIAN,
    revision="3331ac70b72bad889177f31b251d95946de39bc0",
)

FLORES_EN_SR_CONFIG = TranslationDatasetConfig(
    name="flores-en-sr",
    pretty_name="FLORES-en-sr",
    source="EuroEval/flores-en-sr",
    task=TRANSLATION,
    languages=[SERBIAN],
    source_language=ENGLISH,
    target_language=SERBIAN,
    revision="dcd87545c761a9d3b4690ade138be215cbc657af",
)

MMS_SR_CONFIG = DatasetConfig(
    name="mms-sr",
    pretty_name="MMS-sr",
    source="EuroEval/mms-sr-mini",
    task=SENT,
    languages=[SERBIAN],
    revision="308d050ff87020499c142ecc541dc8fd4307fcb4",
)

SCALA_SR_CONFIG = DatasetConfig(
    name="scala-sr",
    pretty_name="ScaLA-sr",
    source="EuroEval/scala-sr",
    task=LA,
    languages=[SERBIAN],
    revision="bdd7c94e9ce18dbb24fe220c288b7d6a2eb954f8",
)

UNER_SR_CONFIG = DatasetConfig(
    name="uner-sr",
    pretty_name="UNER-sr",
    source="EuroEval/uner-sr-mini",
    task=NER,
    languages=[SERBIAN],
    revision="cb568a992fdd700e6924fa1d2a5f20caf3231aad",
)

MULTI_WIKI_QA_SR_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sr",
    pretty_name="MultiWikiQA-sr",
    source="EuroEval/multi-wiki-qa-sr-mini",
    task=RC,
    languages=[SERBIAN],
    revision="3ba450d4948f14b1159f7768749321fae311493b",
)

LR_SUM_SR_CONFIG = DatasetConfig(
    name="lr-sum-sr",
    pretty_name="LRSum-sr",
    source="EuroEval/lr-sum-sr-mini",
    task=SUMM,
    languages=[SERBIAN],
    revision="efea9947e2c621509a6a2e88049d4730806ac97c",
)

WINOGRANDE_SR_CONFIG = DatasetConfig(
    name="winogrande-sr",
    pretty_name="Winogrande-sr",
    source="EuroEval/winogrande-sr",
    task=COMMON_SENSE,
    languages=[SERBIAN],
    labels=["a", "b"],
    revision="bd218bad5daeedbc3b99b18c55837db001c59cbb",
)

MULTI_IFEVAL_SR_CONFIG = DatasetConfig(
    name="multi-ifeval-sr",
    pretty_name="MultiIFEval-sr",
    source="EuroEval/multi-ifeval-sr",
    task=INSTRUCTION_FOLLOWING,
    languages=[SERBIAN],
    train_split=None,
    val_split=None,
    revision="8e66239baaa04a7bb9b9d7476bdb1d130bdca293",
)

RAGTRUTH_SR_CONFIG = DatasetConfig(
    name="ragtruth-sr",
    pretty_name="RAGTruth-sr",
    source="EuroEval/ragtruth-translated-hallucinations-sr-mini",
    task=HALLU,
    languages=[SERBIAN],
    train_split=None,
    revision="02573e0405c5397e12ec85f69da792354732087c",
)


INCLUDE_SR_CONFIG = DatasetConfig(
    name="include-sr",
    pretty_name="INCLUDE-sr",
    source="EuroEval/include-sr-mini",
    task=KNOW,
    languages=[SERBIAN],
    # Unlike the other Serbian knowledge datasets (which are in Latin), INCLUDE-sr is
    # written in Cyrillic, so we override the shared Latin KNOW template with a Cyrillic
    # one to keep the prompt in the same script as the content.
    prompt_prefix="Следе питања вишеструког избора (са одговорима).",
    prompt_template="Питање: {text}\nОдговор: {label}",
    instruction_prompt=(
        "Питање: {text}\n\nОдговорите на наведено питање користећи "
        "{labels_str}, и ништа друго."
    ),
    revision="4439f16ef895f9e0f5988304f2ca9809f9bfa2f8",
)


# Unofficial datasets ###

MULTI_BFCL_SR_CONFIG = DatasetConfig(
    name="multi-bfcl-sr",
    pretty_name="MultiBFCL-sr",
    source="EuroEval/multi-bfcl-sr-mini",
    task=TOOL_CALLING,
    languages=[SERBIAN],
    train_split=None,
    revision="1ff81e62df3b13baa114121d863c9fd0eaf80eec",
)


WMT24PP_SR_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-sr-en",
    pretty_name="WMT24++-sr-en",
    source="EuroEval/wmt24pp-sr-en",
    task=TRANSLATION,
    languages=[SERBIAN],
    source_language=SERBIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="ccd09d7ab73265de30f7b23f62233fd26b3843c6",
)

FLORES_SR_EN_CONFIG = TranslationDatasetConfig(
    name="flores-sr-en",
    pretty_name="FLORES-sr-en",
    source="EuroEval/flores-sr-en",
    task=TRANSLATION,
    languages=[SERBIAN],
    source_language=SERBIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="b793115d939fe949c9e821ce2b375a9b2614f3f7",
)

MMLU_SR_CONFIG = DatasetConfig(
    name="mmlu-sr",
    pretty_name="MMLU-sr",
    source="EuroEval/mmlu-sr-mini",
    task=KNOW,
    languages=[SERBIAN],
    unofficial=True,
    revision="08ad59a342853e65146d6f32693cdbc63aef9471",
)
