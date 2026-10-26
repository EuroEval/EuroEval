"""All Italian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, ITALIAN
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
    WIC,
)

# Official datasets ###

WMT24PP_EN_IT_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-it",
    pretty_name="WMT24++-en-it",
    source="EuroEval/wmt24pp-en-it",
    task=TRANSLATION,
    languages=[ITALIAN],
    source_language=ENGLISH,
    target_language=ITALIAN,
    revision="dbb0a71d1db4314469f163e870cf0ba3d73a858d",
)

FLORES_EN_IT_CONFIG = TranslationDatasetConfig(
    name="flores-en-it",
    pretty_name="FLORES-en-it",
    source="EuroEval/flores-en-it",
    task=TRANSLATION,
    languages=[ITALIAN],
    source_language=ENGLISH,
    target_language=ITALIAN,
    revision="d6e9b0596e7143153028d0d262c921035d13ffc7",
)

SENTIPOLC_CONFIG = DatasetConfig(
    name="sentipolc16",
    pretty_name="Sentipolc16",
    source="EuroEval/sentipolc16-mini",
    task=SENT,
    languages=[ITALIAN],
    revision="be2f860e008314868d8fd87aa87aba74cc9fd793",
)

SCALA_IT_CONFIG = DatasetConfig(
    name="scala-it",
    pretty_name="ScaLA-it",
    source="EuroEval/scala-it",
    task=LA,
    languages=[ITALIAN],
    revision="7d31ea7e56580f0f8622d18be40bf77599d3d794",
)

MULTINERD_IT_CONFIG = DatasetConfig(
    name="multinerd-it",
    pretty_name="MultiNERD-it",
    source="EuroEval/multinerd-mini-it",
    task=NER,
    languages=[ITALIAN],
    revision="f232cf9f02e16ec87887b808a5c3c98a083484af",
)

SQUAD_IT_CONFIG = DatasetConfig(
    name="squad-it",
    pretty_name="SQuAD-it",
    source="EuroEval/squad-it-mini",
    task=RC,
    languages=[ITALIAN],
    revision="023d3dab819a18c95520720f74a8834e3f036f6b",
)

ILPOST_SUM_CONFIG = DatasetConfig(
    name="ilpost-sum",
    pretty_name="IlPost-Sum",
    source="EuroEval/ilpost-sum",
    task=SUMM,
    languages=[ITALIAN],
    revision="6105e5abd51b57ab7472fb93833eb30944f5892d",
)

MULTI_IFEVAL_IT_CONFIG = DatasetConfig(
    name="multi-ifeval-it",
    pretty_name="MultiIFEval-it",
    source="EuroEval/multi-ifeval-it",
    task=INSTRUCTION_FOLLOWING,
    languages=[ITALIAN],
    train_split=None,
    val_split=None,
    revision="d72c42874fbf906ab7b152941582461ad4a79139",
)

VALEU_IT_CONFIG = DatasetConfig(
    name="valeu-it",
    pretty_name="VaLEU-it",
    source="EuroEval/european-values-it",
    task=EUROPEAN_VALUES,
    languages=[ITALIAN],
    train_split=None,
    val_split=None,
    bootstrap_samples=False,
    instruction_prompt="{text}",
    revision="1a467e5bfb10be77d0b2eecfe71b0167df0daeef",
)

RAGTRUTH_IT_CONFIG = DatasetConfig(
    name="ragtruth-it",
    pretty_name="RAGTruth-it",
    source="EuroEval/ragtruth-translated-hallucinations-it-mini",
    task=HALLU,
    languages=[ITALIAN],
    train_split=None,
    revision="6c950c3c46b32b18ec53734ec7052bf7b2515ece",
)


WINOGRANDE_IT_CONFIG = DatasetConfig(
    name="winogrande-it",
    pretty_name="Winogrande-it",
    source="EuroEval/winogrande-it",
    task=COMMON_SENSE,
    languages=[ITALIAN],
    labels=["a", "b"],
    revision="3229c396f2f154388d62eb03945bd06840175484",
)

INCLUDE_IT_CONFIG = DatasetConfig(
    name="include-it",
    pretty_name="INCLUDE-it",
    source="EuroEval/include-it-mini",
    task=KNOW,
    languages=[ITALIAN],
    revision="5aa09bfe19e83655e1bd85f326fad6c43eec5a4d",
)

MULTILOKO_IT_CONFIG = DatasetConfig(
    name="multiloko-it",
    pretty_name="MultiLoKo-it",
    source="EuroEval/multiloko-it-mini",
    task=KNOW,
    languages=[ITALIAN],
    val_split=None,
    revision="51ab19e48e1c760a5d9046c2ffd4b54c0c525331",
)


# Unofficial datasets ###

MULTI_BFCL_IT_CONFIG = DatasetConfig(
    name="multi-bfcl-it",
    pretty_name="MultiBFCL-it",
    source="EuroEval/multi-bfcl-it-mini",
    task=TOOL_CALLING,
    languages=[ITALIAN],
    train_split=None,
    revision="f23151c9594497169d4487a437d6575f3ca203fc",
)


WMT24PP_IT_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-it-en",
    pretty_name="WMT24++-it-en",
    source="EuroEval/wmt24pp-it-en",
    task=TRANSLATION,
    languages=[ITALIAN],
    source_language=ITALIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="8f7015008fb909e51549cfa7ed7f443ff4ed006f",
)

FLORES_IT_EN_CONFIG = TranslationDatasetConfig(
    name="flores-it-en",
    pretty_name="FLORES-it-en",
    source="EuroEval/flores-it-en",
    task=TRANSLATION,
    languages=[ITALIAN],
    source_language=ITALIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="c9339acec4bf31378b700499d4a5f4ecb98e7aca",
)

MMLU_IT_CONFIG = DatasetConfig(
    name="mmlu-it",
    pretty_name="MMLU-it",
    source="EuroEval/mmlu-it-mini",
    task=KNOW,
    languages=[ITALIAN],
    unofficial=True,
    revision="da36afcacd981fb770c2f1f887d36e1775849a61",
)

HELLASWAG_IT_CONFIG = DatasetConfig(
    name="hellaswag-it",
    pretty_name="HellaSwag-it",
    source="EuroEval/hellaswag-it-mini",
    task=COMMON_SENSE,
    languages=[ITALIAN],
    unofficial=True,
    revision="0867114220680e5ea05ffd7c3b783cb527905217",
)

IFEVAL_IT_CONFIG = DatasetConfig(
    name="ifeval-it",
    pretty_name="IFEval-it",
    source="EuroEval/ifeval-it",
    task=INSTRUCTION_FOLLOWING,
    languages=[ITALIAN],
    train_split=None,
    val_split=None,
    unofficial=True,
    revision="e43975d88b043a3ac5e019d79a5e905c9cf271db",
)

WIKINEURAL_IT_CONFIG = DatasetConfig(
    name="wikineural-it",
    pretty_name="WikiNeural-it",
    source="EuroEval/wikineural-mini-it",
    task=NER,
    languages=[ITALIAN],
    unofficial=True,
    revision="47574fefc7dc92e61681c25a742ef85ae0727ba7",
)

BELEBELE_IT_CONFIG = DatasetConfig(
    name="belebele-it",
    pretty_name="Belebele-it",
    source="EuroEval/belebele-it-mini",
    task=MCRC,
    languages=[ITALIAN],
    unofficial=True,
    revision="e97a40ff8d9555a96ee38dbe1a7b67f845ecd51c",
)

MULTI_WIKI_QA_IT_CONFIG = DatasetConfig(
    name="multi-wiki-qa-it",
    pretty_name="MultiWikiQA-it",
    source="EuroEval/multi-wiki-qa-it-mini",
    task=RC,
    languages=[ITALIAN],
    unofficial=True,
    revision="11836547afc8fcace89b21f1f934dfc1c2e6220d",
)

GOLDENSWAG_IT_CONFIG = DatasetConfig(
    name="goldenswag-it",
    pretty_name="GoldenSwag-it",
    source="EuroEval/goldenswag-it-mini",
    task=COMMON_SENSE,
    languages=[ITALIAN],
    unofficial=True,
    revision="41bfeda3e0bbc79f4e0285bc12e2f25e3498c69b",
)

WIC_ITA_CONFIG = DatasetConfig(
    name="wic-ita",
    pretty_name="WiC-ITA",
    source="EuroEval/wic-ita",
    task=WIC,
    languages=[ITALIAN],
    unofficial=True,
    revision="7cc8e5882a0d4e54a64cb3f2d4a00c750bc67946",
)

EU_MMLU_IT_CONFIG = DatasetConfig(
    name="eu-mmlu-it",
    pretty_name="EU-MMLU-it",
    source="EuroEval/eu-mmlu-it",
    task=KNOW,
    languages=[ITALIAN],
    unofficial=True,
    revision="0f630ae6cd43027caa6e02374c405293820864d1",
)

MKQA_IT_CONFIG = DatasetConfig(
    name="mkqa-it",
    pretty_name="MKQA-it",
    source="EuroEval/mkqa-it-mini",
    task=KNOW,
    languages=[ITALIAN],
    unofficial=True,
    revision="246163cfa7931452da76e776ba9c40571da90a6f",
)
