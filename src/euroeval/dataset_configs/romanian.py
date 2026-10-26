"""All Romanian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, ROMANIAN
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

WMT24PP_EN_RO_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-ro",
    pretty_name="WMT24++-en-ro",
    source="EuroEval/wmt24pp-en-ro",
    task=TRANSLATION,
    languages=[ROMANIAN],
    source_language=ENGLISH,
    target_language=ROMANIAN,
    revision="93db170c9cb5d31242a5f06ac3231279160c9c50",
)

FLORES_EN_RO_CONFIG = TranslationDatasetConfig(
    name="flores-en-ro",
    pretty_name="FLORES-en-ro",
    source="EuroEval/flores-en-ro",
    task=TRANSLATION,
    languages=[ROMANIAN],
    source_language=ENGLISH,
    target_language=ROMANIAN,
    revision="08159cbd288936d8833f30d7825db869050b8246",
)

ROSENT_CONFIG = DatasetConfig(
    name="ro-sent",
    pretty_name="RoSent",
    source="EuroEval/ro-sent-mini",
    task=SENT,
    languages=[ROMANIAN],
    labels=["positive", "negative"],
    revision="7ef8263f940924aacce92c91cc02f9ca32f033ff",
)

SCALA_RO_CONFIG = DatasetConfig(
    name="scala-ro",
    pretty_name="ScaLA-ro",
    source="EuroEval/scala-ro",
    task=LA,
    languages=[ROMANIAN],
    revision="5e962844f2cb50d6cf87aca03c7f4d7d9cb2f2a1",
)

RONEC_CONFIG = DatasetConfig(
    name="ronec",
    pretty_name="RoNEC",
    source="EuroEval/ronec-mini",
    task=NER,
    languages=[ROMANIAN],
    revision="d9faaffc51be55d3927b2bf3382d3afca5dd3736",
)

MULTI_WIKI_QA_RO_CONFIG = DatasetConfig(
    name="multi-wiki-qa-ro",
    pretty_name="MultiWikiQA-ro",
    source="EuroEval/multi-wiki-qa-ro-mini",
    task=RC,
    languages=[ROMANIAN],
    revision="ceb45e50cbd1a3c31e104c1c9112c1257112b959",
)

SUMO_RO_CONFIG = DatasetConfig(
    name="sumo-ro",
    pretty_name="SumO-Ro",
    source="EuroEval/sumo-ro-mini",
    task=SUMM,
    languages=[ROMANIAN],
    revision="708e3504bb3d843edd37b1c25aed93c70a06a75a",
)

GLOBAL_MMLU_RO_CONFIG = DatasetConfig(
    name="global-mmlu-ro",
    pretty_name="GlobalMMLU-ro",
    source="EuroEval/global-mmlu-ro-mini",
    task=KNOW,
    languages=[ROMANIAN],
    revision="c30a6f35956dce2266ed15a194497423fb332631",
)

WINOGRANDE_RO_CONFIG = DatasetConfig(
    name="winogrande-ro",
    pretty_name="Winogrande-ro",
    source="EuroEval/winogrande-ro",
    task=COMMON_SENSE,
    languages=[ROMANIAN],
    labels=["a", "b"],
    revision="f55b2a51cf82b2d174fad8b6466a0381f1cd5db5",
)

MULTI_IFEVAL_RO_CONFIG = DatasetConfig(
    name="multi-ifeval-ro",
    pretty_name="MultiIFEval-ro",
    source="EuroEval/multi-ifeval-ro",
    task=INSTRUCTION_FOLLOWING,
    languages=[ROMANIAN],
    train_split=None,
    val_split=None,
    revision="3b93df866de275af7336af2160a95ac584d7d3d2",
)

RAGTRUTH_RO_CONFIG = DatasetConfig(
    name="ragtruth-ro",
    pretty_name="RAGTruth-ro",
    source="EuroEval/ragtruth-translated-hallucinations-ro-mini",
    task=HALLU,
    languages=[ROMANIAN],
    train_split=None,
    revision="98815292df69e533bb0fe077ff898de135b7ac86",
)


# Unofficial datasets ###

MULTI_BFCL_RO_CONFIG = DatasetConfig(
    name="multi-bfcl-ro",
    pretty_name="MultiBFCL-ro",
    source="EuroEval/multi-bfcl-ro-mini",
    task=TOOL_CALLING,
    languages=[ROMANIAN],
    train_split=None,
    revision="4deb884cf4a86277b1c593ba049edfa9824288a4",
)


WMT24PP_RO_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-ro-en",
    pretty_name="WMT24++-ro-en",
    source="EuroEval/wmt24pp-ro-en",
    task=TRANSLATION,
    languages=[ROMANIAN],
    source_language=ROMANIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="37d4af13f1ef7ffd253ffb5b19c9a23deeacca79",
)

FLORES_RO_EN_CONFIG = TranslationDatasetConfig(
    name="flores-ro-en",
    pretty_name="FLORES-ro-en",
    source="EuroEval/flores-ro-en",
    task=TRANSLATION,
    languages=[ROMANIAN],
    source_language=ROMANIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="fb5125359310231a91e7f0804b711233f38baf3f",
)

EU_MMLU_RO_CONFIG = DatasetConfig(
    name="eu-mmlu-ro",
    pretty_name="EU-MMLU-ro",
    source="EuroEval/eu-mmlu-ro",
    task=KNOW,
    languages=[ROMANIAN],
    unofficial=True,
    revision="91c0435a6ecc1c2401d678c8a34e2e94d5b0f6c2",
)
