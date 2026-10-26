"""All Slovak dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, SLOVAK
from ..tasks import (
    COMMON_SENSE,
    HALLU,
    INSTRUCTION_FOLLOWING,
    KNOW,
    LA,
    NER,
    NLI,
    RC,
    SENT,
    TOOL_CALLING,
    TRANSLATION,
)

# Official datasets ###

WMT24PP_EN_SK_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-sk",
    pretty_name="WMT24++-en-sk",
    source="EuroEval/wmt24pp-en-sk",
    task=TRANSLATION,
    languages=[SLOVAK],
    source_language=ENGLISH,
    target_language=SLOVAK,
    revision="300e4694af0c39a3be440aa731050a3bbab448ac",
)

FLORES_EN_SK_CONFIG = TranslationDatasetConfig(
    name="flores-en-sk",
    pretty_name="FLORES-en-sk",
    source="EuroEval/flores-en-sk",
    task=TRANSLATION,
    languages=[SLOVAK],
    source_language=ENGLISH,
    target_language=SLOVAK,
    revision="aad19ac4157d989c31b7968bd36332f8bc1cfb15",
)

CSFD_SENTIMENT_SK_CONFIG = DatasetConfig(
    name="csfd-sentiment-sk",
    pretty_name="CSFD Sentiment SK",
    source="EuroEval/csfd-sentiment-sk-mini",
    task=SENT,
    languages=[SLOVAK],
    revision="3f47e7cd06d2b3bebfb251f9576bdf2b00e775d7",
)

SCALA_SK_CONFIG = DatasetConfig(
    name="scala-sk",
    pretty_name="ScaLA-sk",
    source="EuroEval/scala-sk",
    task=LA,
    languages=[SLOVAK],
    revision="1a611c8bd3e546a0b8796a8400e76b6e3abbb84f",
)

UNER_SK_CONFIG = DatasetConfig(
    name="uner-sk",
    pretty_name="UNER-sk",
    source="EuroEval/uner-sk-mini",
    task=NER,
    languages=[SLOVAK],
    revision="032ad01bf71425656cd217a45dfe03ef3c4a80e5",
)

MULTI_WIKI_QA_SK_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sk",
    pretty_name="MultiWikiQA-sk",
    source="EuroEval/multi-wiki-qa-sk-mini",
    task=RC,
    languages=[SLOVAK],
    revision="2b7372da634324434e08ee0cb407631aa733023c",
)

MMLU_SK_CONFIG = DatasetConfig(
    name="mmlu-sk",
    pretty_name="MMLU-sk",
    source="EuroEval/mmlu-sk-mini",
    task=KNOW,
    languages=[SLOVAK],
    revision="cb4a6548de11fc24f4486fd53faf6e44987d0cd3",
)

WINOGRANDE_SK_CONFIG = DatasetConfig(
    name="winogrande-sk",
    pretty_name="Winogrande-sk",
    source="EuroEval/winogrande-sk",
    task=COMMON_SENSE,
    languages=[SLOVAK],
    revision="33f8f2deca38fe1dffc7c8ec257712842f65d1b2",
)

MULTI_IFEVAL_SK_CONFIG = DatasetConfig(
    name="multi-ifeval-sk",
    pretty_name="MultiIFEval-sk",
    source="EuroEval/multi-ifeval-sk",
    task=INSTRUCTION_FOLLOWING,
    languages=[SLOVAK],
    train_split=None,
    val_split=None,
    revision="6d3be34c70769885072c5111219c73ddbe81956b",
)

RAGTRUTH_SK_CONFIG = DatasetConfig(
    name="ragtruth-sk",
    pretty_name="RAGTruth-sk",
    source="EuroEval/ragtruth-translated-hallucinations-sk-mini",
    task=HALLU,
    languages=[SLOVAK],
    train_split=None,
    revision="30c94ee478887884a61564b702ca3d57b0d362c5",
)


# Unofficial datasets ###

MULTI_BFCL_SK_CONFIG = DatasetConfig(
    name="multi-bfcl-sk",
    pretty_name="MultiBFCL-sk",
    source="EuroEval/multi-bfcl-sk-mini",
    task=TOOL_CALLING,
    languages=[SLOVAK],
    train_split=None,
    revision="b0d6c6b580d524e0a4d34027f9ff701ab2c55e11",
)


WMT24PP_SK_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-sk-en",
    pretty_name="WMT24++-sk-en",
    source="EuroEval/wmt24pp-sk-en",
    task=TRANSLATION,
    languages=[SLOVAK],
    source_language=SLOVAK,
    target_language=ENGLISH,
    unofficial=True,
    revision="4116a07e8979afa0323a8e03babdb70257d295e0",
)

FLORES_SK_EN_CONFIG = TranslationDatasetConfig(
    name="flores-sk-en",
    pretty_name="FLORES-sk-en",
    source="EuroEval/flores-sk-en",
    task=TRANSLATION,
    languages=[SLOVAK],
    source_language=SLOVAK,
    target_language=ENGLISH,
    unofficial=True,
    revision="8a40edf083f7f8c73abc7e946bcd0c2d810e8960",
)

SKLEP_NLI_CONFIG = DatasetConfig(
    name="sklep-nli",
    pretty_name="SKLEP NLI",
    source="EuroEval/sklep-nli-mini",
    task=NLI,
    languages=[SLOVAK],
    unofficial=True,
    revision="f1591170f3377d5f2b5da8b29c597185c2102cbf",
)

SK_QUAD_CONFIG = DatasetConfig(
    name="sk-quad",
    pretty_name="SK-QuAD",
    source="EuroEval/sk-quad-mini",
    task=RC,
    languages=[SLOVAK],
    unofficial=True,
    revision="f3f0d96abab477f7f45285dd525e29f70c8a3e9c",
)

WIKIGOLDSK_CONFIG = DatasetConfig(
    name="wikigold-sk",
    pretty_name="WikiGoldSK",
    source="EuroEval/wikigold-sk-mini",
    task=NER,
    languages=[SLOVAK],
    unofficial=True,
    revision="e80614fc5f317037a2e41f0d03658c3083bb9b00",
)

SKLEP_RTE_CONFIG = DatasetConfig(
    name="sklep-rte",
    pretty_name="SKLEP RTE",
    source="EuroEval/sklep-rte-mini",
    task=NLI,
    languages=[SLOVAK],
    labels=["entailment", "not entailment"],
    prompt_label_mapping={"entailment": "pravda", "not entailment": "nepravda"},
    prompt_prefix=(
        "Nasledujú páry tvrdení a ich logická súvislosť, ktorá môže byť {labels_str}."
    ),
    prompt_template="{text}\nImplikácia: {label}",
    instruction_prompt=(
        "{text}\n\nUrčite, či druhé tvrdenie vyplýva z prvého. Odpovedzte so "
        "{labels_str}, a nič iné."
    ),
    unofficial=True,
    revision="f45a6c20b4353354dc00ea9d5f0ce887fe936856",
)

REVIEWS3_CONFIG = DatasetConfig(
    name="reviews3",
    pretty_name="Reviews3",
    source="EuroEval/reviews3-mini",
    task=SENT,
    languages=[SLOVAK],
    labels=["negative", "positive"],
    unofficial=True,
    revision="660cbd55adf936a793677651e696bc4b3e742843",
)

EU_MMLU_SK_CONFIG = DatasetConfig(
    name="eu-mmlu-sk",
    pretty_name="EU-MMLU-sk",
    source="EuroEval/eu-mmlu-sk",
    task=KNOW,
    languages=[SLOVAK],
    unofficial=True,
    revision="9ec292c7a99b25b9b8c8edd41bf871b1892b7469",
)
