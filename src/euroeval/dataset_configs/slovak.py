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
)

FLORES_EN_SK_CONFIG = TranslationDatasetConfig(
    name="flores-en-sk",
    pretty_name="FLORES-en-sk",
    source="EuroEval/flores-en-sk",
    task=TRANSLATION,
    languages=[SLOVAK],
    source_language=ENGLISH,
    target_language=SLOVAK,
)

CSFD_SENTIMENT_SK_CONFIG = DatasetConfig(
    name="csfd-sentiment-sk",
    pretty_name="CSFD Sentiment SK",
    source="EuroEval/csfd-sentiment-sk-mini",
    task=SENT,
    languages=[SLOVAK],
)

SCALA_SK_CONFIG = DatasetConfig(
    name="scala-sk",
    pretty_name="ScaLA-sk",
    source="EuroEval/scala-sk",
    task=LA,
    languages=[SLOVAK],
)

UNER_SK_CONFIG = DatasetConfig(
    name="uner-sk",
    pretty_name="UNER-sk",
    source="EuroEval/uner-sk-mini",
    task=NER,
    languages=[SLOVAK],
)

MULTI_WIKI_QA_SK_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sk",
    pretty_name="MultiWikiQA-sk",
    source="EuroEval/multi-wiki-qa-sk-mini",
    task=RC,
    languages=[SLOVAK],
)

MMLU_SK_CONFIG = DatasetConfig(
    name="mmlu-sk",
    pretty_name="MMLU-sk",
    source="EuroEval/mmlu-sk-mini",
    task=KNOW,
    languages=[SLOVAK],
)

WINOGRANDE_SK_CONFIG = DatasetConfig(
    name="winogrande-sk",
    pretty_name="Winogrande-sk",
    source="EuroEval/winogrande-sk",
    task=COMMON_SENSE,
    languages=[SLOVAK],
)

MULTI_IFEVAL_SK_CONFIG = DatasetConfig(
    name="multi-ifeval-sk",
    pretty_name="MultiIFEval-sk",
    source="EuroEval/multi-ifeval-sk",
    task=INSTRUCTION_FOLLOWING,
    languages=[SLOVAK],
    train_split=None,
    val_split=None,
)

RAGTRUTH_SK_CONFIG = DatasetConfig(
    name="ragtruth-sk",
    pretty_name="RAGTruth-sk",
    source="EuroEval/ragtruth-translated-hallucinations-sk-mini",
    task=HALLU,
    languages=[SLOVAK],
    train_split=None,
)


# Unofficial datasets ###

WMT24PP_SK_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-sk-en",
    pretty_name="WMT24++-sk-en",
    source="EuroEval/wmt24pp-sk-en",
    task=TRANSLATION,
    languages=[SLOVAK],
    source_language=SLOVAK,
    target_language=ENGLISH,
    unofficial=True,
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
)

SKLEP_NLI_CONFIG = DatasetConfig(
    name="sklep-nli",
    pretty_name="SKLEP NLI",
    source="EuroEval/sklep-nli-mini",
    task=NLI,
    languages=[SLOVAK],
    unofficial=True,
)

SK_QUAD_CONFIG = DatasetConfig(
    name="sk-quad",
    pretty_name="SK-QuAD",
    source="EuroEval/sk-quad-mini",
    task=RC,
    languages=[SLOVAK],
    unofficial=True,
)

WIKIGOLDSK_CONFIG = DatasetConfig(
    name="wikigold-sk",
    pretty_name="WikiGoldSK",
    source="EuroEval/wikigold-sk-mini",
    task=NER,
    languages=[SLOVAK],
    unofficial=True,
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
)

REVIEWS3_CONFIG = DatasetConfig(
    name="reviews3",
    pretty_name="Reviews3",
    source="EuroEval/reviews3-mini",
    task=SENT,
    languages=[SLOVAK],
    labels=["negative", "positive"],
    unofficial=True,
)

EU_MMLU_SK_CONFIG = DatasetConfig(
    name="eu-mmlu-sk",
    pretty_name="EU-MMLU-sk",
    source="EuroEval/eu-mmlu-sk",
    task=KNOW,
    languages=[SLOVAK],
    unofficial=True,
)
