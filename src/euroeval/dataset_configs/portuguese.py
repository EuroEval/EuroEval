"""All Portuguese dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, EUROPEAN_PORTUGUESE, PORTUGUESE
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
)

# Official datasets ###

WMT24PP_EN_PT_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-pt",
    pretty_name="WMT24++-en-pt",
    source="EuroEval/wmt24pp-en-pt",
    task=TRANSLATION,
    languages=[PORTUGUESE],
    source_language=ENGLISH,
    target_language=PORTUGUESE,
    revision="681523ae612fafbc41e4af295eddbdb1f1d000d1",
)

FLORES_EN_PT_CONFIG = TranslationDatasetConfig(
    name="flores-en-pt",
    pretty_name="FLORES-en-pt",
    source="EuroEval/flores-en-pt",
    task=TRANSLATION,
    languages=[PORTUGUESE],
    source_language=ENGLISH,
    target_language=PORTUGUESE,
    revision="081d29c86205bafeb7886434d98d8a282ae7b13a",
)

SST2_PT_CONFIG = DatasetConfig(
    name="sst2-pt",
    pretty_name="SST2-pt",
    source="EuroEval/sst2-pt-mini",
    task=SENT,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    labels=["positive", "negative"],
    revision="4e0622e0c7f5b24c89f9a6427fff0846b1848b1d",
)

SCALA_PT = DatasetConfig(
    name="scala-pt",
    pretty_name="ScaLA-pt",
    source="EuroEval/scala-pt",
    task=LA,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    revision="d209d8731e5a6f4813aef845fd4147e02c1f2a36",
)

HAREM_CONFIG = DatasetConfig(
    name="harem",
    pretty_name="HAREM",
    source="EuroEval/harem",
    task=NER,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    revision="58a9952359e29063c402721f9975729f9732fb7a",
)

MULTI_WIKI_QA_PT_CONFIG = DatasetConfig(
    name="multi-wiki-qa-pt",
    pretty_name="MultiWikiQA-pt",
    source="EuroEval/multi-wiki-qa-pt-pt-mini",
    task=RC,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    revision="044c8d89be2df6ed864acd6d785de43696813da3",
)

PUBLICO_CONFIG = DatasetConfig(
    name="publico",
    pretty_name="Publico",
    source="EuroEval/publico-mini",
    task=SUMM,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    revision="3153731935b25cbcb86ee5539a12b94ebfcd22fa",
)

MULTI_IFEVAL_PT_CONFIG = DatasetConfig(
    name="multi-ifeval-pt",
    pretty_name="MultiIFEval-pt",
    source="EuroEval/multi-ifeval-pt",
    task=INSTRUCTION_FOLLOWING,
    languages=[PORTUGUESE],
    train_split=None,
    val_split=None,
    revision="637d00a364ff620c1cfc36a21eb32da0c561a9ca",
)

VALEU_PT_CONFIG = DatasetConfig(
    name="valeu-pt",
    pretty_name="VaLEU-pt",
    source="EuroEval/european-values-pt",
    task=EUROPEAN_VALUES,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    train_split=None,
    val_split=None,
    bootstrap_samples=False,
    instruction_prompt="{text}",
    revision="e68718415c0ed18b2d10725892be29a890197b93",
)


ALBA_MCQ_PT_CONFIG = DatasetConfig(
    name="alba-mcq-pt",
    pretty_name="ALBA-MCQ",
    source="EuroEval/euroeval-amalia-alba-mcq-pt",
    task=KNOW,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    labels=["a", "b", "c"],
    val_split=None,
    revision="8ec38ec06ea84415218f681613f229beea93bd7f",
)

CULTURA_VIVA_PT_CONFIG = DatasetConfig(
    name="cultura-viva-pt",
    pretty_name="CulturaVivaPT",
    source="EuroEval/euroeval-amalia-cultura-viva-pt",
    task=KNOW,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    revision="96a9d48320fa7ff2f3e43dcca8f196fdc2c8464e",
)

WINOGRANDE_PT_CONFIG = DatasetConfig(
    name="winogrande-pt",
    pretty_name="Winogrande-pt",
    source="EuroEval/winogrande-pt",
    task=COMMON_SENSE,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    labels=["a", "b"],
    revision="36702672886ef6386684b9faa2114cbfa3cc582c",
)

RAGTRUTH_PT_CONFIG = DatasetConfig(
    name="ragtruth-pt",
    pretty_name="RAGTruth-pt",
    source="EuroEval/ragtruth-translated-hallucinations-pt-mini",
    task=HALLU,
    languages=[PORTUGUESE],
    train_split=None,
    revision="5161438fa7b07491bfabbbba70f37abd1a6b99d1",
)


# Unofficial datasets ###

MULTI_BFCL_PT_CONFIG = DatasetConfig(
    name="multi-bfcl-pt",
    pretty_name="MultiBFCL-pt",
    source="EuroEval/multi-bfcl-pt-mini",
    task=TOOL_CALLING,
    languages=[PORTUGUESE],
    train_split=None,
    revision="6b490519433143df9c0474d2814e5c0a6cd2bf54",
)


WMT24PP_PT_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-pt-en",
    pretty_name="WMT24++-pt-en",
    source="EuroEval/wmt24pp-pt-en",
    task=TRANSLATION,
    languages=[PORTUGUESE],
    source_language=PORTUGUESE,
    target_language=ENGLISH,
    unofficial=True,
    revision="0d638c498c707567fd20b042c1ffd5c5dd35a522",
)

FLORES_PT_EN_CONFIG = TranslationDatasetConfig(
    name="flores-pt-en",
    pretty_name="FLORES-pt-en",
    source="EuroEval/flores-pt-en",
    task=TRANSLATION,
    languages=[PORTUGUESE],
    source_language=PORTUGUESE,
    target_language=ENGLISH,
    unofficial=True,
    revision="f444011318958e23ba61e9f631833d15da39a8f2",
)

GOLDENSWAG_PT_CONFIG = DatasetConfig(
    name="goldenswag-pt",
    pretty_name="GoldenSwag-pt",
    source="EuroEval/goldenswag-pt-mini",
    task=COMMON_SENSE,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    unofficial=True,
    revision="a03a16bd2d668adbb3897e5ba18e643468886b50",
)

MMLU_PT_CONFIG = DatasetConfig(
    name="mmlu-pt",
    pretty_name="MMLU-pt",
    source="EuroEval/mmlu-pt-mini",
    task=KNOW,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    unofficial=True,
    revision="abb2ebe17ca6ba27cc24c9c534ef825f2d7dc109",
)

IFEVAL_PT_CONFIG = DatasetConfig(
    name="ifeval-pt",
    pretty_name="IFEval-pt",
    source="EuroEval/ifeval-pt",
    task=INSTRUCTION_FOLLOWING,
    languages=[PORTUGUESE],
    train_split=None,
    val_split=None,
    unofficial=True,
    revision="2ccb4baef1164dc15f5d922c5410ceb7a7bb659a",
)

BOOLQ_PT_CONFIG = DatasetConfig(
    name="boolq-pt",
    pretty_name="BoolQ-pt",
    source="EuroEval/boolq-pt",
    task=MCRC,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    unofficial=True,
    revision="933e574026bc87f38db9cc4bb34f80a154b26377",
)

INCLUDE_PT_CONFIG = DatasetConfig(
    name="include-pt",
    pretty_name="INCLUDE-pt",
    source="EuroEval/include-pt-mini",
    task=KNOW,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    unofficial=True,
    revision="4ac217aef89f1ee8313346893aafd41fd951a03e",
)

MULTILOKO_PT_CONFIG = DatasetConfig(
    name="multiloko-pt",
    pretty_name="MultiLoKo-pt",
    source="EuroEval/multiloko-pt-mini",
    task=KNOW,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    val_split=None,
    unofficial=True,
    revision="ffd3a9d4bae9dec6d98544429bb3ff138f512c22",
)

PT_EXAMS_CONFIG = DatasetConfig(
    name="pt-exams",
    pretty_name="PT Exams",
    source="EuroEval/euroeval-amalia-pt-exams",
    task=KNOW,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    unofficial=True,
    revision="eb05649ad04c336dcdc33e079b5f7467a497e57d",
)

SAUDADE_PT_CONFIG = DatasetConfig(
    name="saudade-pt",
    pretty_name="SAUDADE-PT",
    source="EuroEval/euroeval-amalia-saudade-pt-mini",
    task=KNOW,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    labels=["a", "b"],
    unofficial=True,
    revision="3c83d0bb8bc13103af75283a944e45a47aace7f8",
)

PT_COMPLETIONS_CONFIG = DatasetConfig(
    name="pt-completions",
    pretty_name="PT-PT Completions",
    source="EuroEval/euroeval-amalia-pt-completions",
    task=KNOW,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    labels=["a", "b"],
    prompt_template="Pergunta: {text}\nVariante em português europeu: {label}",
    instruction_prompt=(
        "{text}\nPergunta: Qual opção está em português europeu?\n\n"
        "Responde à pergunta acima usando só {labels_str}, e nada mais."
    ),
    val_split=None,
    unofficial=True,
    revision="52eb5955299c441944780b755242d4e6a20cf80b",
)

EU_MMLU_PT_CONFIG = DatasetConfig(
    name="eu-mmlu-pt",
    pretty_name="EU-MMLU-pt",
    source="EuroEval/eu-mmlu-pt",
    task=KNOW,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    unofficial=True,
    revision="f09e7b8d7f907be600d9649705a9069d3ab9918c",
)

MKQA_PT_CONFIG = DatasetConfig(
    name="mkqa-pt",
    pretty_name="MKQA-pt",
    source="EuroEval/mkqa-pt-mini",
    task=KNOW,
    languages=[PORTUGUESE],
    unofficial=True,
    revision="4e3b24977b8b719e8c72529f443b2a4ea3d23d94",
)
