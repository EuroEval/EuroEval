"""All Czech dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import CZECH, ENGLISH
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

WMT24PP_EN_CS_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-cs",
    pretty_name="WMT24++-en-cs",
    source="EuroEval/wmt24pp-en-cs",
    task=TRANSLATION,
    languages=[CZECH],
    source_language=ENGLISH,
    target_language=CZECH,
    revision="8f83d26dbc5b2d7a5b84d01e0318cf77be49bd48",
)

FLORES_EN_CS_CONFIG = TranslationDatasetConfig(
    name="flores-en-cs",
    pretty_name="FLORES-en-cs",
    source="EuroEval/flores-en-cs",
    task=TRANSLATION,
    languages=[CZECH],
    source_language=ENGLISH,
    target_language=CZECH,
    revision="0135bb05ff029b1d868bf456239137daec7d7e85",
)

CSFD_SENTIMENT_CONFIG = DatasetConfig(
    name="csfd-sentiment",
    pretty_name="CSFD Sentiment",
    source="EuroEval/csfd-sentiment-mini",
    task=SENT,
    languages=[CZECH],
    revision="e37a88400c5417f7865a67a6c33037540b72bafb",
)

CS_GEC_CONFIG = DatasetConfig(
    name="cs-gec",
    pretty_name="CS-GEC",
    source="EuroEval/cs-gec-mini",
    task=LA,
    languages=[CZECH],
    revision="86b350bb54ece5f032eb25d183add1db4b5bedab",
)

PONER_CONFIG = DatasetConfig(
    name="poner",
    pretty_name="PoNER",
    source="EuroEval/poner-mini",
    task=NER,
    languages=[CZECH],
    revision="7aab741baed88bd99f50e3899b5dd88488c478e1",
)

SQAD_CONFIG = DatasetConfig(
    name="sqad",
    pretty_name="SQAD",
    source="EuroEval/sqad-mini",
    task=RC,
    languages=[CZECH],
    revision="442434c36eb4dfdd4f1ee2bdb2b2f6d834f13682",
)

CZECH_NEWS_CONFIG = DatasetConfig(
    name="czech-news",
    pretty_name="Czech News",
    source="EuroEval/czech-news-mini",
    task=SUMM,
    languages=[CZECH],
    revision="5cd874e7f187a27c251ea78a934a41460a20523c",
)

UMIMETO_QA_CONFIG = DatasetConfig(
    name="umimeto-qa",
    pretty_name="Umimeto QA",
    source="EuroEval/umimeto-qa",
    task=KNOW,
    languages=[CZECH],
    revision="8e96f264882e30039519f7e6b3d79ff338634a8d",
)

HELLASWAG_CS_CONFIG = DatasetConfig(
    name="hellaswag-cs",
    pretty_name="HellaSwag-cs",
    source="EuroEval/hellaswag-cs-mini",
    task=COMMON_SENSE,
    languages=[CZECH],
    revision="d9c750b7bacccb68abe55a074c1d67d50a692ea0",
)

MULTI_IFEVAL_CS_CONFIG = DatasetConfig(
    name="multi-ifeval-cs",
    pretty_name="MultiIFEval-cs",
    source="EuroEval/multi-ifeval-cs",
    task=INSTRUCTION_FOLLOWING,
    languages=[CZECH],
    train_split=None,
    val_split=None,
    revision="8efd487c28ad640f7ff92fd63d1c41ec0f39b8cd",
)

RAGTRUTH_CS_CONFIG = DatasetConfig(
    name="ragtruth-cs",
    pretty_name="RAGTruth-cs",
    source="EuroEval/ragtruth-translated-hallucinations-cs-mini",
    task=HALLU,
    languages=[CZECH],
    train_split=None,
    revision="e88d1413fc280edbbc7092ac31e03cf61d32f88f",
)


# Unofficial datasets ###

MULTI_BFCL_CS_CONFIG = DatasetConfig(
    name="multi-bfcl-cs",
    pretty_name="MultiBFCL-cs",
    source="EuroEval/multi-bfcl-cs-mini",
    task=TOOL_CALLING,
    languages=[CZECH],
    train_split=None,
    revision="186d36302aa082fb7d57c7428c410672aeea0501",
)


WMT24PP_CS_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-cs-en",
    pretty_name="WMT24++-cs-en",
    source="EuroEval/wmt24pp-cs-en",
    task=TRANSLATION,
    languages=[CZECH],
    source_language=CZECH,
    target_language=ENGLISH,
    unofficial=True,
    revision="d3f258b81237fc1759862dbc65547897243895f2",
)

FLORES_CS_EN_CONFIG = TranslationDatasetConfig(
    name="flores-cs-en",
    pretty_name="FLORES-cs-en",
    source="EuroEval/flores-cs-en",
    task=TRANSLATION,
    languages=[CZECH],
    source_language=CZECH,
    target_language=ENGLISH,
    unofficial=True,
    revision="30569e5956984aabe63bc1419d40b815d2d0b83b",
)

SCALA_CS_CONFIG = DatasetConfig(
    name="scala-cs",
    pretty_name="ScaLA-cs",
    source="EuroEval/scala-cs",
    task=LA,
    languages=[CZECH],
    unofficial=True,
    revision="a4ed703ebf1e5ac9d1bcbc9cafc725ebd9992c3a",
)

EU_MMLU_CS_CONFIG = DatasetConfig(
    name="eu-mmlu-cs",
    pretty_name="EU-MMLU-cs",
    source="EuroEval/eu-mmlu-cs",
    task=KNOW,
    languages=[CZECH],
    unofficial=True,
    revision="9dd660a7712d9e41163e3fa36b5ac9276f46daa5",
)
