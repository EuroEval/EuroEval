"""All Albanian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ALBANIAN, ENGLISH
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

FLORES_EN_SQ_CONFIG = TranslationDatasetConfig(
    name="flores-en-sq",
    pretty_name="FLORES-en-sq",
    source="EuroEval/flores-en-sq",
    task=TRANSLATION,
    languages=[ALBANIAN],
    source_language=ENGLISH,
    target_language=ALBANIAN,
    revision="2fa66910650fcfbc10e382a563728c870b142958",
)

MMS_SQ_CONFIG = DatasetConfig(
    name="mms-sq",
    pretty_name="MMS-sq",
    source="EuroEval/mms-sq-mini",
    task=SENT,
    languages=[ALBANIAN],
    revision="41f66824b0877b8205387e09239a66ded96a60a7",
)

SCALA_SQ_CONFIG = DatasetConfig(
    name="scala-sq",
    pretty_name="ScaLA-sq",
    source="EuroEval/scala-sq",
    task=LA,
    languages=[ALBANIAN],
    revision="762bfc05c2cbe3cc5242a9e317f353e7f40f1408",
)

WIKIANN_SQ_CONFIG = DatasetConfig(
    name="wikiann-sq",
    pretty_name="WikiANN-sq",
    source="EuroEval/wikiann-sq-mini",
    task=NER,
    languages=[ALBANIAN],
    revision="fc37d1aaef58e649b2aeb31319a47cb40d0f3443",
)

MULTI_WIKI_QA_SQ_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sq",
    pretty_name="MultiWikiQA-sq",
    source="EuroEval/multi-wiki-qa-sq-mini",
    task=RC,
    languages=[ALBANIAN],
    revision="2335fe982a8723b3f14df1653b1dbab18c6b071a",
)

LR_SUM_SQ_CONFIG = DatasetConfig(
    name="lr-sum-sq",
    pretty_name="LRSum-sq",
    source="EuroEval/lr-sum-sq-mini",
    task=SUMM,
    languages=[ALBANIAN],
    revision="5c54fbbac1b10191ad72c6b7b6399facdc07f880",
)

WINOGRANDE_SQ_CONFIG = DatasetConfig(
    name="winogrande-sq",
    pretty_name="Winogrande-sq",
    source="EuroEval/winogrande-sq",
    task=COMMON_SENSE,
    languages=[ALBANIAN],
    labels=["a", "b"],
    revision="daad5e8df31ce4ef44ec11a2eef761258e45e9c1",
)

MULTI_IFEVAL_SQ_CONFIG = DatasetConfig(
    name="multi-ifeval-sq",
    pretty_name="MultiIFEval-sq",
    source="EuroEval/multi-ifeval-sq",
    task=INSTRUCTION_FOLLOWING,
    languages=[ALBANIAN],
    train_split=None,
    val_split=None,
    revision="4d5663e88a5f1fa0490978a0a013625ce5065fd7",
)

INCLUDE_SQ_CONFIG = DatasetConfig(
    name="include-sq",
    pretty_name="INCLUDE-sq",
    source="EuroEval/include-sq-mini",
    task=KNOW,
    languages=[ALBANIAN],
    revision="d62d8d46ce5e42063401deb3c7f729866873d760",
)

RAGTRUTH_SQ_CONFIG = DatasetConfig(
    name="ragtruth-sq",
    pretty_name="RAGTruth-sq",
    source="EuroEval/ragtruth-translated-hallucinations-sq-mini",
    task=HALLU,
    languages=[ALBANIAN],
    train_split=None,
    revision="2390e64c9935d6d48b8471007b826a470c28ca87",
)


# Unofficial datasets ###

MULTI_BFCL_SQ_CONFIG = DatasetConfig(
    name="multi-bfcl-sq",
    pretty_name="MultiBFCL-sq",
    source="EuroEval/multi-bfcl-sq-mini",
    task=TOOL_CALLING,
    languages=[ALBANIAN],
    train_split=None,
    revision="83c1a764b4260940a20a32de311e89ab6d2af821",
)


FLORES_SQ_EN_CONFIG = TranslationDatasetConfig(
    name="flores-sq-en",
    pretty_name="FLORES-sq-en",
    source="EuroEval/flores-sq-en",
    task=TRANSLATION,
    languages=[ALBANIAN],
    source_language=ALBANIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="3ce698b9139e2f3b928583881f0d81430197435c",
)

GLOBAL_MMLU_LITE_SQ_CONFIG = DatasetConfig(
    name="global-mmlu-lite-sq",
    pretty_name="GlobalMMLULite-sq",
    source="EuroEval/global-mmlu-lite-sq",
    task=KNOW,
    languages=[ALBANIAN],
    unofficial=True,
    revision="b126c126cf152326233038aeb48ebf4adf5f52b2",
)
