"""All Belarusian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import BELARUSIAN, ENGLISH
from ..tasks import (
    COMMON_SENSE,
    HALLU,
    INSTRUCTION_FOLLOWING,
    LA,
    NER,
    NLI,
    RC,
    SENT,
    TOOL_CALLING,
    TRANSLATION,
    WIC,
)

# Official datasets ###

FLORES_EN_BE_CONFIG = TranslationDatasetConfig(
    name="flores-en-be",
    pretty_name="FLORES-en-be",
    source="EuroEval/flores-en-be",
    task=TRANSLATION,
    languages=[BELARUSIAN],
    source_language=ENGLISH,
    target_language=BELARUSIAN,
    revision="255c163e9773d34bb2a57fbcb85aaab09687bcb6",
)

BESLS_CONFIG = DatasetConfig(
    name="besls",
    pretty_name="BeSLS",
    source="EuroEval/besls",
    task=SENT,
    languages=[BELARUSIAN],
    revision="8f0535d8cb3cbc0c558d8a5d61a876b120ccee05",
)

WIKIANN_BE_CONFIG = DatasetConfig(
    name="wikiann-be",
    pretty_name="WikiANN-be",
    source="EuroEval/wikiann-be-mini",
    task=NER,
    languages=[BELARUSIAN],
    revision="a2309460910cfed13ff99af6ec3addc0e459ff87",
)

MULTI_WIKI_QA_BE_CONFIG = DatasetConfig(
    name="multi-wiki-qa-be",
    pretty_name="MultiWikiQA-be",
    source="EuroEval/multi-wiki-qa-be-mini",
    task=RC,
    languages=[BELARUSIAN],
    revision="c90f35f5d68811f9061209dca7e5355310058f14",
)

BE_WSC_CONFIG = DatasetConfig(
    name="be-wsc",
    pretty_name="BE-WSC",
    source="EuroEval/be-wsc",
    task=COMMON_SENSE,
    languages=[BELARUSIAN],
    labels=["a", "b"],
    revision="aa21acc8f6546b909a92fe4d36b37ccaa69c781f",
)

MULTI_IFEVAL_BE_CONFIG = DatasetConfig(
    name="multi-ifeval-be",
    pretty_name="MultiIFEval-be",
    source="EuroEval/multi-ifeval-be",
    task=INSTRUCTION_FOLLOWING,
    languages=[BELARUSIAN],
    train_split=None,
    val_split=None,
    revision="76d1554e03d2b9e0cfad28917088cce264dadfbc",
)

BERTE_WD_CONFIG = DatasetConfig(
    name="berte-wd",
    pretty_name="BeRTE-WD",
    source="EuroEval/berte-wd-mini",
    task=NLI,
    languages=[BELARUSIAN],
    labels=["entailment", "non_entailment"],
    prompt_label_mapping=dict(entailment="праўда", non_entailment="не вынікае"),
    prompt_prefix="Ніжэй прыведзены пары сцвярджэнняў. Вызначце, ці вынікае "
    "другое сцвярджэнне з першага. Адказ можа быць {labels_str}.",
    prompt_template="{text}\nІмплікацыя: {label}",
    instruction_prompt="{text}\n\nВызначце, ці вынікае другое сцвярджэнне з "
    "першага. Адкажыце толькі {labels_str}, і нічога іншага.",
    revision="8eda523691205afcbadc02a4c50950b0237dd278",
)

BEWIC_CONFIG = DatasetConfig(
    name="bewic",
    pretty_name="BeWiC",
    source="EuroEval/bewic-mini",
    task=WIC,
    languages=[BELARUSIAN],
    revision="c7fc25a47855aa140b409c17ba7cf941bcea36b8",
)

BELACOLA_CONFIG = DatasetConfig(
    name="belacola",
    pretty_name="BelaCoLA",
    source="EuroEval/belacola-mini",
    task=LA,
    languages=[BELARUSIAN],
    revision="5437400f01c540d27d2f3966e800b010467417c2",
)

# Unofficial datasets ###

MULTI_BFCL_BE_CONFIG = DatasetConfig(
    name="multi-bfcl-be",
    pretty_name="MultiBFCL-be",
    source="EuroEval/multi-bfcl-be-mini",
    task=TOOL_CALLING,
    languages=[BELARUSIAN],
    train_split=None,
    revision="1dbf62751841bfdca40a4ec73e4d3156d796c479",
)


FLORES_BE_EN_CONFIG = TranslationDatasetConfig(
    name="flores-be-en",
    pretty_name="FLORES-be-en",
    source="EuroEval/flores-be-en",
    task=TRANSLATION,
    languages=[BELARUSIAN],
    source_language=BELARUSIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="d23b3b45142f79c9b84a7bab6ddc0effb7bd7070",
)

SCALA_BE_CONFIG = DatasetConfig(
    name="scala-be",
    pretty_name="ScaLA-be",
    source="EuroEval/scala-be",
    task=LA,
    languages=[BELARUSIAN],
    unofficial=True,
    revision="7fa44f23be44362b66c4d6fe47df352f027b7958",
)

RAGTRUTH_BE_CONFIG = DatasetConfig(
    name="ragtruth-be",
    pretty_name="RAGTruth-be",
    source="EuroEval/ragtruth-translated-hallucinations-be-mini",
    task=HALLU,
    languages=[BELARUSIAN],
    train_split=None,
    revision="6b38f8d4111561193922933c1bfe25fe198eff28",
)
