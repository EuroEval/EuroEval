"""All Belarusian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import BELARUSIAN
from ..tasks import (
    COMMON_SENSE,
    HALLU,
    INSTRUCTION_FOLLOWING,
    LA,
    NER,
    NLI,
    RC,
    SENT,
    WIC,
)

# Official datasets ###

BESLS_CONFIG = DatasetConfig(
    name="besls",
    pretty_name="BeSLS",
    source="EuroEval/besls",
    task=SENT,
    languages=[BELARUSIAN],
)

SCALA_BE_CONFIG = DatasetConfig(
    name="scala-be",
    pretty_name="ScaLA-be",
    source="EuroEval/scala-be",
    task=LA,
    languages=[BELARUSIAN],
)

WIKIANN_BE_CONFIG = DatasetConfig(
    name="wikiann-be",
    pretty_name="WikiANN-be",
    source="EuroEval/wikiann-be-mini",
    task=NER,
    languages=[BELARUSIAN],
)

MULTI_WIKI_QA_BE_CONFIG = DatasetConfig(
    name="multi-wiki-qa-be",
    pretty_name="MultiWikiQA-be",
    source="EuroEval/multi-wiki-qa-be-mini",
    task=RC,
    languages=[BELARUSIAN],
)

BE_WSC_CONFIG = DatasetConfig(
    name="be-wsc",
    pretty_name="BE-WSC",
    source="EuroEval/be-wsc",
    task=COMMON_SENSE,
    languages=[BELARUSIAN],
)

MULTI_IFEVAL_BE_CONFIG = DatasetConfig(
    name="multi-ifeval-be",
    pretty_name="MultiIFEval-be",
    source="EuroEval/multi-ifeval-be",
    task=INSTRUCTION_FOLLOWING,
    languages=[BELARUSIAN],
    train_split=None,
    val_split=None,
)


# Unofficial datasets ###

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
    unofficial=True,
)

RAGTRUTH_BE_CONFIG = DatasetConfig(
    name="ragtruth-be",
    pretty_name="RAGTruth-be",
    source="EuroEval/ragtruth-translated-hallucinations-be-mini",
    task=HALLU,
    languages=[BELARUSIAN],
    train_split=None,
    unofficial=True,
)

BELACOLA_CONFIG = DatasetConfig(
    name="belacola",
    pretty_name="BelaCoLA",
    source="EuroEval/belacola-mini",
    task=LA,
    languages=[BELARUSIAN],
    unofficial=True,
)

BEWIC_CONFIG = DatasetConfig(
    name="bewic",
    pretty_name="BeWiC",
    source="EuroEval/bewic-mini",
    task=WIC,
    languages=[BELARUSIAN],
    unofficial=True,
)
