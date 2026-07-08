"""All Luxembourgish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import LUXEMBOURGISH
from ..tasks import LA, NER, NLI, RC, SENT, SUMM, TEXT_CLASSIFICATION

# Official datasets ###

LTZGLUE_SA_CONFIG = DatasetConfig(
    name="ltzglue-sa",
    pretty_name="ltzGLUE-SA",
    source="plumaj/ltzglue-sa",
    task=SENT,
    languages=[LUXEMBOURGISH],
    labels=["negative", "neutral", "positive"],
)

LTZGLUE_LA_BINARY_CONFIG = DatasetConfig(
    name="ltzglue-la-binary",
    pretty_name="ltzGLUE-LA",
    source="plumaj/ltzglue-lab",
    task=LA,
    languages=[LUXEMBOURGISH],
    labels=["incorrect", "correct"],
)

LTZGLUE_NER_CONFIG = DatasetConfig(
    name="ltzglue-ner",
    pretty_name="ltzGLUE-NER",
    source="plumaj/ltzglue-ner",
    task=NER,
    languages=[LUXEMBOURGISH],
)

MULTI_WIKI_QA_LB_CONFIG = DatasetConfig(
    name="multi-wiki-qa-lb",
    pretty_name="MultiWikiQA-lb",
    source="alexandrainst/multi-wiki-qa",
    subset="lb",
    task=RC,
    languages=[LUXEMBOURGISH],
)

LUXGEN_SUMM_CONFIG = DatasetConfig(
    name="luxgen-summ",
    pretty_name="LuxGen-Summ",
    source="EuroEval/luxgen-summ",
    task=SUMM,
    languages=[LUXEMBOURGISH],
)

LTZGLUE_HC_CONFIG = DatasetConfig(
    name="ltzglue-hc",
    pretty_name="ltzGLUE-HC",
    source="plumaj/ltzglue-hc",
    task=TEXT_CLASSIFICATION,
    languages=[LUXEMBOURGISH],
    labels=["no", "yes"],
)

LTZGLUE_RTE_CONFIG = DatasetConfig(
    name="ltzglue-rte",
    pretty_name="ltzGLUE-RTE",
    source="plumaj/ltzglue-rte",
    task=NLI,
    languages=[LUXEMBOURGISH],
    labels=["entailment", "neutral", "contradiction"],
)

# Unofficial datasets ###

SCALA_LB_CONFIG = DatasetConfig(
    name="scala-lb",
    pretty_name="ScaLA-lb",
    source="EuroEval/scala-lb",
    task=LA,
    languages=[LUXEMBOURGISH],
    unofficial=True,
    labels=["incorrect", "correct"],
)

LTZGLUE_LA_MULTI_CONFIG = DatasetConfig(
    name="ltzglue-la-multi",
    pretty_name="ltzGLUE-LA (Multi-class)",
    source="plumaj/ltzglue-lam",
    task=LA,
    languages=[LUXEMBOURGISH],
    unofficial=True,
    labels=["correct", "word_order", "agreement", "morphology", "other"],
)

LTZGLUE_ID_CONFIG = DatasetConfig(
    name="ltzglue-id",
    pretty_name="ltzGLUE-ID",
    source="plumaj/ltzglue-id",
    task=TEXT_CLASSIFICATION,
    languages=[LUXEMBOURGISH],
    unofficial=True,
    labels=["question", "statement", "command", "exclamation"],
)

LTZGLUE_TC_CONFIG = DatasetConfig(
    name="ltzglue-tc",
    pretty_name="ltzGLUE-TC",
    source="plumaj/ltzglue-tc",
    task=TEXT_CLASSIFICATION,
    languages=[LUXEMBOURGISH],
    unofficial=True,
    labels=["politics", "sports", "culture", "economy", "technology"],
)
