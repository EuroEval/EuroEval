"""All CATALAN dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import CATALAN
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

GUIA_CAT_CONFIG = DatasetConfig(
    name="guia-cat",
    pretty_name="GuiaCat",
    source="EuroEval/guia-cat-mini",
    task=SENT,
    languages=[CATALAN],
)

SCALA_CA_CONFIG = DatasetConfig(
    name="scala-ca",
    pretty_name="ScaLA-ca",
    source="EuroEval/scala-ca",
    task=LA,
    languages=[CATALAN],
)

WIKIANN_CA_CONFIG = DatasetConfig(
    name="wikiann-ca",
    pretty_name="WikiANN-ca",
    source="EuroEval/wikiann-ca-mini",
    task=NER,
    languages=[CATALAN],
)

MULTI_WIKI_QA_CA_CONFIG = DatasetConfig(
    name="multi-wiki-qa-ca",
    pretty_name="MultiWikiQA-ca",
    source="EuroEval/multi-wiki-qa-ca-mini",
    task=RC,
    languages=[CATALAN],
)

DACSA_CA_CONFIG = DatasetConfig(
    name="dacsa-ca",
    pretty_name="DACSA-ca",
    source="EuroEval/dacsa-ca-mini",
    task=SUMM,
    languages=[CATALAN],
)

MMLU_CA_CONFIG = DatasetConfig(
    name="mmlu-ca",
    pretty_name="MMLU-ca",
    source="EuroEval/mmlu-ca-mini",
    task=KNOW,
    languages=[CATALAN],
)

WINOGRANDE_CA_CONFIG = DatasetConfig(
    name="winogrande-ca",
    pretty_name="Winogrande-ca",
    source="EuroEval/winogrande-ca",
    task=COMMON_SENSE,
    languages=[CATALAN],
    _labels=["a", "b"],
)
