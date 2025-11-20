"""All CATALAN dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import CATALAN
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT

### Official datasets ###

GUIA_CAT_CONFIG = DatasetConfig(
    name="guia-cat",
    pretty_name="GuiaCat",
    source="EuroEval/guia-cat-mini",
    task=SENT,
    languages=[CATALAN],
)

SCALA_HR_CONFIG = DatasetConfig(
    name="scala-hr",
    pretty_name="ScaLA-hr",
    source="EuroEval/scala-hr",
    task=LA,
    languages=[CATALAN],
)

WIKIANN_HR_CONFIG = DatasetConfig(
    name="wikiann-hr",
    pretty_name="WikiANN-hr",
    source="EuroEval/wikiann-hr-mini",
    task=NER,
    languages=[CATALAN],
)

MULTI_WIKI_QA_HR_CONFIG = DatasetConfig(
    name="multi-wiki-qa-hr",
    pretty_name="MultiWikiQA-hr",
    source="EuroEval/multi-wiki-qa-hr-mini",
    task=RC,
    languages=[CATALAN],
)

MMLU_HR_CONFIG = DatasetConfig(
    name="mmlu-hr",
    pretty_name="MMLU-hr",
    source="EuroEval/mmlu-hr-mini",
    task=KNOW,
    languages=[CATALAN],
)

WINOGRANDE_HR_CONFIG = DatasetConfig(
    name="winogrande-hr",
    pretty_name="Winogrande-hr",
    source="EuroEval/winogrande-hr",
    task=COMMON_SENSE,
    languages=[CATALAN],
    _labels=["a", "b"],
)
