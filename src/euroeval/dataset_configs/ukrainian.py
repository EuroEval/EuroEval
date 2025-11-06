"""All Ukrainian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import UKRAINIAN
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

CROSS_DOMAIN_UK_REVIEWS_CONFIG = DatasetConfig(
    name="cross-domain-uk-reviews",
    pretty_name="Cross Domain Ukrainian Reviews",
    source="EuroEval/cross-domain-uk-reviews-mini",
    task=SENT,
    languages=[UKRAINIAN],
)

SCALA_UK_CONFIG = DatasetConfig(
    name="scala-uk",
    pretty_name="ScaLA-uk",
    source="EuroEval/scala-uk",
    task=LA,
    languages=[UKRAINIAN],
)

NER_UK_CONFIG = DatasetConfig(
    name="ner-uk",
    pretty_name="NER-uk",
    source="EuroEval/ner-uk-mini",
    task=NER,
    languages=[UKRAINIAN],
)

MULTI_WIKI_QA_UK_CONFIG = DatasetConfig(
    name="multi-wiki-qa-uk",
    pretty_name="MultiWikiQA-uk",
    source="EuroEval/multi-wiki-qa-uk-mini",
    task=RC,
    languages=[UKRAINIAN],
)

LR_SUM_UK_CONFIG = DatasetConfig(
    name="lr-sum-uk",
    pretty_name="LRSum-uk",
    source="EuroEval/lr-sum-uk-mini",
    task=SUMM,
    languages=[UKRAINIAN],
)

GLOBAL_MMLU_UK_CONFIG = DatasetConfig(
    name="global-mmlu-uk",
    pretty_name="GlobalMMLU-uk",
    source="EuroEval/global-mmlu-uk-mini",
    task=KNOW,
    languages=[UKRAINIAN],
)

WINOGRANDE_UK_CONFIG = DatasetConfig(
    name="winogrande-uk",
    pretty_name="Winogrande-uk",
    source="EuroEval/winogrande-uk",
    task=COMMON_SENSE,
    languages=[UKRAINIAN],
    _labels=["a", "b"],
)
