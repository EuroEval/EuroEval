"""All Ukrainian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import UK
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

CROSS_DOMAIN_UK_REVIEWS_CONFIG = DatasetConfig(
    name="cross-domain-uk-reviews",
    pretty_name="the truncated version of the Ukrainian sentiment classification "
    "dataset Cross-Domain UK Reviews",
    huggingface_id="EuroEval/cross-domain-uk-reviews-mini",
    task=SENT,
    languages=[UK],
)

SCALA_UK_CONFIG = DatasetConfig(
    name="scala-uk",
    pretty_name="the Ukrainian part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-uk",
    task=LA,
    languages=[UK],
)

NER_UK_CONFIG = DatasetConfig(
    name="ner-uk",
    pretty_name="the truncated version of the Ukrainian named entity recognition "
    "dataset NER-uk",
    huggingface_id="EuroEval/ner-uk-mini",
    task=NER,
    languages=[UK],
)

MULTI_WIKI_QA_UK_CONFIG = DatasetConfig(
    name="multi-wiki-qa-uk",
    pretty_name="the truncated version of the Ukrainian part of the reading "
    "comprehension dataset MultiWikiQA",
    huggingface_id="EuroEval/multi-wiki-qa-uk-mini",
    task=RC,
    languages=[UK],
)

LR_SUM_UK_CONFIG = DatasetConfig(
    name="lr-sum-uk",
    pretty_name="the truncated version of the Ukrainian part of the "
    "summarisation dataset LR-Sum",
    huggingface_id="EuroEval/lr-sum-uk-mini",
    task=SUMM,
    languages=[UK],
)

GLOBAL_MMLU_UK_CONFIG = DatasetConfig(
    name="global-mmlu-uk",
    pretty_name="the truncated version of the Ukrainian knowledge dataset "
    "GlobalMMLU-uk, machine translated from the English GlobalMMLU dataset",
    huggingface_id="EuroEval/global-mmlu-uk-mini",
    task=KNOW,
    languages=[UK],
)

WINOGRANDE_UK_CONFIG = DatasetConfig(
    name="winogrande-uk",
    pretty_name="the Ukrainian common-sense reasoning dataset Winogrande-uk, "
    "translated from the English Winogrande dataset",
    huggingface_id="EuroEval/winogrande-uk",
    task=COMMON_SENSE,
    languages=[UK],
    _labels=["a", "b"],
)
