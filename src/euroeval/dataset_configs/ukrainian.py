"""All Slovak dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import UK
from ..tasks import LA, NER, RC, SENT

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
