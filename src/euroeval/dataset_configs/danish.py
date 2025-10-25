"""All Danish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import DANISH
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

ANGRY_TWEETS_CONFIG = DatasetConfig(
    name="angry-tweets",
    source="EuroEval/angry-tweets-mini",
    task=SENT,
    languages=[DANISH],
)

SCALA_DA_CONFIG = DatasetConfig(
    name="scala-da", source="EuroEval/scala-da", task=LA, languages=[DANISH]
)

DANSK_CONFIG = DatasetConfig(
    name="dansk", source="EuroEval/dansk-mini", task=NER, languages=[DANISH]
)

MULTI_WIKI_QA_DA_CONFIG = DatasetConfig(
    name="multi-wiki-qa-da",
    source="EuroEval/multi-wiki-qa-da-mini",
    task=RC,
    languages=[DANISH],
)

NORDJYLLAND_NEWS_CONFIG = DatasetConfig(
    name="nordjylland-news",
    source="EuroEval/nordjylland-news-mini",
    task=SUMM,
    languages=[DANISH],
)

DANSKE_TALEMAADER_CONFIG = DatasetConfig(
    name="danske-talemaader",
    source="EuroEval/danske-talemaader",
    task=KNOW,
    languages=[DANISH],
)

DANISH_CITIZEN_TESTS_CONFIG = DatasetConfig(
    name="danish-citizen-tests",
    source="EuroEval/danish-citizen-tests-updated",
    task=KNOW,
    languages=[DANISH],
)

HELLASWAG_DA_CONFIG = DatasetConfig(
    name="hellaswag-da",
    source="EuroEval/hellaswag-da-mini",
    task=COMMON_SENSE,
    languages=[DANISH],
)

EUROPEAN_VALUES_DA_CONFIG = DatasetConfig(
    name="european-values-da",
    source="EuroEval/european-values-da",
    task=EUROPEAN_VALUES,
    languages=[DANISH],
    splits=["test"],
    bootstrap_samples=False,
)


### Unofficial datasets ###

DANE_CONFIG = DatasetConfig(
    name="dane",
    source="EuroEval/dane-mini",
    task=NER,
    languages=[DANISH],
    unofficial=True,
)

MMLU_DA_CONFIG = DatasetConfig(
    name="mmlu-da",
    source="EuroEval/mmlu-da-mini",
    task=KNOW,
    languages=[DANISH],
    unofficial=True,
)

ARC_DA_CONFIG = DatasetConfig(
    name="arc-da",
    source="EuroEval/arc-da-mini",
    task=KNOW,
    languages=[DANISH],
    unofficial=True,
)

BELEBELE_DA_CONFIG = DatasetConfig(
    name="belebele-da",
    source="EuroEval/belebele-da-mini",
    task=MCRC,
    languages=[DANISH],
    unofficial=True,
)

SCANDIQA_DA_CONFIG = DatasetConfig(
    name="scandiqa-da",
    source="EuroEval/scandiqa-da-mini",
    task=RC,
    languages=[DANISH],
    unofficial=True,
)

GOLDENSWAG_DA_CONFIG = DatasetConfig(
    name="goldenswag-da",
    source="EuroEval/goldenswag-da-mini",
    task=COMMON_SENSE,
    languages=[DANISH],
    unofficial=True,
)

WINOGRANDE_DA_CONFIG = DatasetConfig(
    name="winogrande-da",
    source="EuroEval/winogrande-da",
    task=COMMON_SENSE,
    languages=[DANISH],
    _labels=["a", "b"],
    unofficial=True,
)

EUROPEAN_VALUES_SITUATIONAL_DA_CONFIG = DatasetConfig(
    name="european-values-situational-da",
    source="EuroEval/european-values-situational-da",
    task=EUROPEAN_VALUES,
    languages=[DANISH],
    splits=["test"],
    bootstrap_samples=False,
    unofficial=True,
)

EUROPEAN_VALUES_COMPLETIONS_DA_CONFIG = DatasetConfig(
    name="european-values-completions-da",
    source="EuroEval/european-values-completions-da",
    task=EUROPEAN_VALUES,
    languages=[DANISH],
    splits=["test"],
    bootstrap_samples=False,
    unofficial=True,
)
