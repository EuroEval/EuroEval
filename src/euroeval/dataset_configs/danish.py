"""All Danish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import DANISH
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

ANGRY_TWEETS_CONFIG = DatasetConfig(
    name="angry-tweets",
    pretty_name="AngryTweets",
    source="EuroEval/angry-tweets-mini",
    task=SENT,
    languages=[DANISH],
)

SCALA_DA_CONFIG = DatasetConfig(
    name="scala-da",
    pretty_name="ScaLA-da",
    source="EuroEval/scala-da",
    task=LA,
    languages=[DANISH],
)

DANSK_CONFIG = DatasetConfig(
    name="dansk",
    pretty_name="DANSK",
    source="EuroEval/dansk-mini",
    task=NER,
    languages=[DANISH],
)

MULTI_WIKI_QA_DA_CONFIG = DatasetConfig(
    name="multi-wiki-qa-da",
    pretty_name="MultiWikiQA-da",
    source="EuroEval/multi-wiki-qa-da-mini",
    task=RC,
    languages=[DANISH],
)

NORDJYLLAND_NEWS_CONFIG = DatasetConfig(
    name="nordjylland-news",
    pretty_name="Nordjylland News",
    source="EuroEval/nordjylland-news-mini",
    task=SUMM,
    languages=[DANISH],
)

DANSKE_TALEMAADER_CONFIG = DatasetConfig(
    name="danske-talemaader",
    pretty_name="Danske Talemåder",
    source="EuroEval/danske-talemaader",
    task=KNOW,
    languages=[DANISH],
)

DANISH_CITIZEN_TESTS_CONFIG = DatasetConfig(
    name="danish-citizen-tests",
    pretty_name="Danish Citizen Tests",
    source="EuroEval/danish-citizen-tests-updated",
    task=KNOW,
    languages=[DANISH],
)

HELLASWAG_DA_CONFIG = DatasetConfig(
    name="hellaswag-da",
    pretty_name="HellaSwag-da",
    source="EuroEval/hellaswag-da-mini",
    task=COMMON_SENSE,
    languages=[DANISH],
)

VALEU_DA_CONFIG = DatasetConfig(
    name="valeu-da",
    pretty_name="ValEU-da",
    source="EuroEval/european-values-da",
    task=EUROPEAN_VALUES,
    languages=[DANISH],
    splits=["test"],
    bootstrap_samples=False,
)


### Unofficial datasets ###

DANE_CONFIG = DatasetConfig(
    name="dane",
    pretty_name="DaNE",
    source="EuroEval/dane-mini",
    task=NER,
    languages=[DANISH],
    unofficial=True,
)

MMLU_DA_CONFIG = DatasetConfig(
    name="mmlu-da",
    pretty_name="MMLU-da",
    source="EuroEval/mmlu-da-mini",
    task=KNOW,
    languages=[DANISH],
    unofficial=True,
)

ARC_DA_CONFIG = DatasetConfig(
    name="arc-da",
    pretty_name="ARC-da",
    source="EuroEval/arc-da-mini",
    task=KNOW,
    languages=[DANISH],
    unofficial=True,
)

BELEBELE_DA_CONFIG = DatasetConfig(
    name="belebele-da",
    pretty_name="Belebele-da",
    source="EuroEval/belebele-da-mini",
    task=MCRC,
    languages=[DANISH],
    unofficial=True,
)

SCANDIQA_DA_CONFIG = DatasetConfig(
    name="scandiqa-da",
    pretty_name="ScandiQA-da",
    source="EuroEval/scandiqa-da-mini",
    task=RC,
    languages=[DANISH],
    unofficial=True,
)

GOLDENSWAG_DA_CONFIG = DatasetConfig(
    name="goldenswag-da",
    pretty_name="GoldenSwag-da",
    source="EuroEval/goldenswag-da-mini",
    task=COMMON_SENSE,
    languages=[DANISH],
    unofficial=True,
)

WINOGRANDE_DA_CONFIG = DatasetConfig(
    name="winogrande-da",
    pretty_name="Winogrande-da",
    source="EuroEval/winogrande-da",
    task=COMMON_SENSE,
    languages=[DANISH],
    _labels=["a", "b"],
    unofficial=True,
)
