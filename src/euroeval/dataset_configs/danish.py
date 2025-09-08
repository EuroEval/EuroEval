"""All Danish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import DA
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

ANGRY_TWEETS_CONFIG = DatasetConfig(
    name="angry-tweets",
    pretty_name="the truncated version of the Danish sentiment classification "
    "dataset AngryTweets",
    huggingface_id="EuroEval/angry-tweets-mini",
    task=SENT,
    languages=[DA],
)

SCALA_DA_CONFIG = DatasetConfig(
    name="scala-da",
    pretty_name="the Danish part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-da",
    task=LA,
    languages=[DA],
)

DANSK_CONFIG = DatasetConfig(
    name="dansk",
    pretty_name="the truncated version of the Danish named entity recognition "
    "dataset DANSK",
    huggingface_id="EuroEval/dansk-mini",
    task=NER,
    languages=[DA],
)

SCANDIQA_DA_CONFIG = DatasetConfig(
    name="scandiqa-da",
    pretty_name="the Danish part of the truncated version of the question answering "
    "dataset ScandiQA",
    huggingface_id="EuroEval/scandiqa-da-mini",
    task=RC,
    languages=[DA],
)

NORDJYLLAND_NEWS_CONFIG = DatasetConfig(
    name="nordjylland-news",
    pretty_name="the truncated version of the Danish summarisation dataset "
    "Nordjylland News",
    huggingface_id="EuroEval/nordjylland-news-mini",
    task=SUMM,
    languages=[DA],
)

DANSKE_TALEMAADER_CONFIG = DatasetConfig(
    name="danske-talemaader",
    pretty_name="the truncated version of the Danish knowledge dataset Danske "
    "Talemåder",
    huggingface_id="EuroEval/danske-talemaader",
    task=KNOW,
    languages=[DA],
)

DANISH_CITIZEN_TESTS_CONFIG = DatasetConfig(
    name="danish-citizen-tests",
    pretty_name="the Danish knowledge dataset Danish Citizen Tests",
    huggingface_id="EuroEval/danish-citizen-tests-updated",
    task=KNOW,
    languages=[DA],
)

HELLASWAG_DA_CONFIG = DatasetConfig(
    name="hellaswag-da",
    pretty_name="the truncated version of the Danish common-sense reasoning dataset "
    "HellaSwag-da, translated from the English HellaSwag dataset",
    huggingface_id="EuroEval/hellaswag-da-mini",
    task=COMMON_SENSE,
    languages=[DA],
)

EUROPEAN_VALUES_DA_CONFIG = DatasetConfig(
    name="european-values-da",
    pretty_name="the Danish version of the European values evaluation dataset",
    huggingface_id="EuroEval/european-values-da",
    task=EUROPEAN_VALUES,
    languages=[DA],
    splits=["test"],
    bootstrap_samples=False,
)


### Unofficial datasets ###

DANE_CONFIG = DatasetConfig(
    name="dane",
    pretty_name="the truncated version of the Danish named entity recognition "
    "dataset DaNE",
    huggingface_id="EuroEval/dane-mini",
    task=NER,
    languages=[DA],
    unofficial=True,
)

MMLU_DA_CONFIG = DatasetConfig(
    name="mmlu-da",
    pretty_name="the truncated version of the Danish knowledge dataset MMLU-da, "
    "translated from the English MMLU dataset",
    huggingface_id="EuroEval/mmlu-da-mini",
    task=KNOW,
    languages=[DA],
    unofficial=True,
)

ARC_DA_CONFIG = DatasetConfig(
    name="arc-da",
    pretty_name="the truncated version of the Danish knowledge dataset ARC-da, "
    "translated from the English ARC dataset",
    huggingface_id="EuroEval/arc-da-mini",
    task=KNOW,
    languages=[DA],
    unofficial=True,
)

BELEBELE_DA_CONFIG = DatasetConfig(
    name="belebele-da",
    pretty_name="the Danish multiple choice reading comprehension dataset BeleBele-da, "
    "translated from the English BeleBele dataset",
    huggingface_id="EuroEval/belebele-da-mini",
    task=MCRC,
    languages=[DA],
    unofficial=True,
)

MULTI_WIKI_QA_DA_CONFIG = DatasetConfig(
    name="multi-wiki-qa-da",
    pretty_name="the truncated version of the Danish part of the reading "
    "comprehension dataset MultiWikiQA",
    huggingface_id="EuroEval/multi-wiki-qa-da-mini",
    task=RC,
    languages=[DA],
    unofficial=True,
)

GOLDENSWAG_DA_CONFIG = DatasetConfig(
    name="goldenswag-da",
    pretty_name="the truncated version of the Danish common-sense reasoning "
    "dataset GoldenSwag-da, translated from the English GoldenSwag dataset",
    huggingface_id="EuroEval/goldenswag-da-mini",
    task=COMMON_SENSE,
    languages=[DA],
    unofficial=True,
)

EUROPEAN_VALUES_SITUATIONAL_DA_CONFIG = DatasetConfig(
    name="european-values-situational-da",
    pretty_name="the Danish version of the European values evaluation dataset, where "
    "the questions are phrased in a situational way",
    huggingface_id="EuroEval/european-values-situational-da",
    task=EUROPEAN_VALUES,
    languages=[DA],
    splits=["test"],
    bootstrap_samples=False,
    unofficial=True,
)

EUROPEAN_VALUES_COMPLETIONS_DA_CONFIG = DatasetConfig(
    name="european-values-completions-da",
    pretty_name="the Danish version of the European values evaluation dataset, where "
    "the questions are phrased as sentence completions",
    huggingface_id="EuroEval/european-values-completions-da",
    task=EUROPEAN_VALUES,
    languages=[DA],
    splits=["test"],
    bootstrap_samples=False,
    unofficial=True,
)
