"""All Spanish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import ES
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

SENTIMENT_HEADLINES_CONFIG = DatasetConfig(
    name="sentiment-headlines-es",
    pretty_name="the truncated version of the Spanish sentiment classification dataset "
    "SentimentHeadlines",
    huggingface_id="EuroEval/sentiment-headlines-es",
    task=SENT,
    languages=[ES],
)

SCALA_ES_CONFIG = DatasetConfig(
    name="scala-es",
    pretty_name="the Spanish part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-es",
    task=LA,
    languages=[ES],
)

CONLL_ES_CONFIG = DatasetConfig(
    name="conll-es",
    pretty_name="the Spanish part of the truncated version of the named entity "
    "recognition dataset CoNLL 2002",
    huggingface_id="EuroEval/conll-es-mini",
    task=NER,
    languages=[ES],
)

MLQA_ES_CONFIG = DatasetConfig(
    name="mlqa-es",
    pretty_name="the Spanish version of the reading comprehension dataset MLQA",
    huggingface_id="EuroEval/mlqa-es",
    task=RC,
    languages=[ES],
)

MLSUM_ES_CONFIG = DatasetConfig(
    name="mlsum-es",
    pretty_name="the truncated version of the Spanish summarisation dataset MLSum-es",
    huggingface_id="EuroEval/mlsum-es-mini",
    task=SUMM,
    languages=[ES],
)

MMLU_ES_CONFIG = DatasetConfig(
    name="mmlu-es",
    pretty_name="the truncated version of the Spanish knowledge dataset MMLU-es, "
    "translated from the English MMLU dataset",
    huggingface_id="EuroEval/mmlu-es-mini",
    task=KNOW,
    languages=[ES],
)

HELLASWAG_ES_CONFIG = DatasetConfig(
    name="hellaswag-es",
    pretty_name="the truncated version of the Spanish common-sense reasoning dataset "
    "HellaSwag-es, translated from the English HellaSwag dataset",
    huggingface_id="EuroEval/hellaswag-es-mini",
    task=COMMON_SENSE,
    languages=[ES],
)

EUROPEAN_VALUES_ES_CONFIG = DatasetConfig(
    name="european-values-es",
    pretty_name="the Spanish version of the European values evaluation dataset",
    huggingface_id="EuroEval/european-values-es",
    task=EUROPEAN_VALUES,
    languages=[ES],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
)


### Unofficial datasets ###

XQUAD_ES_CONFIG = DatasetConfig(
    name="xquad-es",
    pretty_name="the Spanish version of the reading comprehension dataset XQuAD",
    huggingface_id="EuroEval/xquad-es",
    task=RC,
    languages=[ES],
    unofficial=True,
)

BELEBELE_ES_CONFIG = DatasetConfig(
    name="belebele-es",
    pretty_name="the Spanish multiple choice reading comprehension dataset "
    "BeleBele-es, translated from the English BeleBele dataset",
    huggingface_id="EuroEval/belebele-es-mini",
    task=MCRC,
    languages=[ES],
    unofficial=True,
)

MULTI_WIKI_QA_ES_CONFIG = DatasetConfig(
    name="multi-wiki-qa-es",
    pretty_name="the truncated version of the Spanish part of the reading "
    "comprehension dataset MultiWikiQA",
    huggingface_id="EuroEval/multi-wiki-qa-es-mini",
    task=RC,
    languages=[ES],
    unofficial=True,
)

GOLDENSWAG_ES_CONFIG = DatasetConfig(
    name="goldenswag-es",
    pretty_name="the truncated version of the Spanish common-sense reasoning "
    "dataset GoldenSwag-es, translated from the English GoldenSwag dataset",
    huggingface_id="EuroEval/goldenswag-es-mini",
    task=COMMON_SENSE,
    languages=[ES],
    unofficial=True,
)

EUROPEAN_VALUES_SITUATIONAL_ES_CONFIG = DatasetConfig(
    name="european-values-situational-es",
    pretty_name="the Spanish version of the European values evaluation dataset, where "
    "the questions are phrased in a situational way",
    huggingface_id="EuroEval/european-values-situational-es",
    task=EUROPEAN_VALUES,
    languages=[ES],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)

EUROPEAN_VALUES_COMPLETIONS_ES_CONFIG = DatasetConfig(
    name="european-values-completions-es",
    pretty_name="the Spanish version of the European values evaluation dataset, where "
    "the questions are phrased as sentence completions",
    huggingface_id="EuroEval/european-values-completions-es",
    task=EUROPEAN_VALUES,
    languages=[ES],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)
