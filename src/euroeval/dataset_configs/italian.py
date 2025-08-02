"""All Italian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import IT
from ..tasks import COMMON_SENSE, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

SENTIPOLC_CONFIG = DatasetConfig(
    name="sentipolc16",
    pretty_name="the truncated version of the Italian sentiment classification "
    "dataset Sentipolc-16",
    huggingface_id="EuroEval/sentipolc16-mini",
    task=SENT,
    languages=[IT],
)

SCALA_IT_CONFIG = DatasetConfig(
    name="scala-it",
    pretty_name="the Italian part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-it",
    task=LA,
    languages=[IT],
)

MULTINERD_IT_CONFIG = DatasetConfig(
    name="multinerd-it",
    pretty_name="the truncated version of the Italian part of the named "
    "entity recognition dataset MultiNERD",
    huggingface_id="EuroEval/multinerd-mini-it",
    task=NER,
    languages=[IT],
)

SQUAD_IT_CONFIG = DatasetConfig(
    name="squad-it",
    pretty_name="the truncated version of the Italian reading comprehension dataset "
    "SQuAD-it, translated from the English SQuAD dataset",
    huggingface_id="EuroEval/squad-it-mini",
    task=RC,
    languages=[IT],
)

ILPOST_SUM_CONFIG = DatasetConfig(
    name="ilpost-sum",
    pretty_name="the truncated version of the Italian summarisation dataset IlPost-Sum",
    huggingface_id="EuroEval/ilpost-sum",
    task=SUMM,
    languages=[IT],
)

MMLU_IT_CONFIG = DatasetConfig(
    name="mmlu-it",
    pretty_name="the truncated version of the Italian knowledge dataset MMLU-it, "
    "translated from the English MMLU dataset",
    huggingface_id="EuroEval/mmlu-it-mini",
    task=KNOW,
    languages=[IT],
)

HELLASWAG_IT_CONFIG = DatasetConfig(
    name="hellaswag-it",
    pretty_name="the truncated version of the Italian common-sense reasoning dataset "
    "HellaSwag-it, translated from the English HellaSwag dataset",
    huggingface_id="EuroEval/hellaswag-it-mini",
    task=COMMON_SENSE,
    languages=[IT],
)


### Unofficial datasets ###

WIKINEURAL_IT_CONFIG = DatasetConfig(
    name="wikineural-it",
    pretty_name="the truncated version of the Italian named "
    "entity recognition dataset WikiNEuRal IT",
    huggingface_id="EuroEval/wikineural-mini-it",
    task=NER,
    languages=[IT],
    unofficial=True,
)

BELEBELE_IT_CONFIG = DatasetConfig(
    name="belebele-it",
    pretty_name="the Italian multiple choice reading comprehension dataset "
    "BeleBele-it, translated from the English BeleBele dataset",
    huggingface_id="EuroEval/belebele-it-mini",
    task=MCRC,
    languages=[IT],
    unofficial=True,
)

MULTI_WIKI_QA_IT_CONFIG = DatasetConfig(
    name="multi-wiki-qa-it",
    pretty_name="the truncated version of the Italian part of the reading "
    "comprehension dataset MultiWikiQA",
    huggingface_id="EuroEval/multi-wiki-qa-it-mini",
    task=RC,
    languages=[IT],
    unofficial=True,
)

GOLDENSWAG_IT_CONFIG = DatasetConfig(
    name="goldenswag-it",
    pretty_name="the truncated version of the Italian common-sense reasoning "
    "dataset GoldenSwag-it, translated from the English GoldenSwag dataset",
    huggingface_id="EuroEval/goldenswag-it-mini",
    task=COMMON_SENSE,
    languages=[IT],
    unofficial=True,
)
