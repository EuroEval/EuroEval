"""All Portuguese dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import PT
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

SST2_PT_CONFIG = DatasetConfig(
    name="sst2-pt",
    pretty_name="the truncated version of the Portuguese sentiment classification "
    "dataset SST2-pt, translated from the English SST2 dataset",
    huggingface_id="EuroEval/sst2-pt-mini",
    task=SENT,
    languages=[PT],
    _labels=["positive", "negative"],
)

SCALA_PT = DatasetConfig(
    name="scala-pt",
    pretty_name="the Portuguese part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-pt",
    task=LA,
    languages=[PT],
)

HAREM_CONFIG = DatasetConfig(
    name="harem",
    pretty_name="the Portuguese named entity recognition dataset HAREM",
    huggingface_id="EuroEval/harem",
    task=NER,
    languages=[PT],
)

MULTI_WIKI_QA_PT_CONFIG = DatasetConfig(
    name="multi-wiki-qa-pt",
    pretty_name="the truncated version of the Portuguese part of the reading "
    "comprehension dataset MultiWikiQA",
    huggingface_id="EuroEval/multi-wiki-qa-pt-pt-mini",
    task=RC,
    languages=[PT],
)

PUBLICO_CONFIG = DatasetConfig(
    name="publico",
    pretty_name="the truncated version of the Portuguese summarisation dataset Público",
    huggingface_id="EuroEval/publico-mini",
    task=SUMM,
    languages=[PT],
)

MMLU_PT_CONFIG = DatasetConfig(
    name="mmlu-pt",
    pretty_name="the truncated version of the Portuguese knowledge dataset MMLU-pt, "
    "translated from the English MMLU dataset",
    huggingface_id="EuroEval/mmlu-pt-mini",
    task=KNOW,
    languages=[PT],
)

GOLDENSWAG_PT_CONFIG = DatasetConfig(
    name="goldenswag-pt",
    pretty_name="the truncated version of the Portuguese common-sense reasoning "
    "dataset GoldenSwag-pt, translated from the English GoldenSwag dataset",
    huggingface_id="EuroEval/goldenswag-pt-mini",
    task=COMMON_SENSE,
    languages=[PT],
)

EUROPEAN_VALUES_PT_CONFIG = DatasetConfig(
    name="european-values-pt",
    pretty_name="the Portuguese version of the European values evaluation dataset",
    huggingface_id="EuroEval/european-values-pt",
    task=EUROPEAN_VALUES,
    languages=[PT],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
)


### Unofficial datasets ###

BOOLQ_PT_CONFIG = DatasetConfig(
    name="boolq-pt",
    pretty_name="the Portuguese multiple choice reading comprehension dataset "
    "BoolQ-pt, translated from the English BoolQ dataset",
    huggingface_id="EuroEval/boolq-pt",
    task=MCRC,
    languages=[PT],
    unofficial=True,
)

EUROPEAN_VALUES_SITUATIONAL_PT_CONFIG = DatasetConfig(
    name="european-values-situational-pt",
    pretty_name="the Portuguese version of the European values evaluation dataset, "
    "where the questions are phrased in a situational way",
    huggingface_id="EuroEval/european-values-situational-pt",
    task=EUROPEAN_VALUES,
    languages=[PT],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)

EUROPEAN_VALUES_COMPLETIONS_PT_CONFIG = DatasetConfig(
    name="european-values-completions-pt",
    pretty_name="the Portuguese version of the European values evaluation dataset, "
    "where the questions are phrased as sentence completions",
    huggingface_id="EuroEval/european-values-completions-pt",
    task=EUROPEAN_VALUES,
    languages=[PT],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)
