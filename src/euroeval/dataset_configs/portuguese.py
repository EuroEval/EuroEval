"""All Portuguese dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import EUROPEAN_PORTUGUESE, PORTUGUESE
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

SST2_PT_CONFIG = DatasetConfig(
    name="sst2-pt",
    source="EuroEval/sst2-pt-mini",
    task=SENT,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    _labels=["positive", "negative"],
)

SCALA_PT = DatasetConfig(
    name="scala-pt",
    source="EuroEval/scala-pt",
    task=LA,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
)

HAREM_CONFIG = DatasetConfig(
    name="harem",
    source="EuroEval/harem",
    task=NER,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
)

MULTI_WIKI_QA_PT_CONFIG = DatasetConfig(
    name="multi-wiki-qa-pt",
    source="EuroEval/multi-wiki-qa-pt-pt-mini",
    task=RC,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
)

PUBLICO_CONFIG = DatasetConfig(
    name="publico",
    source="EuroEval/publico-mini",
    task=SUMM,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
)

MMLU_PT_CONFIG = DatasetConfig(
    name="mmlu-pt",
    source="EuroEval/mmlu-pt-mini",
    task=KNOW,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
)

GOLDENSWAG_PT_CONFIG = DatasetConfig(
    name="goldenswag-pt",
    source="EuroEval/goldenswag-pt-mini",
    task=COMMON_SENSE,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
)

EUROPEAN_VALUES_PT_CONFIG = DatasetConfig(
    name="european-values-pt",
    source="EuroEval/european-values-pt",
    task=EUROPEAN_VALUES,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
)


### Unofficial datasets ###

BOOLQ_PT_CONFIG = DatasetConfig(
    name="boolq-pt",
    source="EuroEval/boolq-pt",
    task=MCRC,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    unofficial=True,
)

WINOGRANDE_PT_CONFIG = DatasetConfig(
    name="winogrande-pt",
    source="EuroEval/winogrande-pt",
    task=COMMON_SENSE,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    _labels=["a", "b"],
    unofficial=True,
)

EUROPEAN_VALUES_SITUATIONAL_PT_CONFIG = DatasetConfig(
    name="european-values-situational-pt",
    source="EuroEval/european-values-situational-pt",
    task=EUROPEAN_VALUES,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)

EUROPEAN_VALUES_COMPLETIONS_PT_CONFIG = DatasetConfig(
    name="european-values-completions-pt",
    source="EuroEval/european-values-completions-pt",
    task=EUROPEAN_VALUES,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)
