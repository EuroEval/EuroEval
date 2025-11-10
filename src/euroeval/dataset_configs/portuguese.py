"""All Portuguese dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import EUROPEAN_PORTUGUESE, PORTUGUESE
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

SST2_PT_CONFIG = DatasetConfig(
    name="sst2-pt",
    pretty_name="SST2-pt",
    source="EuroEval/sst2-pt-mini",
    task=SENT,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    _labels=["positive", "negative"],
)

SCALA_PT = DatasetConfig(
    name="scala-pt",
    pretty_name="ScaLA-pt",
    source="EuroEval/scala-pt",
    task=LA,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
)

HAREM_CONFIG = DatasetConfig(
    name="harem",
    pretty_name="HAREM",
    source="EuroEval/harem",
    task=NER,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
)

MULTI_WIKI_QA_PT_CONFIG = DatasetConfig(
    name="multi-wiki-qa-pt",
    pretty_name="MultiWikiQA-pt",
    source="EuroEval/multi-wiki-qa-pt-pt-mini",
    task=RC,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
)

PUBLICO_CONFIG = DatasetConfig(
    name="publico",
    pretty_name="Publico",
    source="EuroEval/publico-mini",
    task=SUMM,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
)

MMLU_PT_CONFIG = DatasetConfig(
    name="mmlu-pt",
    pretty_name="MMLU-pt",
    source="EuroEval/mmlu-pt-mini",
    task=KNOW,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
)

GOLDENSWAG_PT_CONFIG = DatasetConfig(
    name="goldenswag-pt",
    pretty_name="GoldenSwag-pt",
    source="EuroEval/goldenswag-pt-mini",
    task=COMMON_SENSE,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
)

VALEU_PT_CONFIG = DatasetConfig(
    name="valeu-pt",
    pretty_name="VaLEU-pt",
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
    pretty_name="BoolQ-pt",
    source="EuroEval/boolq-pt",
    task=MCRC,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    unofficial=True,
)

WINOGRANDE_PT_CONFIG = DatasetConfig(
    name="winogrande-pt",
    pretty_name="Winogrande-pt",
    source="EuroEval/winogrande-pt",
    task=COMMON_SENSE,
    languages=[PORTUGUESE, EUROPEAN_PORTUGUESE],
    _labels=["a", "b"],
    unofficial=True,
)
