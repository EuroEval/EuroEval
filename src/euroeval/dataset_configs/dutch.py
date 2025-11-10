"""All Dutch dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import DUTCH
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

DBRD_CONFIG = DatasetConfig(
    name="dbrd",
    pretty_name="DBRD",
    source="EuroEval/dbrd-mini",
    task=SENT,
    languages=[DUTCH],
    _labels=["negative", "positive"],
)

SCALA_NL_CONFIG = DatasetConfig(
    name="scala-nl",
    pretty_name="ScaLA-nl",
    source="EuroEval/scala-nl",
    task=LA,
    languages=[DUTCH],
)

CONLL_NL_CONFIG = DatasetConfig(
    name="conll-nl",
    pretty_name="CoNLL-nl",
    source="EuroEval/conll-nl-mini",
    task=NER,
    languages=[DUTCH],
)

SQUAD_NL_CONFIG = DatasetConfig(
    name="squad-nl",
    pretty_name="SQuAD-nl",
    source="EuroEval/squad-nl-v2-mini",
    task=RC,
    languages=[DUTCH],
)

WIKI_LINGUA_NL_CONFIG = DatasetConfig(
    name="wiki-lingua-nl",
    pretty_name="WikiLingua-nl",
    source="EuroEval/wiki-lingua-nl-mini",
    task=SUMM,
    languages=[DUTCH],
)

MMLU_NL_CONFIG = DatasetConfig(
    name="mmlu-nl",
    pretty_name="MMLU-nl",
    source="EuroEval/mmlu-nl-mini",
    task=KNOW,
    languages=[DUTCH],
)

HELLASWAG_NL_CONFIG = DatasetConfig(
    name="hellaswag-nl",
    pretty_name="HellaSwag-nl",
    source="EuroEval/hellaswag-nl-mini",
    task=COMMON_SENSE,
    languages=[DUTCH],
)

VALEU_NL_CONFIG = DatasetConfig(
    name="valeu-nl",
    pretty_name="VaLEU-nl",
    source="EuroEval/european-values-nl",
    task=EUROPEAN_VALUES,
    languages=[DUTCH],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
)


### Unofficial datasets ###

DUTCH_COLA_CONFIG = DatasetConfig(
    name="dutch-cola",
    pretty_name="Dutch CoLA",
    source="EuroEval/dutch-cola",
    task=LA,
    languages=[DUTCH],
    unofficial=True,
)

DUTCH_COLA_FULL_CONFIG = DatasetConfig(
    name="dutch-cola-full",
    pretty_name="Dutch CoLA Full",
    source="EuroEval/dutch-cola-full",
    task=LA,
    languages=[DUTCH],
    unofficial=True,
)

ARC_NL_CONFIG = DatasetConfig(
    name="arc-nl",
    pretty_name="ARC-nl",
    source="EuroEval/arc-nl-mini",
    task=KNOW,
    languages=[DUTCH],
    unofficial=True,
)

BELEBELE_NL_CONFIG = DatasetConfig(
    name="belebele-nl",
    pretty_name="Belebele-nl",
    source="EuroEval/belebele-nl-mini",
    task=MCRC,
    languages=[DUTCH],
    unofficial=True,
)

MULTI_WIKI_QA_NL_CONFIG = DatasetConfig(
    name="multi-wiki-qa-nl",
    pretty_name="MultiWikiQA-nl",
    source="EuroEval/multi-wiki-qa-nl-mini",
    task=RC,
    languages=[DUTCH],
    unofficial=True,
)

GOLDENSWAG_NL_CONFIG = DatasetConfig(
    name="goldenswag-nl",
    pretty_name="GoldenSwag-nl",
    source="EuroEval/goldenswag-nl-mini",
    task=COMMON_SENSE,
    languages=[DUTCH],
    unofficial=True,
)

WINOGRANDE_NL_CONFIG = DatasetConfig(
    name="winogrande-nl",
    pretty_name="Winogrande-nl",
    source="EuroEval/winogrande-nl",
    task=COMMON_SENSE,
    languages=[DUTCH],
    _labels=["a", "b"],
    unofficial=True,
)
