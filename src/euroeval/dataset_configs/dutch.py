"""All Dutch dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import DUTCH
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

DBRD_CONFIG = DatasetConfig(
    name="dbrd",
    source="EuroEval/dbrd-mini",
    task=SENT,
    languages=[DUTCH],
    _labels=["negative", "positive"],
)

SCALA_NL_CONFIG = DatasetConfig(
    name="scala-nl", source="EuroEval/scala-nl", task=LA, languages=[DUTCH]
)

CONLL_NL_CONFIG = DatasetConfig(
    name="conll-nl", source="EuroEval/conll-nl-mini", task=NER, languages=[DUTCH]
)

SQUAD_NL_CONFIG = DatasetConfig(
    name="squad-nl", source="EuroEval/squad-nl-v2-mini", task=RC, languages=[DUTCH]
)

WIKI_LINGUA_NL_CONFIG = DatasetConfig(
    name="wiki-lingua-nl",
    source="EuroEval/wiki-lingua-nl-mini",
    task=SUMM,
    languages=[DUTCH],
)

MMLU_NL_CONFIG = DatasetConfig(
    name="mmlu-nl", source="EuroEval/mmlu-nl-mini", task=KNOW, languages=[DUTCH]
)

HELLASWAG_NL_CONFIG = DatasetConfig(
    name="hellaswag-nl",
    source="EuroEval/hellaswag-nl-mini",
    task=COMMON_SENSE,
    languages=[DUTCH],
)

EUROPEAN_VALUES_NL_CONFIG = DatasetConfig(
    name="european-values-nl",
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
    source="EuroEval/dutch-cola",
    task=LA,
    languages=[DUTCH],
    unofficial=True,
)

DUTCH_COLA_FULL_CONFIG = DatasetConfig(
    name="dutch-cola-full",
    source="EuroEval/dutch-cola-full",
    task=LA,
    languages=[DUTCH],
    unofficial=True,
)

ARC_NL_CONFIG = DatasetConfig(
    name="arc-nl",
    source="EuroEval/arc-nl-mini",
    task=KNOW,
    languages=[DUTCH],
    unofficial=True,
)

BELEBELE_NL_CONFIG = DatasetConfig(
    name="belebele-nl",
    source="EuroEval/belebele-nl-mini",
    task=MCRC,
    languages=[DUTCH],
    unofficial=True,
)

MULTI_WIKI_QA_NL_CONFIG = DatasetConfig(
    name="multi-wiki-qa-nl",
    source="EuroEval/multi-wiki-qa-nl-mini",
    task=RC,
    languages=[DUTCH],
    unofficial=True,
)

GOLDENSWAG_NL_CONFIG = DatasetConfig(
    name="goldenswag-nl",
    source="EuroEval/goldenswag-nl-mini",
    task=COMMON_SENSE,
    languages=[DUTCH],
    unofficial=True,
)

WINOGRANDE_NL_CONFIG = DatasetConfig(
    name="winogrande-nl",
    source="EuroEval/winogrande-nl",
    task=COMMON_SENSE,
    languages=[DUTCH],
    _labels=["a", "b"],
    unofficial=True,
)

EUROPEAN_VALUES_SITUATIONAL_NL_CONFIG = DatasetConfig(
    name="european-values-situational-nl",
    source="EuroEval/european-values-situational-nl",
    task=EUROPEAN_VALUES,
    languages=[DUTCH],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)

EUROPEAN_VALUES_COMPLETIONS_NL_CONFIG = DatasetConfig(
    name="european-values-completions-nl",
    source="EuroEval/european-values-completions-nl",
    task=EUROPEAN_VALUES,
    languages=[DUTCH],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)
