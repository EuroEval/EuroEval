"""All Italian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import ITALIAN
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

SENTIPOLC_CONFIG = DatasetConfig(
    name="sentipolc16",
    source="EuroEval/sentipolc16-mini",
    task=SENT,
    languages=[ITALIAN],
)

SCALA_IT_CONFIG = DatasetConfig(
    name="scala-it", source="EuroEval/scala-it", task=LA, languages=[ITALIAN]
)

MULTINERD_IT_CONFIG = DatasetConfig(
    name="multinerd-it",
    source="EuroEval/multinerd-mini-it",
    task=NER,
    languages=[ITALIAN],
)

SQUAD_IT_CONFIG = DatasetConfig(
    name="squad-it", source="EuroEval/squad-it-mini", task=RC, languages=[ITALIAN]
)

ILPOST_SUM_CONFIG = DatasetConfig(
    name="ilpost-sum", source="EuroEval/ilpost-sum", task=SUMM, languages=[ITALIAN]
)

MMLU_IT_CONFIG = DatasetConfig(
    name="mmlu-it", source="EuroEval/mmlu-it-mini", task=KNOW, languages=[ITALIAN]
)

HELLASWAG_IT_CONFIG = DatasetConfig(
    name="hellaswag-it",
    source="EuroEval/hellaswag-it-mini",
    task=COMMON_SENSE,
    languages=[ITALIAN],
)

EUROPEAN_VALUES_IT_CONFIG = DatasetConfig(
    name="european-values-it",
    source="EuroEval/european-values-it",
    task=EUROPEAN_VALUES,
    languages=[ITALIAN],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
)


### Unofficial datasets ###

WIKINEURAL_IT_CONFIG = DatasetConfig(
    name="wikineural-it",
    source="EuroEval/wikineural-mini-it",
    task=NER,
    languages=[ITALIAN],
    unofficial=True,
)

BELEBELE_IT_CONFIG = DatasetConfig(
    name="belebele-it",
    source="EuroEval/belebele-it-mini",
    task=MCRC,
    languages=[ITALIAN],
    unofficial=True,
)

MULTI_WIKI_QA_IT_CONFIG = DatasetConfig(
    name="multi-wiki-qa-it",
    source="EuroEval/multi-wiki-qa-it-mini",
    task=RC,
    languages=[ITALIAN],
    unofficial=True,
)

GOLDENSWAG_IT_CONFIG = DatasetConfig(
    name="goldenswag-it",
    source="EuroEval/goldenswag-it-mini",
    task=COMMON_SENSE,
    languages=[ITALIAN],
    unofficial=True,
)

WINOGRANDE_IT_CONFIG = DatasetConfig(
    name="winogrande-it",
    source="EuroEval/winogrande-it",
    task=COMMON_SENSE,
    languages=[ITALIAN],
    _labels=["a", "b"],
    unofficial=True,
)

EUROPEAN_VALUES_SITUATIONAL_IT_CONFIG = DatasetConfig(
    name="european-values-situational-it",
    source="EuroEval/european-values-situational-it",
    task=EUROPEAN_VALUES,
    languages=[ITALIAN],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)

EUROPEAN_VALUES_COMPLETIONS_IT_CONFIG = DatasetConfig(
    name="european-values-completions-it",
    source="EuroEval/european-values-completions-it",
    task=EUROPEAN_VALUES,
    languages=[ITALIAN],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)
