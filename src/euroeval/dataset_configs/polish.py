"""All Polish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import POLISH
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

POLEMO2_CONFIG = DatasetConfig(
    name="polemo2", source="EuroEval/polemo2-mini", task=SENT, languages=[POLISH]
)

SCALA_PL_CONFIG = DatasetConfig(
    name="scala-pl", source="EuroEval/scala-pl", task=LA, languages=[POLISH]
)

KPWR_NER_CONFIG = DatasetConfig(
    name="kpwr-ner", source="EuroEval/kpwr-ner", task=NER, languages=[POLISH]
)

POQUAD_CONFIG = DatasetConfig(
    name="poquad", source="EuroEval/poquad-mini", task=RC, languages=[POLISH]
)

PSC_CONFIG = DatasetConfig(
    name="psc", source="EuroEval/psc-mini", task=SUMM, languages=[POLISH]
)

LLMZSZL_CONFIG = DatasetConfig(
    name="llmzszl", source="EuroEval/llmzszl-mini", task=KNOW, languages=[POLISH]
)

WINOGRANDE_PL_CONFIG = DatasetConfig(
    name="winogrande-pl",
    source="EuroEval/winogrande-pl",
    task=COMMON_SENSE,
    languages=[POLISH],
    _labels=["a", "b"],
)

EUROPEAN_VALUES_PL_CONFIG = DatasetConfig(
    name="european-values-pl",
    source="EuroEval/european-values-pl",
    task=EUROPEAN_VALUES,
    languages=[POLISH],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
)


### Unofficial datasets ###

MULTI_WIKI_QA_PL_CONFIG = DatasetConfig(
    name="multi-wiki-qa-pl",
    source="EuroEval/multi-wiki-qa-pl-mini",
    task=RC,
    languages=[POLISH],
    unofficial=True,
)

GOLDENSWAG_PL_CONFIG = DatasetConfig(
    name="goldenswag-pl",
    source="EuroEval/goldenswag-pl-mini",
    task=COMMON_SENSE,
    languages=[POLISH],
    unofficial=True,
)

EUROPEAN_VALUES_SITUATIONAL_PL_CONFIG = DatasetConfig(
    name="european-values-situational-pl",
    source="EuroEval/european-values-situational-pl",
    task=EUROPEAN_VALUES,
    languages=[POLISH],
    splits=["test"],
    bootstrap_samples=False,
    unofficial=True,
)

EUROPEAN_VALUES_COMPLETIONS_PL_CONFIG = DatasetConfig(
    name="european-values-completions-pl",
    source="EuroEval/european-values-completions-pl",
    task=EUROPEAN_VALUES,
    languages=[POLISH],
    splits=["test"],
    bootstrap_samples=False,
    unofficial=True,
)
