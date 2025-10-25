"""All English dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import ENGLISH
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

SST5_CONFIG = DatasetConfig(
    name="sst5", source="EuroEval/sst5-mini", task=SENT, languages=[ENGLISH]
)

SCALA_EN_CONFIG = DatasetConfig(
    name="scala-en", source="EuroEval/scala-en", task=LA, languages=[ENGLISH]
)

CONLL_EN_CONFIG = DatasetConfig(
    name="conll-en", source="EuroEval/conll-en-mini", task=NER, languages=[ENGLISH]
)

SQUAD_CONFIG = DatasetConfig(
    name="squad", source="EuroEval/squad-mini", task=RC, languages=[ENGLISH]
)

CNN_DAILYMAIL_CONFIG = DatasetConfig(
    name="cnn-dailymail",
    source="EuroEval/cnn-dailymail-mini",
    task=SUMM,
    languages=[ENGLISH],
)

LIFE_IN_THE_UK_CONFIG = DatasetConfig(
    name="life-in-the-uk",
    source="EuroEval/life-in-the-uk",
    task=KNOW,
    languages=[ENGLISH],
)

HELLASWAG_CONFIG = DatasetConfig(
    name="hellaswag",
    source="EuroEval/hellaswag-mini",
    task=COMMON_SENSE,
    languages=[ENGLISH],
)

EUROPEAN_VALUES_EN_CONFIG = DatasetConfig(
    name="european-values-en",
    source="EuroEval/european-values-en",
    task=EUROPEAN_VALUES,
    languages=[ENGLISH],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
)


### Unofficial datasets ###

XQUAD_EN_CONFIG = DatasetConfig(
    name="xquad-en",
    source="EuroEval/xquad-en",
    task=RC,
    languages=[ENGLISH],
    unofficial=True,
)

ARC_CONFIG = DatasetConfig(
    name="arc",
    source="EuroEval/arc-mini",
    task=KNOW,
    languages=[ENGLISH],
    unofficial=True,
)

BELEBELE_CONFIG = DatasetConfig(
    name="belebele-en",
    source="EuroEval/belebele-mini",
    task=MCRC,
    languages=[ENGLISH],
    unofficial=True,
)

MMLU_CONFIG = DatasetConfig(
    name="mmlu",
    source="EuroEval/mmlu-mini",
    task=KNOW,
    languages=[ENGLISH],
    unofficial=True,
)

MULTI_WIKI_QA_EN_CONFIG = DatasetConfig(
    name="multi-wiki-qa-en",
    source="EuroEval/multi-wiki-qa-en-mini",
    task=RC,
    languages=[ENGLISH],
    unofficial=True,
)

WINOGRANDE_CONFIG = DatasetConfig(
    name="winogrande",
    source="EuroEval/winogrande-en",
    task=COMMON_SENSE,
    languages=[ENGLISH],
    _labels=["a", "b"],
    unofficial=True,
)

EUROPEAN_VALUES_SITUATIONAL_EN_CONFIG = DatasetConfig(
    name="european-values-situational-en",
    source="EuroEval/european-values-situational-en",
    task=EUROPEAN_VALUES,
    languages=[ENGLISH],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)

EUROPEAN_VALUES_COMPLETIONS_EN_CONFIG = DatasetConfig(
    name="european-values-completions-en",
    source="EuroEval/european-values-completions-en",
    task=EUROPEAN_VALUES,
    languages=[ENGLISH],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)
