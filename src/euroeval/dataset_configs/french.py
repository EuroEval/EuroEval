"""All French dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import FRENCH
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

ALLOCINE_CONFIG = DatasetConfig(
    name="allocine",
    source="EuroEval/allocine-mini",
    task=SENT,
    languages=[FRENCH],
    _labels=["negative", "positive"],
    _prompt_label_mapping=dict(positive="positif", negative="négatif"),
)

SCALA_FR_CONFIG = DatasetConfig(
    name="scala-fr", source="EuroEval/scala-fr", task=LA, languages=[FRENCH]
)

ELTEC_CONFIG = DatasetConfig(
    name="eltec", source="EuroEval/eltec-mini", task=NER, languages=[FRENCH]
)

FQUAD_CONFIG = DatasetConfig(
    name="fquad", source="EuroEval/fquad-mini", task=RC, languages=[FRENCH]
)

ORANGE_SUM_CONFIG = DatasetConfig(
    name="orange-sum", source="EuroEval/orange-sum-mini", task=SUMM, languages=[FRENCH]
)

MMLU_FR_CONFIG = DatasetConfig(
    name="mmlu-fr", source="EuroEval/mmlu-fr-mini", task=KNOW, languages=[FRENCH]
)

HELLASWAG_FR_CONFIG = DatasetConfig(
    name="hellaswag-fr",
    source="EuroEval/hellaswag-fr-mini",
    task=COMMON_SENSE,
    languages=[FRENCH],
)

EUROPEAN_VALUES_FR_CONFIG = DatasetConfig(
    name="european-values-fr",
    source="EuroEval/european-values-fr",
    task=EUROPEAN_VALUES,
    languages=[FRENCH],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
)


### Unofficial datasets ###

BELEBELE_FR_CONFIG = DatasetConfig(
    name="belebele-fr",
    source="EuroEval/belebele-fr-mini",
    task=MCRC,
    languages=[FRENCH],
    unofficial=True,
)

MULTI_WIKI_QA_FR_CONFIG = DatasetConfig(
    name="multi-wiki-qa-fr",
    source="EuroEval/multi-wiki-qa-fr-mini",
    task=RC,
    languages=[FRENCH],
    unofficial=True,
)

GOLDENSWAG_FR_CONFIG = DatasetConfig(
    name="goldenswag-fr",
    source="EuroEval/goldenswag-fr-mini",
    task=COMMON_SENSE,
    languages=[FRENCH],
    unofficial=True,
)

WINOGRANDE_FR_CONFIG = DatasetConfig(
    name="winogrande-fr",
    source="EuroEval/winogrande-fr",
    task=COMMON_SENSE,
    languages=[FRENCH],
    _labels=["a", "b"],
    unofficial=True,
)

EUROPEAN_VALUES_SITUATIONAL_FR_CONFIG = DatasetConfig(
    name="european-values-situational-fr",
    source="EuroEval/european-values-situational-fr",
    task=EUROPEAN_VALUES,
    languages=[FRENCH],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)

EUROPEAN_VALUES_COMPLETIONS_FR_CONFIG = DatasetConfig(
    name="european-values-completions-fr",
    source="EuroEval/european-values-completions-fr",
    task=EUROPEAN_VALUES,
    languages=[FRENCH],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)
