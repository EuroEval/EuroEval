"""All Swedish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import SWEDISH
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

SWEREC_CONFIG = DatasetConfig(
    name="swerec", source="EuroEval/swerec-mini", task=SENT, languages=[SWEDISH]
)

SCALA_SV_CONFIG = DatasetConfig(
    name="scala-sv", source="EuroEval/scala-sv", task=LA, languages=[SWEDISH]
)

SUC3_CONFIG = DatasetConfig(
    name="suc3", source="EuroEval/suc3-mini", task=NER, languages=[SWEDISH]
)

MULTI_WIKI_QA_SV_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sv",
    source="EuroEval/multi-wiki-qa-sv-mini",
    task=RC,
    languages=[SWEDISH],
)

SWEDN_CONFIG = DatasetConfig(
    name="swedn", source="EuroEval/swedn-mini", task=SUMM, languages=[SWEDISH]
)

MMLU_SV_CONFIG = DatasetConfig(
    name="mmlu-sv", source="EuroEval/mmlu-sv-mini", task=KNOW, languages=[SWEDISH]
)

HELLASWAG_SV_CONFIG = DatasetConfig(
    name="hellaswag-sv",
    source="EuroEval/hellaswag-sv-mini",
    task=COMMON_SENSE,
    languages=[SWEDISH],
)

EUROPEAN_VALUES_SV_CONFIG = DatasetConfig(
    name="european-values-sv",
    source="EuroEval/european-values-sv",
    task=EUROPEAN_VALUES,
    languages=[SWEDISH],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
)


### Unofficial datasets ###

SCHIBSTED_SV_CONFIG = DatasetConfig(
    name="schibsted-sv",
    source="EuroEval/schibsted-article-summaries-sv",
    task=SUMM,
    languages=[SWEDISH],
    unofficial=True,
)

ARC_SV_CONFIG = DatasetConfig(
    name="arc-sv",
    source="EuroEval/arc-sv-mini",
    task=KNOW,
    languages=[SWEDISH],
    unofficial=True,
)

BELEBELE_SV_CONFIG = DatasetConfig(
    name="belebele-sv",
    source="EuroEval/belebele-sv-mini",
    task=MCRC,
    languages=[SWEDISH],
    unofficial=True,
)

SCANDIQA_SV_CONFIG = DatasetConfig(
    name="scandiqa-sv",
    source="EuroEval/scandiqa-sv-mini",
    task=RC,
    languages=[SWEDISH],
    unofficial=True,
)

GOLDENSWAG_SV_CONFIG = DatasetConfig(
    name="goldenswag-sv",
    source="EuroEval/goldenswag-sv-mini",
    task=COMMON_SENSE,
    languages=[SWEDISH],
    unofficial=True,
)

WINOGRANDE_SV_CONFIG = DatasetConfig(
    name="winogrande-sv",
    source="EuroEval/winogrande-sv",
    task=COMMON_SENSE,
    languages=[SWEDISH],
    _labels=["a", "b"],
    unofficial=True,
)

EUROPEAN_VALUES_SITUATIONAL_SV_CONFIG = DatasetConfig(
    name="european-values-situational-sv",
    source="EuroEval/european-values-situational-sv",
    task=EUROPEAN_VALUES,
    languages=[SWEDISH],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)

EUROPEAN_VALUES_COMPLETIONS_SV_CONFIG = DatasetConfig(
    name="european-values-completions-sv",
    source="EuroEval/european-values-completions-sv",
    task=EUROPEAN_VALUES,
    languages=[SWEDISH],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)

SKOLPROV_CONFIG = DatasetConfig(
    name="skolprov",
    source="EuroEval/skolprov",
    task=KNOW,
    languages=[SWEDISH],
    unofficial=True,
)
