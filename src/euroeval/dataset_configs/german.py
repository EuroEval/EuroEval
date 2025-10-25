"""All German dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import GERMAN
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

SB10K_CONFIG = DatasetConfig(
    name="sb10k", source="EuroEval/sb10k-mini", task=SENT, languages=[GERMAN]
)

SCALA_DE_CONFIG = DatasetConfig(
    name="scala-de", source="EuroEval/scala-de", task=LA, languages=[GERMAN]
)

GERMEVAL_CONFIG = DatasetConfig(
    name="germeval", source="EuroEval/germeval-mini", task=NER, languages=[GERMAN]
)

GERMANQUAD_CONFIG = DatasetConfig(
    name="germanquad", source="EuroEval/germanquad-mini", task=RC, languages=[GERMAN]
)

MLSUM_DE_CONFIG = DatasetConfig(
    name="mlsum-de", source="EuroEval/mlsum-mini", task=SUMM, languages=[GERMAN]
)

MMLU_DE_CONFIG = DatasetConfig(
    name="mmlu-de", source="EuroEval/mmlu-de-mini", task=KNOW, languages=[GERMAN]
)

HELLASWAG_DE_CONFIG = DatasetConfig(
    name="hellaswag-de",
    source="EuroEval/hellaswag-de-mini",
    task=COMMON_SENSE,
    languages=[GERMAN],
)

EUROPEAN_VALUES_DE_CONFIG = DatasetConfig(
    name="european-values-de",
    source="EuroEval/european-values-de",
    task=EUROPEAN_VALUES,
    languages=[GERMAN],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
)


### Unofficial datasets ###

XQUAD_DE_CONFIG = DatasetConfig(
    name="xquad-de",
    source="EuroEval/xquad-de",
    task=RC,
    languages=[GERMAN],
    unofficial=True,
)

ARC_DE_CONFIG = DatasetConfig(
    name="arc-de",
    source="EuroEval/arc-de-mini",
    task=KNOW,
    languages=[GERMAN],
    unofficial=True,
)

BELEBELE_DE_CONFIG = DatasetConfig(
    name="belebele-de",
    source="EuroEval/belebele-de-mini",
    task=MCRC,
    languages=[GERMAN],
    unofficial=True,
)

MULTI_WIKI_QA_DE_CONFIG = DatasetConfig(
    name="multi-wiki-qa-de",
    source="EuroEval/multi-wiki-qa-de-mini",
    task=RC,
    languages=[GERMAN],
    unofficial=True,
)

GOLDENSWAG_DE_CONFIG = DatasetConfig(
    name="goldenswag-de",
    source="EuroEval/goldenswag-de-mini",
    task=COMMON_SENSE,
    languages=[GERMAN],
    unofficial=True,
)

WINOGRANDE_DE_CONFIG = DatasetConfig(
    name="winogrande-de",
    source="EuroEval/winogrande-de",
    task=COMMON_SENSE,
    languages=[GERMAN],
    _labels=["a", "b"],
    unofficial=True,
)

EUROPEAN_VALUES_SITUATIONAL_DE_CONFIG = DatasetConfig(
    name="european-values-situational-de",
    source="EuroEval/european-values-situational-de",
    task=EUROPEAN_VALUES,
    languages=[GERMAN],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)

EUROPEAN_VALUES_COMPLETIONS_DE_CONFIG = DatasetConfig(
    name="european-values-completions-de",
    source="EuroEval/european-values-completions-de",
    task=EUROPEAN_VALUES,
    languages=[GERMAN],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)
