"""All French dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import FRENCH
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

ALLOCINE_CONFIG = DatasetConfig(
    name="allocine",
    pretty_name="AlloCiné",
    source="EuroEval/allocine-mini",
    task=SENT,
    languages=[FRENCH],
    _labels=["negative", "positive"],
    _prompt_label_mapping=dict(positive="positif", negative="négatif"),
)

SCALA_FR_CONFIG = DatasetConfig(
    name="scala-fr",
    pretty_name="ScaLA-fr",
    source="EuroEval/scala-fr",
    task=LA,
    languages=[FRENCH],
)

ELTEC_CONFIG = DatasetConfig(
    name="eltec",
    pretty_name="ELTeC",
    source="EuroEval/eltec-mini",
    task=NER,
    languages=[FRENCH],
)

FQUAD_CONFIG = DatasetConfig(
    name="fquad",
    pretty_name="FQuAD",
    source="EuroEval/fquad-mini",
    task=RC,
    languages=[FRENCH],
)

ORANGE_SUM_CONFIG = DatasetConfig(
    name="orange-sum",
    pretty_name="OrangeSum",
    source="EuroEval/orange-sum-mini",
    task=SUMM,
    languages=[FRENCH],
)

MMLU_FR_CONFIG = DatasetConfig(
    name="mmlu-fr",
    pretty_name="MMLU-fr",
    source="EuroEval/mmlu-fr-mini",
    task=KNOW,
    languages=[FRENCH],
)

HELLASWAG_FR_CONFIG = DatasetConfig(
    name="hellaswag-fr",
    pretty_name="HellaSwag-fr",
    source="EuroEval/hellaswag-fr-mini",
    task=COMMON_SENSE,
    languages=[FRENCH],
)

VALEU_FR_CONFIG = DatasetConfig(
    name="valeu-fr",
    pretty_name="VaLEU-fr",
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
    pretty_name="Belebele-fr",
    source="EuroEval/belebele-fr-mini",
    task=MCRC,
    languages=[FRENCH],
    unofficial=True,
)

MULTI_WIKI_QA_FR_CONFIG = DatasetConfig(
    name="multi-wiki-qa-fr",
    pretty_name="MultiWikiQA-fr",
    source="EuroEval/multi-wiki-qa-fr-mini",
    task=RC,
    languages=[FRENCH],
    unofficial=True,
)

GOLDENSWAG_FR_CONFIG = DatasetConfig(
    name="goldenswag-fr",
    pretty_name="GoldenSwag-fr",
    source="EuroEval/goldenswag-fr-mini",
    task=COMMON_SENSE,
    languages=[FRENCH],
    unofficial=True,
)

WINOGRANDE_FR_CONFIG = DatasetConfig(
    name="winogrande-fr",
    pretty_name="Winogrande-fr",
    source="EuroEval/winogrande-fr",
    task=COMMON_SENSE,
    languages=[FRENCH],
    _labels=["a", "b"],
    unofficial=True,
)
