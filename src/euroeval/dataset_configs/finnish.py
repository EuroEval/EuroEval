"""All Finnish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import FINNISH
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

SCANDISENT_FI_CONFIG = DatasetConfig(
    name="scandisent-fi",
    pretty_name="ScandiSent-fi",
    source="EuroEval/scandisent-fi-mini",
    task=SENT,
    languages=[FINNISH],
    _labels=["negative", "positive"],
)

TURKU_NER_FI_CONFIG = DatasetConfig(
    name="turku-ner-fi",
    pretty_name="Turku NER-fi",
    source="EuroEval/turku-ner-fi-mini",
    task=NER,
    languages=[FINNISH],
)

TYDIQA_FI_CONFIG = DatasetConfig(
    name="tydiqa-fi",
    pretty_name="TyDiQA-fi",
    source="EuroEval/tydiqa-fi-mini",
    task=RC,
    languages=[FINNISH],
)

XLSUM_FI_CONFIG = DatasetConfig(
    name="xlsum-fi",
    pretty_name="XLSum-fi",
    source="EuroEval/xlsum-fi-mini",
    task=SUMM,
    languages=[FINNISH],
)

HELLASWAG_FI_CONFIG = DatasetConfig(
    name="hellaswag-fi",
    pretty_name="HellaSwag-fi",
    source="EuroEval/hellaswag-fi-mini",
    task=COMMON_SENSE,
    languages=[FINNISH],
)

SCALA_FI_CONFIG = DatasetConfig(
    name="scala-fi",
    pretty_name="ScaLA-fi",
    source="EuroEval/scala-fi",
    task=LA,
    languages=[FINNISH],
)

VALEU_FI_CONFIG = DatasetConfig(
    name="valeu-fi",
    pretty_name="VaLEU-fi",
    source="EuroEval/european-values-fi",
    task=EUROPEAN_VALUES,
    languages=[FINNISH],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
)


### Unofficial datasets ###

BELEBELE_FI_CONFIG = DatasetConfig(
    name="belebele-fi",
    pretty_name="Belebele-fi",
    source="EuroEval/belebele-fi-mini",
    task=MCRC,
    languages=[FINNISH],
    unofficial=True,
)

MULTI_WIKI_QA_FI_CONFIG = DatasetConfig(
    name="multi-wiki-qa-fi",
    pretty_name="MultiWikiQA-fi",
    source="EuroEval/multi-wiki-qa-fi-mini",
    task=RC,
    languages=[FINNISH],
    unofficial=True,
)

GOLDENSWAG_FI_CONFIG = DatasetConfig(
    name="goldenswag-fi",
    pretty_name="GoldenSwag-fi",
    source="EuroEval/goldenswag-fi-mini",
    task=COMMON_SENSE,
    languages=[FINNISH],
    unofficial=True,
)

WINOGRANDE_FI_CONFIG = DatasetConfig(
    name="winogrande-fi",
    pretty_name="Winogrande-fi",
    source="EuroEval/winogrande-fi",
    task=COMMON_SENSE,
    languages=[FINNISH],
    _labels=["a", "b"],
    unofficial=True,
)
