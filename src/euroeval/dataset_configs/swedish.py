"""All Swedish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import SWEDISH
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

SWEREC_CONFIG = DatasetConfig(
    name="swerec",
    pretty_name="SweReC",
    source="EuroEval/swerec-mini",
    task=SENT,
    languages=[SWEDISH],
)

SCALA_SV_CONFIG = DatasetConfig(
    name="scala-sv",
    pretty_name="ScaLA-sv",
    source="EuroEval/scala-sv",
    task=LA,
    languages=[SWEDISH],
)

SUC3_CONFIG = DatasetConfig(
    name="suc3",
    pretty_name="SUC3",
    source="EuroEval/suc3-mini",
    task=NER,
    languages=[SWEDISH],
)

MULTI_WIKI_QA_SV_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sv",
    pretty_name="MultiWikiQA-sv",
    source="EuroEval/multi-wiki-qa-sv-mini",
    task=RC,
    languages=[SWEDISH],
)

SWEDN_CONFIG = DatasetConfig(
    name="swedn",
    pretty_name="SweDN",
    source="EuroEval/swedn-mini",
    task=SUMM,
    languages=[SWEDISH],
)

MMLU_SV_CONFIG = DatasetConfig(
    name="mmlu-sv",
    pretty_name="MMLU-sv",
    source="EuroEval/mmlu-sv-mini",
    task=KNOW,
    languages=[SWEDISH],
)

HELLASWAG_SV_CONFIG = DatasetConfig(
    name="hellaswag-sv",
    pretty_name="HellaSwag-sv",
    source="EuroEval/hellaswag-sv-mini",
    task=COMMON_SENSE,
    languages=[SWEDISH],
)

VALEU_SV_CONFIG = DatasetConfig(
    name="valeu-sv",
    pretty_name="VaLEU-sv",
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
    pretty_name="Schibsted-sv",
    source="EuroEval/schibsted-article-summaries-sv",
    task=SUMM,
    languages=[SWEDISH],
    unofficial=True,
)

ARC_SV_CONFIG = DatasetConfig(
    name="arc-sv",
    pretty_name="ARC-sv",
    source="EuroEval/arc-sv-mini",
    task=KNOW,
    languages=[SWEDISH],
    unofficial=True,
)

BELEBELE_SV_CONFIG = DatasetConfig(
    name="belebele-sv",
    pretty_name="Belebele-sv",
    source="EuroEval/belebele-sv-mini",
    task=MCRC,
    languages=[SWEDISH],
    unofficial=True,
)

SCANDIQA_SV_CONFIG = DatasetConfig(
    name="scandiqa-sv",
    pretty_name="ScandiQA-sv",
    source="EuroEval/scandiqa-sv-mini",
    task=RC,
    languages=[SWEDISH],
    unofficial=True,
)

GOLDENSWAG_SV_CONFIG = DatasetConfig(
    name="goldenswag-sv",
    pretty_name="GoldenSwag-sv",
    source="EuroEval/goldenswag-sv-mini",
    task=COMMON_SENSE,
    languages=[SWEDISH],
    unofficial=True,
)

WINOGRANDE_SV_CONFIG = DatasetConfig(
    name="winogrande-sv",
    pretty_name="Winogrande-sv",
    source="EuroEval/winogrande-sv",
    task=COMMON_SENSE,
    languages=[SWEDISH],
    _labels=["a", "b"],
    unofficial=True,
)

SKOLPROV_CONFIG = DatasetConfig(
    name="skolprov",
    pretty_name="Skolprov",
    source="EuroEval/skolprov",
    task=KNOW,
    languages=[SWEDISH],
    unofficial=True,
)
