"""All German dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import GERMAN
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

SB10K_CONFIG = DatasetConfig(
    name="sb10k",
    pretty_name="SB10K",
    source="EuroEval/sb10k-mini",
    task=SENT,
    languages=[GERMAN],
)

SCALA_DE_CONFIG = DatasetConfig(
    name="scala-de",
    pretty_name="ScaLA-de",
    source="EuroEval/scala-de",
    task=LA,
    languages=[GERMAN],
)

GERMEVAL_CONFIG = DatasetConfig(
    name="germeval",
    pretty_name="GermEval",
    source="EuroEval/germeval-mini",
    task=NER,
    languages=[GERMAN],
)

GERMANQUAD_CONFIG = DatasetConfig(
    name="germanquad",
    pretty_name="GermanQuAD",
    source="EuroEval/germanquad-mini",
    task=RC,
    languages=[GERMAN],
)

MLSUM_DE_CONFIG = DatasetConfig(
    name="mlsum-de",
    pretty_name="MLSUM-de",
    source="EuroEval/mlsum-mini",
    task=SUMM,
    languages=[GERMAN],
)

MMLU_DE_CONFIG = DatasetConfig(
    name="mmlu-de",
    pretty_name="MMLU-de",
    source="EuroEval/mmlu-de-mini",
    task=KNOW,
    languages=[GERMAN],
)

HELLASWAG_DE_CONFIG = DatasetConfig(
    name="hellaswag-de",
    pretty_name="HellaSwag-de",
    source="EuroEval/hellaswag-de-mini",
    task=COMMON_SENSE,
    languages=[GERMAN],
)

VALEU_DE_CONFIG = DatasetConfig(
    name="valeu-de",
    pretty_name="VaLEU-de",
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
    pretty_name="XQuAD-de",
    source="EuroEval/xquad-de",
    task=RC,
    languages=[GERMAN],
    unofficial=True,
)

ARC_DE_CONFIG = DatasetConfig(
    name="arc-de",
    pretty_name="ARC-de",
    source="EuroEval/arc-de-mini",
    task=KNOW,
    languages=[GERMAN],
    unofficial=True,
)

BELEBELE_DE_CONFIG = DatasetConfig(
    name="belebele-de",
    pretty_name="Belebele-de",
    source="EuroEval/belebele-de-mini",
    task=MCRC,
    languages=[GERMAN],
    unofficial=True,
)

MULTI_WIKI_QA_DE_CONFIG = DatasetConfig(
    name="multi-wiki-qa-de",
    pretty_name="MultiWikiQA-de",
    source="EuroEval/multi-wiki-qa-de-mini",
    task=RC,
    languages=[GERMAN],
    unofficial=True,
)

GOLDENSWAG_DE_CONFIG = DatasetConfig(
    name="goldenswag-de",
    pretty_name="GoldenSwag-de",
    source="EuroEval/goldenswag-de-mini",
    task=COMMON_SENSE,
    languages=[GERMAN],
    unofficial=True,
)

WINOGRANDE_DE_CONFIG = DatasetConfig(
    name="winogrande-de",
    pretty_name="Winogrande-de",
    source="EuroEval/winogrande-de",
    task=COMMON_SENSE,
    languages=[GERMAN],
    _labels=["a", "b"],
    unofficial=True,
)
