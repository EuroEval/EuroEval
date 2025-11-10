"""All Polish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import POLISH
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

POLEMO2_CONFIG = DatasetConfig(
    name="polemo2",
    pretty_name="Polemo2",
    source="EuroEval/polemo2-mini",
    task=SENT,
    languages=[POLISH],
)

SCALA_PL_CONFIG = DatasetConfig(
    name="scala-pl",
    pretty_name="ScaLA-pl",
    source="EuroEval/scala-pl",
    task=LA,
    languages=[POLISH],
)

KPWR_NER_CONFIG = DatasetConfig(
    name="kpwr-ner",
    pretty_name="KPWr-NER",
    source="EuroEval/kpwr-ner",
    task=NER,
    languages=[POLISH],
)

POQUAD_CONFIG = DatasetConfig(
    name="poquad",
    pretty_name="PoQuAD",
    source="EuroEval/poquad-mini",
    task=RC,
    languages=[POLISH],
)

PSC_CONFIG = DatasetConfig(
    name="psc",
    pretty_name="PSC",
    source="EuroEval/psc-mini",
    task=SUMM,
    languages=[POLISH],
)

LLMZSZL_CONFIG = DatasetConfig(
    name="llmzszl",
    pretty_name="LLMzSzŁ",
    source="EuroEval/llmzszl-mini",
    task=KNOW,
    languages=[POLISH],
)

WINOGRANDE_PL_CONFIG = DatasetConfig(
    name="winogrande-pl",
    pretty_name="Winogrande-pl",
    source="EuroEval/winogrande-pl",
    task=COMMON_SENSE,
    languages=[POLISH],
    _labels=["a", "b"],
)

VALEU_PL_CONFIG = DatasetConfig(
    name="valeu-pl",
    pretty_name="VaLEU-pl",
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
    pretty_name="MultiWikiQA-pl",
    source="EuroEval/multi-wiki-qa-pl-mini",
    task=RC,
    languages=[POLISH],
    unofficial=True,
)

GOLDENSWAG_PL_CONFIG = DatasetConfig(
    name="goldenswag-pl",
    pretty_name="GoldenSwag-pl",
    source="EuroEval/goldenswag-pl-mini",
    task=COMMON_SENSE,
    languages=[POLISH],
    unofficial=True,
)
