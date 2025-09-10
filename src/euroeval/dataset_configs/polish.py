"""All Polish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..enums import ModelType
from ..languages import PL
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

POLEMO2_CONFIG = DatasetConfig(
    name="polemo2",
    pretty_name="the Polish sentiment classification dataset PolEmo2",
    huggingface_id="EuroEval/polemo2-mini",
    task=SENT,
    languages=[PL],
)

SCALA_PL_CONFIG = DatasetConfig(
    name="scala-pl",
    pretty_name="the Polish part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-pl",
    task=LA,
    languages=[PL],
)

KPWR_NER_CONFIG = DatasetConfig(
    name="kpwr-ner",
    pretty_name="the Polish entity recognition dataset KPWr-NER",
    huggingface_id="EuroEval/kpwr-ner",
    task=NER,
    languages=[PL],
)

POQUAD_CONFIG = DatasetConfig(
    name="poquad",
    pretty_name="the Polish question answering dataset PoQuAD",
    huggingface_id="EuroEval/poquad-mini",
    task=RC,
    languages=[PL],
)

PSC_CONFIG = DatasetConfig(
    name="psc",
    pretty_name="the Polish summarisation dataset PSC",
    huggingface_id="EuroEval/psc-mini",
    task=SUMM,
    languages=[PL],
)

LLMZSZL_CONFIG = DatasetConfig(
    name="llmzszl",
    pretty_name="the Polish knowledge dataset LLMzSzŁ",
    huggingface_id="EuroEval/llmzszl-mini",
    task=KNOW,
    languages=[PL],
)

WINOGRANDE_PL_CONFIG = DatasetConfig(
    name="winogrande-pl",
    pretty_name="the Polish common-sense reasoning dataset Winogrande-pl, translated "
    "from the English Winogrande dataset",
    huggingface_id="EuroEval/winogrande-pl",
    task=COMMON_SENSE,
    languages=[PL],
    splits=["train", "test"],
    _labels=["a", "b"],
    _allowed_model_types=[ModelType.GENERATIVE],
)

EUROPEAN_VALUES_PL_CONFIG = DatasetConfig(
    name="european-values-pl",
    pretty_name="the Polish version of the European values evaluation dataset",
    huggingface_id="EuroEval/european-values-pl",
    task=EUROPEAN_VALUES,
    languages=[PL],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
)


### Unofficial datasets ###

MULTI_WIKI_QA_PL_CONFIG = DatasetConfig(
    name="multi-wiki-qa-pl",
    pretty_name="the truncated version of the Polish part of the reading "
    "comprehension dataset MultiWikiQA",
    huggingface_id="EuroEval/multi-wiki-qa-pl-mini",
    task=RC,
    languages=[PL],
    unofficial=True,
)

GOLDENSWAG_PL_CONFIG = DatasetConfig(
    name="goldenswag-pl",
    pretty_name="the truncated version of the Polish common-sense reasoning "
    "dataset GoldenSwag-pl, translated from the English GoldenSwag dataset",
    huggingface_id="EuroEval/goldenswag-pl-mini",
    task=COMMON_SENSE,
    languages=[PL],
    unofficial=True,
)

EUROPEAN_VALUES_SITUATIONAL_PL_CONFIG = DatasetConfig(
    name="european-values-situational-pl",
    pretty_name="the Polish version of the European values evaluation dataset, where "
    "the questions are phrased in a situational way",
    huggingface_id="EuroEval/european-values-situational-pl",
    task=EUROPEAN_VALUES,
    languages=[PL],
    splits=["test"],
    bootstrap_samples=False,
    unofficial=True,
)

EUROPEAN_VALUES_COMPLETIONS_PL_CONFIG = DatasetConfig(
    name="european-values-completions-pl",
    pretty_name="the Polish version of the European values evaluation dataset, where "
    "the questions are phrased as sentence completions",
    huggingface_id="EuroEval/european-values-completions-pl",
    task=EUROPEAN_VALUES,
    languages=[PL],
    splits=["test"],
    bootstrap_samples=False,
    unofficial=True,
)
