"""All Polish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import PL
from ..tasks import KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

SCALA_PL_CONFIG = DatasetConfig(
    name="scala-pl",
    pretty_name="the Polish part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-pl",
    task=LA,
    languages=[PL],
)

POQUAD_CONFIG = DatasetConfig(
    name="poquad",
    pretty_name="the Polish question answering dataset PoQuAD",
    huggingface_id="EuroEval/poquad-mini",
    task=RC,
    languages=[PL],
)

POLEMO2_CONFIG = DatasetConfig(
    name="polemo2",
    pretty_name="the Polish sentiment analysis dataset PolEmo 2.0",
    huggingface_id="EuroEval/polemo2-mini",
    task=SENT,
    languages=[PL],
)

KPWR_NER_CONFIG = DatasetConfig(
    name="kpwr-ner",
    pretty_name="the Polish entity recognition dataset KPWr-NER",
    huggingface_id="EuroEval/kpwr-ner",
    task=NER,
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
    pretty_name="the Polish multiple choice dataset LLMzSzŁ",
    huggingface_id="EuroEval/llmzszl-mini",
    task=KNOW,
    languages=[PL],
)
