"""All Portuguese dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import PT
from ..tasks import COMMON_SENSE, KNOW, LA, MCRC, NER, SENT, SUMM

### Official datasets ###

PREFIX = "duarteocarmo"

SST2_PT_CONFIG = DatasetConfig(
    name="sst2-pt",
    pretty_name="extraglue-sst2-pt dataset for portuguese (PT)",
    huggingface_id=f"{PREFIX}/extraglue-sst2-pt",
    task=SENT,
    languages=[PT],
    unofficial=False,
    _labels=["positive", "negative"],
)

BOOLQ_PT_CONFIG = DatasetConfig(
    name="boolq-pt",
    pretty_name="extraglue-boolq-pt dataset for portuguese (PT)",
    huggingface_id=f"{PREFIX}/extraglue-boolq-pt",
    task=MCRC,
    languages=[PT],
    unofficial=False,
)


MMLU_PT_CONFIG = DatasetConfig(
    name="mmlu-pt",
    pretty_name="Portuguese version of mmlu mini",
    huggingface_id=f"{PREFIX}/mmlu-pt-mini",
    task=KNOW,
    languages=[PT],
    unofficial=False,
)


HELLASWAG_PT_CONFIG = DatasetConfig(
    name="hellaswag-pt",
    pretty_name="Portuguese version of HellaSwag mini",
    huggingface_id=f"{PREFIX}/hellaswag-pt-mini",
    task=COMMON_SENSE,
    languages=[PT],
    unofficial=False,
)


SCALA_PT = DatasetConfig(
    name="scala-pt",
    pretty_name="the Portuguese part of the linguistic acceptability dataset ScaLA",
    huggingface_id=f"{PREFIX}/scala-pt",
    task=LA,
    languages=[PT],
    unofficial=False,
)

HAREM_PT_CONFIG = DatasetConfig(
    name="harem-pt",
    pretty_name="the harem dataset for NER in portuguese",
    huggingface_id=f"{PREFIX}/harem-pt-mini",
    task=NER,
    languages=[PT],
    unofficial=False,
)

PUBLICO_PT_CONFIG = DatasetConfig(
    name="publico-pt",
    pretty_name="a summarisation dataset for portuguese based on"
    " articles from publico.pt",
    huggingface_id=f"{PREFIX}/sum-pt-publico",
    task=SUMM,
    languages=[PT],
)
