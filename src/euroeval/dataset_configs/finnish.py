"""All Finnish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import FI
from ..tasks import COMMON_SENSE, NER, RC, SENT, SUMM

### Official datasets ###

SCANDISENT_FI_CONFIG = DatasetConfig(
    name="scandisent-fi",
    pretty_name="the truncated version of the Finnish part of the binary sentiment "
    "classification dataset ScandiSent",
    huggingface_id="EuroEval/scandisent-fi-mini",
    task=SENT,
    languages=[FI],
)

TURKU_NER_FI_CONFIG = DatasetConfig(
    name="turku-ner-fi",
    pretty_name="the Finnish part of the named entity recognition dataset Turku NER",
    huggingface_id="EuroEval/turku-ner-fi-mini",
    task=NER,
    languages=[FI],
)

TYDIQA_FI_CONFIG = DatasetConfig(
    name="tydiqa-fi",
    pretty_name="the Finnish part of the TydiQA reading comprehension dataset",
    huggingface_id="EuroEval/tydiqa-fi-mini",
    task=RC,
    languages=[FI],
)

XLSUM_FI_CONFIG = DatasetConfig(
    name="xlsum-fi",
    pretty_name="the Finnish summarisation dataset XL-Sum",
    huggingface_id="EuroEval/xlsum-fi-mini",
    task=SUMM,
    languages=[FI],
)


HELLASWAG_FI_CONFIG = DatasetConfig(
    name="hellaswag-fi",
    pretty_name="the truncated version of the Finnish common-sense reasoning dataset "
    "HellaSwag-fi, translated from the English HellaSwag dataset",
    huggingface_id="EuroEval/hellaswag-fi-mini",
    task=COMMON_SENSE,
    languages=[FI],
)

### Unofficial datasets ###
