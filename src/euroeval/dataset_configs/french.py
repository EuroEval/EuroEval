"""All French dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import FR
from ..tasks import COMMON_SENSE, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

ALLOCINE_CONFIG = DatasetConfig(
    name="allocine",
    pretty_name="the truncated version of the French sentiment classification "
    "dataset AlloCiné",
    huggingface_id="EuroEval/allocine-mini",
    task=SENT,
    languages=[FR],
    _labels=["negative", "positive"],
    _prompt_label_mapping=dict(positive="positif", negative="négatif"),
)

SCALA_FR_CONFIG = DatasetConfig(
    name="scala-fr",
    pretty_name="the French part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-fr",
    task=LA,
    languages=[FR],
)

ELTEC_CONFIG = DatasetConfig(
    name="eltec",
    pretty_name="the truncated version of the French named entity recognition "
    "dataset ELTeC",
    huggingface_id="EuroEval/eltec-mini",
    task=NER,
    languages=[FR],
)

FQUAD_CONFIG = DatasetConfig(
    name="fquad",
    pretty_name="the truncated version of the French reading comprehension dataset "
    "FQuAD",
    huggingface_id="EuroEval/fquad-mini",
    task=RC,
    languages=[FR],
)

ORANGE_SUM_CONFIG = DatasetConfig(
    name="orange-sum",
    pretty_name="the truncated version of the French summarisation dataset OrangeSum",
    huggingface_id="EuroEval/orange-sum-mini",
    task=SUMM,
    languages=[FR],
)

MMLU_FR_CONFIG = DatasetConfig(
    name="mmlu-fr",
    pretty_name="the truncated version of the French knowledge dataset MMLU-fr, "
    "translated from the English MMLU dataset",
    huggingface_id="EuroEval/mmlu-fr-mini",
    task=KNOW,
    languages=[FR],
)

HELLASWAG_FR_CONFIG = DatasetConfig(
    name="hellaswag-fr",
    pretty_name="the truncated version of the French common-sense reasoning dataset "
    "HellaSwag-fr, translated from the English HellaSwag dataset",
    huggingface_id="EuroEval/hellaswag-fr-mini",
    task=COMMON_SENSE,
    languages=[FR],
)


### Unofficial datasets ###

BELEBELE_FR_CONFIG = DatasetConfig(
    name="belebele-fr",
    pretty_name="the French multiple choice reading comprehension dataset BeleBele-fr, "
    "translated from the English BeleBele dataset",
    huggingface_id="EuroEval/belebele-fr-mini",
    task=MCRC,
    languages=[FR],
    unofficial=True,
)
