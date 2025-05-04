"""All English dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import EN
from ..tasks import COMMON_SENSE, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

SST5_CONFIG = DatasetConfig(
    name="sst5",
    pretty_name="the truncated version of the English sentiment classification "
    "dataset SST5",
    huggingface_id="EuroEval/sst5-mini",
    task=SENT,
    languages=[EN],
)

SCALA_EN_CONFIG = DatasetConfig(
    name="scala-en",
    pretty_name="the English part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-en",
    task=LA,
    languages=[EN],
)

CONLL_EN_CONFIG = DatasetConfig(
    name="conll-en",
    pretty_name="the truncated version of the English named entity recognition "
    "dataset CoNLL 2003",
    huggingface_id="EuroEval/conll-en-mini",
    task=NER,
    languages=[EN],
)

SQUAD_CONFIG = DatasetConfig(
    name="squad",
    pretty_name="the truncated version of the English question answering dataset SQuAD",
    huggingface_id="EuroEval/squad-mini",
    task=RC,
    languages=[EN],
)

CNN_DAILYMAIL_CONFIG = DatasetConfig(
    name="cnn-dailymail",
    pretty_name="the truncated version of the English summarisation dataset "
    "CNN-DailyMail",
    huggingface_id="EuroEval/cnn-dailymail-mini",
    task=SUMM,
    languages=[EN],
)

MMLU_CONFIG = DatasetConfig(
    name="mmlu",
    pretty_name="the truncated version of the English knowledge dataset MMLU",
    huggingface_id="EuroEval/mmlu-mini",
    task=KNOW,
    languages=[EN],
)

HELLASWAG_CONFIG = DatasetConfig(
    name="hellaswag",
    pretty_name="the truncated version of the English common-sense reasoning "
    "dataset HellaSwag",
    huggingface_id="EuroEval/hellaswag-mini",
    task=COMMON_SENSE,
    languages=[EN],
)


### Unofficial datasets ###

ARC_CONFIG = DatasetConfig(
    name="arc",
    pretty_name="the truncated version of the English knowledge dataset ARC",
    huggingface_id="EuroEval/arc-mini",
    task=KNOW,
    languages=[EN],
    unofficial=True,
)

BELEBELE_CONFIG = DatasetConfig(
    name="belebele-en",
    pretty_name="the English multiple choice reading comprehension dataset BeleBele",
    huggingface_id="EuroEval/belebele-mini",
    task=MCRC,
    languages=[EN],
    unofficial=True,
)
