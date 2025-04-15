"""All Spanish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import ES
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

SENTIMENT_HEADLINES_CONFIG = DatasetConfig(
    name="sentiment-headlines-es",
    pretty_name="the truncated version of the Spanish sentiment headlines dataset",
    huggingface_id="EuroEval/sentiment-headlines-es",
    task=SENT,
    languages=[ES],
)

SCALA_ES_CONFIG = DatasetConfig(
    name="scala-es",
    pretty_name="the Spanish part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-es",
    task=LA,
    languages=[ES],
)

CONLL_ES_CONFIG = DatasetConfig(
    name="conll-es",
    pretty_name="the Spanish part of the truncated version of the named entity "
    "recognition dataset CoNLL 2002",
    huggingface_id="EuroEval/conll-es-mini",
    task=NER,
    languages=[ES],
)

MLQA_ES_CONFIG = DatasetConfig(
    name="mlqa-es",
    pretty_name="the Spanish version of the MLQA reading comprehension dataset",
    huggingface_id="EuroEval/mlqa-es",
    task=RC,
    languages=[ES],
)

MLSUM_ES_CONFIG = DatasetConfig(
    name="mlsum-es",
    pretty_name="the truncated version of the Spanish summarisation dataset MLSum-es",
    huggingface_id="EuroEval/mlsum-es-mini",
    task=SUMM,
    languages=[ES],
)

MMLU_ES_CONFIG = DatasetConfig(
    name="mmlu-es",
    pretty_name="the truncated version of the Spanish knowledge dataset MMLU-es, "
    "translated from the English MMLU dataset",
    huggingface_id="EuroEval/mmlu-es-mini",
    task=KNOW,
    languages=[ES],
)

HELLASWAG_ES_CONFIG = DatasetConfig(
    name="hellaswag-es",
    pretty_name="the truncated version of the Spanish common-sense reasoning dataset "
    "HellaSwag-es, translated from the English HellaSwag dataset",
    huggingface_id="EuroEval/hellaswag-es-mini",
    task=COMMON_SENSE,
    languages=[ES],
)


### Unofficial datasets ###

XQUAD_ES_CONFIG = DatasetConfig(
    name="xquad-es",
    pretty_name="the Spanish version of the XQuAD reading comprehension dataset",
    huggingface_id="EuroEval/xquad-es",
    task=RC,
    languages=[ES],
    unofficial=True,
)
