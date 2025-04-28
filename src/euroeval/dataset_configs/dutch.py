"""All Dutch dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import NL
from ..tasks import COMMON_SENSE, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

DBRD_CONFIG = DatasetConfig(
    name="dbrd",
    pretty_name="the truncated version of the Dutch sentiment classification "
    "dataset DBRD",
    huggingface_id="EuroEval/dbrd-mini",
    task=SENT,
    languages=[NL],
    _labels=["negative", "positive"],
)

SCALA_NL_CONFIG = DatasetConfig(
    name="scala-nl",
    pretty_name="the Dutch part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-nl",
    task=LA,
    languages=[NL],
)

CONLL_NL_CONFIG = DatasetConfig(
    name="conll-nl",
    pretty_name="the Dutch part of the truncated version of the named entity "
    "recognition dataset CoNLL 2002",
    huggingface_id="EuroEval/conll-nl-mini",
    task=NER,
    languages=[NL],
)

SQUAD_NL_CONFIG = DatasetConfig(
    name="squad-nl",
    pretty_name="the truncated version of the Dutch reading comprehension dataset "
    "SQuAD-nl, translated from the English SQuAD dataset",
    huggingface_id="EuroEval/squad-nl-v2-mini",
    task=RC,
    languages=[NL],
)

WIKI_LINGUA_NL_CONFIG = DatasetConfig(
    name="wiki-lingua-nl",
    pretty_name="the Dutch part of the truncated version of the summarisation dataset "
    "WikiLingua",
    huggingface_id="EuroEval/wiki-lingua-nl-mini",
    task=SUMM,
    languages=[NL],
)

MMLU_NL_CONFIG = DatasetConfig(
    name="mmlu-nl",
    pretty_name="the truncated version of the Dutch knowledge dataset MMLU-nl, "
    "translated from the English MMLU dataset",
    huggingface_id="EuroEval/mmlu-nl-mini",
    task=KNOW,
    languages=[NL],
)

HELLASWAG_NL_CONFIG = DatasetConfig(
    name="hellaswag-nl",
    pretty_name="the truncated version of the Dutch common-sense reasoning dataset "
    "HellaSwag-nl, translated from the English HellaSwag dataset",
    huggingface_id="EuroEval/hellaswag-nl-mini",
    task=COMMON_SENSE,
    languages=[NL],
)


### Unofficial datasets ###

DUTCH_COLA_CONFIG = DatasetConfig(
    name="dutch-cola",
    pretty_name="the truncated version of the Dutch linguistic acceptability dataset "
    "Dutch CoLA",
    huggingface_id="EuroEval/dutch-cola",
    task=LA,
    languages=[NL],
    unofficial=True,
)

DUTCH_COLA_FULL_CONFIG = DatasetConfig(
    name="dutch-cola-full",
    pretty_name="the Dutch linguistic acceptability dataset Dutch CoLA",
    huggingface_id="EuroEval/dutch-cola-full",
    task=LA,
    languages=[NL],
    unofficial=True,
)

ARC_NL_CONFIG = DatasetConfig(
    name="arc-nl",
    pretty_name="the truncated version of the Dutch knowledge dataset ARC-nl, "
    "translated from the English ARC dataset",
    huggingface_id="EuroEval/arc-nl-mini",
    task=KNOW,
    languages=[NL],
    unofficial=True,
)

BELEBELE_NL_CONFIG = DatasetConfig(
    name="belebele-nl",
    pretty_name="the Dutch multiple choice reading comprehension dataset BeleBele-nl, "
    "translated from the English BeleBele dataset",
    huggingface_id="EuroEval/belebele-nl-mini",
    task=MCRC,
    languages=[NL],
    unofficial=True,
)
