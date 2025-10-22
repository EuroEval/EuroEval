"""All Greek dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import EL
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

GREEK_SA_CONFIG = DatasetConfig(
    name="greek-sa",
    pretty_name="the truncated version of the Greek sentiment "
    "classification dataset Greek-SA",
    huggingface_id="EuroEval/greek-sa-mini",
    task=SENT,
    languages=[EL],
    _labels=["negative", "positive"],
)

SCALA_EL_CONFIG = DatasetConfig(
    name="scala-el",
    pretty_name="the Greek part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-el",
    task=LA,
    languages=[EL],
)

ELNER_CONFIG = DatasetConfig(
    name="elner",
    pretty_name="the truncated version of the Greek named entity "
    "recognition dataset elNER",
    huggingface_id="EuroEval/elner-mini",
    task=NER,
    languages=[EL],
)

MULTI_WIKI_QA_EL_CONFIG = DatasetConfig(
    name="multi-wiki-qa-el",
    pretty_name="the truncated version of the Greek part of the reading comprehension "
    "dataset MultiWikiQA",
    huggingface_id="EuroEval/multi-wiki-qa-el-mini",
    task=RC,
    languages=[EL],
)

GREEK_WIKIPEDIA_CONFIG = DatasetConfig(
    name="greek-wikipedia",
    pretty_name="the truncated version of the Greek summarisation dataset "
    "Greek-Wikipedia",
    huggingface_id="EuroEval/greek-wikipedia-mini",
    task=SUMM,
    languages=[EL],
)

GLOBAL_MMLU_EL_CONFIG = DatasetConfig(
    name="global-mmlu-el",
    pretty_name="the truncated version of the Greek knowledge dataset "
    "GlobalMMLU-el, machine translated from the English GlobalMMLU dataset",
    huggingface_id="EuroEval/global-mmlu-el-mini",
    task=KNOW,
    languages=[EL],
)

WINOGRANDE_EL_CONFIG = DatasetConfig(
    name="winogrande-el",
    pretty_name="the Greek common-sense reasoning dataset Winogrande-el, translated "
    "from the English Winogrande dataset",
    huggingface_id="EuroEval/winogrande-el",
    task=COMMON_SENSE,
    languages=[EL],
)
