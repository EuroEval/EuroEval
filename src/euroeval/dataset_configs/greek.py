"""All Greek dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import GREEK
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

GREEK_SA_CONFIG = DatasetConfig(
    name="greek-sa",
    pretty_name="Greek Sentiment Analysis",
    source="EuroEval/greek-sa-mini",
    task=SENT,
    languages=[GREEK],
    _labels=["negative", "positive"],
)

SCALA_EL_CONFIG = DatasetConfig(
    name="scala-el",
    pretty_name="ScaLA-el",
    source="EuroEval/scala-el",
    task=LA,
    languages=[GREEK],
)

ELNER_CONFIG = DatasetConfig(
    name="elner",
    pretty_name="ElNER",
    source="EuroEval/elner-mini",
    task=NER,
    languages=[GREEK],
)

MULTI_WIKI_QA_EL_CONFIG = DatasetConfig(
    name="multi-wiki-qa-el",
    pretty_name="MultiWikiQA-el",
    source="EuroEval/multi-wiki-qa-el-mini",
    task=RC,
    languages=[GREEK],
)

GREEK_WIKIPEDIA_CONFIG = DatasetConfig(
    name="greek-wikipedia",
    pretty_name="Greek Wikipedia",
    source="EuroEval/greek-wikipedia-mini",
    task=SUMM,
    languages=[GREEK],
)

GLOBAL_MMLU_EL_CONFIG = DatasetConfig(
    name="global-mmlu-el",
    pretty_name="GlobalMMLU-el",
    source="EuroEval/global-mmlu-el-mini",
    task=KNOW,
    languages=[GREEK],
)

WINOGRANDE_EL_CONFIG = DatasetConfig(
    name="winogrande-el",
    pretty_name="Winogrande-el",
    source="EuroEval/winogrande-el",
    task=COMMON_SENSE,
    languages=[GREEK],
)
