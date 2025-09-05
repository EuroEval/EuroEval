"""All Estonian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import ET
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, LA, NER, RC, SENT, SUMM

from euroeval.common import DatasetConfig
from euroeval.tasks import KNOW
from euroeval.languages import ET

# Add this under the ### Unofficial datasets ### section



### Official datasets ###
EXAM_ET_CONFIG = DatasetConfig(
    name="exam_et",
    pretty_name="Estonian Exam Dataset", 
    huggingface_id="EuroEval/exam_et",
    task=KNOW,
    languages=[ET],
    unofficial=True,
)

ESTONIAN_VALENCE_CONFIG = DatasetConfig(
    name="estonian-valence",
    pretty_name="the Estonian sentiment classification dataset Estonian Valence",
    huggingface_id="EuroEval/estonian-valence",
    task=SENT,
    languages=[ET],
)

GRAMMAR_ET_CONFIG = DatasetConfig(
    name="grammar-et",
    pretty_name="the Estonian linguistic acceptability dataset Grammar-et",
    huggingface_id="EuroEval/grammar-et",
    task=LA,
    languages=[ET],
)

ESTNER_CONFIG = DatasetConfig(
    name="estner",
    pretty_name="the Estonian named entity recognition dataset EstNER",
    huggingface_id="EuroEval/estner-mini",
    task=NER,
    languages=[ET],
)

MULTI_WIKI_QA_ET_CONFIG = DatasetConfig(
    name="multi-wiki-qa-et",
    pretty_name="the truncated version of the Estonian part of the reading "
    "comprehension dataset MultiWikiQA",
    huggingface_id="EuroEval/multi-wiki-qa-et-mini",
    task=RC,
    languages=[ET],
)

ERR_NEWS_CONFIG = DatasetConfig(
    name="err-news",
    pretty_name="the Estonian summarisation dataset ErrNews",
    huggingface_id="EuroEval/err-news-mini",
    task=SUMM,
    languages=[ET],
)

# TODO: Missing knowledge dataset

WINOGRANDE_ET_CONFIG = DatasetConfig(
    name="winogrande-et",
    pretty_name="the Estonian common-sense reasoning dataset Winogrande-et",
    huggingface_id="EuroEval/winogrande-et",
    task=COMMON_SENSE,
    languages=[ET],
    _prompt_prefix="Sulle esitatakse lüngaga (_) tekstülesanded, "
    "igal ülesandel on kaks vastusevarianti (a ja b).",
    _prompt_template="Tekstülesanne: {text}\nVastus: {label}",
    _instruction_prompt="Tekstülesanne: {text}\n\n"
    "Sinu ülesanne on valida lünka sobiv vastusevariant. "
    "Vasta ainult {labels_str}. Muud vastused ei ole lubatud.",
    _labels=["a", "b"],
)

EUROPEAN_VALUES_ET_CONFIG = DatasetConfig(
    name="european-values-et",
    pretty_name="the Estonian version of the European values evaluation dataset",
    huggingface_id="EuroEval/european-values-et",
    task=EUROPEAN_VALUES,
    languages=[ET],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
)


### Unofficial datasets ###

SCALA_ET_CONFIG = DatasetConfig(
    name="scala-et",
    pretty_name="the Estonian part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-et",
    task=LA,
    languages=[ET],
    unofficial=True,
)
