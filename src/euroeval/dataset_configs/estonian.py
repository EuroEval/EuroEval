"""All Estonian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import ESTONIAN
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

ESTONIAN_VALENCE_CONFIG = DatasetConfig(
    name="estonian-valence",
    source="EuroEval/estonian-valence",
    task=SENT,
    languages=[ESTONIAN],
)

GRAMMAR_ET_CONFIG = DatasetConfig(
    name="grammar-et", source="EuroEval/grammar-et", task=LA, languages=[ESTONIAN]
)

ESTNER_CONFIG = DatasetConfig(
    name="estner", source="EuroEval/estner-mini", task=NER, languages=[ESTONIAN]
)

MULTI_WIKI_QA_ET_CONFIG = DatasetConfig(
    name="multi-wiki-qa-et",
    source="EuroEval/multi-wiki-qa-et-mini",
    task=RC,
    languages=[ESTONIAN],
)

ERR_NEWS_CONFIG = DatasetConfig(
    name="err-news", source="EuroEval/err-news-mini", task=SUMM, languages=[ESTONIAN]
)

TRIVIA_ET_CONFIG = DatasetConfig(
    name="trivia-et", source="EuroEval/trivia-et", task=KNOW, languages=[ESTONIAN]
)

WINOGRANDE_ET_CONFIG = DatasetConfig(
    name="winogrande-et",
    source="EuroEval/winogrande-et",
    task=COMMON_SENSE,
    languages=[ESTONIAN],
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
    source="EuroEval/european-values-et",
    task=EUROPEAN_VALUES,
    languages=[ESTONIAN],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
)

### Unofficial datasets ###

SCALA_ET_CONFIG = DatasetConfig(
    name="scala-et",
    source="EuroEval/scala-et",
    task=LA,
    languages=[ESTONIAN],
    unofficial=True,
)

EXAM_ET_CONFIG = DatasetConfig(
    name="exam-et",
    source="EuroEval/exam-et",
    task=KNOW,
    languages=[ESTONIAN],
    _labels=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"],
    unofficial=True,
)

MMLU_ET_CONFIG = DatasetConfig(
    name="mmlu-et",
    source="EuroEval/mmlu-et-mini",
    task=KNOW,
    languages=[ESTONIAN],
    unofficial=True,
)
