"""All Estonian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import ET
from ..tasks import COMMON_SENSE, NER, SENT, SUMM

### Official datasets ###

ESTONIAN_VALENCE_CONFIG = DatasetConfig(
    name="estonian-valence",
    pretty_name="the Estonian sentiment classification dataset Estonian Valence",
    huggingface_id="EuroEval/estonian-valence",
    task=SENT,
    languages=[ET],
)

WINOGRANDE_ET_CONFIG = DatasetConfig(
    name="winogrande-et",
    pretty_name="the Estonian common-sense reasoning dataset Winogrande-et",
    huggingface_id="EuroEval/winogrande-et",
    task=COMMON_SENSE,
    languages=[ET],
    # requires custom templates as WinoGrande is different from
    # the usual multiple choice tasks
    _prompt_prefix="Sulle esitatakse lüngaga (_) tekstülesanded, "
    "igal ülesandel on kaks vastusevarianti (a ja b).",
    # includes the question and the options
    _prompt_template="Tekstülesanne: {text}\nVastus: {label}",
    _instruction_prompt="Tekstülesanne: {text}\n\n"
    "Sinu ülesanne on valida lünka sobiv vastusevariant. "
    "Vasta ainult {labels_str}. Muud vastused ei ole lubatud.",
    _labels=["a", "b"],
)

ESTNER_CONFIG = DatasetConfig(
    name="estner",
    pretty_name="the Estonian named entity recognition dataset EstNER",
    huggingface_id="EuroEval/estner-mini",
    task=NER,
    languages=[ET],
)

ERR_NEWS_CONFIG = DatasetConfig(
    name="err-news",
    pretty_name="the Estonian summarisation dataset ErrNews",
    huggingface_id="EuroEval/err-news-mini",
    task=SUMM,
    languages=[ET],
)
