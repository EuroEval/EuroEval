"""All French dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import FR
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

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

EUROPEAN_VALUES_FR_CONFIG = DatasetConfig(
    name="european-values-fr",
    pretty_name="the French version of the European values evaluation dataset",
    huggingface_id="EuroEval/european-values-fr",
    task=EUROPEAN_VALUES,
    languages=[FR],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
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

MULTI_WIKI_QA_FR_CONFIG = DatasetConfig(
    name="multi-wiki-qa-fr",
    pretty_name="the truncated version of the French part of the reading "
    "comprehension dataset MultiWikiQA",
    huggingface_id="EuroEval/multi-wiki-qa-fr-mini",
    task=RC,
    languages=[FR],
    unofficial=True,
)

GOLDENSWAG_FR_CONFIG = DatasetConfig(
    name="goldenswag-fr",
    pretty_name="the truncated version of the French common-sense reasoning "
    "dataset GoldenSwag-fr, translated from the English GoldenSwag dataset",
    huggingface_id="EuroEval/goldenswag-fr-mini",
    task=COMMON_SENSE,
    languages=[FR],
    unofficial=True,
)

EUROPEAN_VALUES_SITUATIONAL_FR_CONFIG = DatasetConfig(
    name="european-values-situational-fr",
    pretty_name="the French version of the European values evaluation dataset, where "
    "the questions are phrased in a situational way",
    huggingface_id="EuroEval/european-values-situational-fr",
    task=EUROPEAN_VALUES,
    languages=[FR],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)

EUROPEAN_VALUES_COMPLETIONS_FR_CONFIG = DatasetConfig(
    name="european-values-completions-fr",
    pretty_name="the French version of the European values evaluation dataset, where "
    "the questions are phrased as sentence completions",
    huggingface_id="EuroEval/european-values-completions-fr",
    task=EUROPEAN_VALUES,
    languages=[FR],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)
