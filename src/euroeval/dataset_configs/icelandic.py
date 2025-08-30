"""All Icelandic dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import IS
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

HOTTER_AND_COLDER_SENTIMENT_CONFIG = DatasetConfig(
    name="hotter-and-colder-sentiment",
    pretty_name="the sentiment classification part of the Icelandic dataset Hotter "
    "and Colder",
    huggingface_id="EuroEval/hotter-and-colder-sentiment",
    task=SENT,
    languages=[IS],
)

SCALA_IS_CONFIG = DatasetConfig(
    name="scala-is",
    pretty_name="the Icelandic part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-is",
    task=LA,
    languages=[IS],
)

MIM_GOLD_NER_CONFIG = DatasetConfig(
    name="mim-gold-ner",
    pretty_name="the truncated version of the Icelandic named entity recognition "
    "dataset MIM-GOLD-NER",
    huggingface_id="EuroEval/mim-gold-ner-mini",
    task=NER,
    languages=[IS],
)

NQII_CONFIG = DatasetConfig(
    name="nqii",
    pretty_name="the truncated version of the Icelandic reading comprehension dataset "
    "Natural Questions in Icelandic",
    huggingface_id="EuroEval/nqii-mini",
    task=RC,
    languages=[IS],
)

RRN_CONFIG = DatasetConfig(
    name="rrn",
    pretty_name="the truncated version of the Icelandic summarisation dataset "
    "RÚV Radio News",
    huggingface_id="EuroEval/rrn-mini",
    task=SUMM,
    languages=[IS],
)

ICELANDIC_KNOWLEDGE_CONFIG = DatasetConfig(
    name="icelandic-knowledge",
    pretty_name="the Icelandic knowledge dataset IcelandicKnowledge, derived from the "
    "IcelandicQA dataset",
    huggingface_id="EuroEval/icelandic-knowledge",
    task=KNOW,
    languages=[IS],
)

WINOGRANDE_IS_CONFIG = DatasetConfig(
    name="winogrande-is",
    pretty_name="the Icelandic common-sense reasoning dataset "
    "Winogrande-is, manually translated from the English Winogrande dataset",
    huggingface_id="EuroEval/winogrande-is",
    task=COMMON_SENSE,
    languages=[IS],
    _labels=["a", "b"],
)

EUROPEAN_VALUES_IS_CONFIG = DatasetConfig(
    name="european-values-is",
    pretty_name="the Icelandic version of the European values evaluation dataset",
    huggingface_id="EuroEval/european-values-is",
    task=EUROPEAN_VALUES,
    languages=[IS],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
)


### Unofficial datasets ###

ICE_EC_CONFIG = DatasetConfig(
    name="ice-ec",
    pretty_name="the truncated version of the Icelandic Error Corpus",
    huggingface_id="EuroEval/ice-ec",
    task=LA,
    languages=[IS],
    unofficial=True,
)

ICE_EC_FULL_CONFIG = DatasetConfig(
    name="ice-ec-full",
    pretty_name="the Icelandic Error Corpus",
    huggingface_id="EuroEval/ice-ec-full",
    task=LA,
    languages=[IS],
    unofficial=True,
)

ICE_LINGUISTIC_CONFIG = DatasetConfig(
    name="ice-linguistic",
    pretty_name="the Icelandic linguistic acceptability dataset IceLinguistic",
    huggingface_id="EuroEval/ice-linguistic",
    task=LA,
    languages=[IS],
    unofficial=True,
)

ICELANDIC_QA_CONFIG = DatasetConfig(
    name="icelandic-qa",
    pretty_name="the Icelandic reading comprehension dataset IcelandicQA",
    huggingface_id="EuroEval/icelandic-qa",
    task=RC,
    languages=[IS],
    unofficial=True,
)

MMLU_IS_CONFIG = DatasetConfig(
    name="mmlu-is",
    pretty_name="the truncated version of the Icelandic knowledge dataset MMLU-is, "
    "translated from the English MMLU dataset",
    huggingface_id="EuroEval/mmlu-is-mini",
    task=KNOW,
    languages=[IS],
    unofficial=True,
)

ARC_IS_CONFIG = DatasetConfig(
    name="arc-is",
    pretty_name="the truncated version of the Icelandic knowledge dataset ARC-is, "
    "translated from the English ARC dataset",
    huggingface_id="EuroEval/arc-is-mini",
    task=KNOW,
    languages=[IS],
    unofficial=True,
)

HELLASWAG_IS_CONFIG = DatasetConfig(
    name="hellaswag-is",
    pretty_name="the truncated version of the Icelandic common-sense reasoning dataset "
    "HellaSwag-is, translated from the English HellaSwag dataset",
    huggingface_id="EuroEval/hellaswag-is-mini",
    task=COMMON_SENSE,
    languages=[IS],
    unofficial=True,
)

BELEBELE_IS_CONFIG = DatasetConfig(
    name="belebele-is",
    pretty_name="the Icelandic multiple choice reading comprehension dataset "
    "BeleBele-is, translated from the English BeleBele dataset",
    huggingface_id="EuroEval/belebele-is-mini",
    task=MCRC,
    languages=[IS],
    unofficial=True,
)

MULTI_WIKI_QA_IS_CONFIG = DatasetConfig(
    name="multi-wiki-qa-is",
    pretty_name="the truncated version of the Icelandic part of the reading "
    "comprehension dataset MultiWikiQA",
    huggingface_id="EuroEval/multi-wiki-qa-is-mini",
    task=RC,
    languages=[IS],
    unofficial=True,
)

EUROPEAN_VALUES_SITUATIONAL_IS_CONFIG = DatasetConfig(
    name="european-values-situational-is",
    pretty_name="the Icelandic version of the European values evaluation dataset, "
    "where the questions are phrased in a situational way",
    huggingface_id="EuroEval/european-values-situational-is",
    task=EUROPEAN_VALUES,
    languages=[IS],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)

EUROPEAN_VALUES_COMPLETIONS_IS_CONFIG = DatasetConfig(
    name="european-values-completions-is",
    pretty_name="the Icelandic version of the European values evaluation dataset, "
    "where the questions are phrased as sentence completions",
    huggingface_id="EuroEval/european-values-completions-is",
    task=EUROPEAN_VALUES,
    languages=[IS],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)
