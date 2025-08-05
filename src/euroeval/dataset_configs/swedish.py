"""All Swedish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import SV
from ..tasks import COMMON_SENSE, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

SWEREC_CONFIG = DatasetConfig(
    name="swerec",
    pretty_name="the truncated version of the Swedish sentiment classification "
    "dataset SweReC",
    huggingface_id="EuroEval/swerec-mini",
    task=SENT,
    languages=[SV],
)

SCALA_SV_CONFIG = DatasetConfig(
    name="scala-sv",
    pretty_name="The Swedish part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-sv",
    task=LA,
    languages=[SV],
)

SUC3_CONFIG = DatasetConfig(
    name="suc3",
    pretty_name="the truncated version of the Swedish named entity recognition "
    "dataset SUC 3.0",
    huggingface_id="EuroEval/suc3-mini",
    task=NER,
    languages=[SV],
)

SCANDIQA_SV_CONFIG = DatasetConfig(
    name="scandiqa-sv",
    pretty_name="the Swedish part of the truncated version of the question answering "
    "dataset ScandiQA",
    huggingface_id="EuroEval/scandiqa-sv-mini",
    task=RC,
    languages=[SV],
)

SWEDN_CONFIG = DatasetConfig(
    name="swedn",
    pretty_name="the truncated version of the Swedish summarisation dataset SweDN",
    huggingface_id="EuroEval/swedn-mini",
    task=SUMM,
    languages=[SV],
)

MMLU_SV_CONFIG = DatasetConfig(
    name="mmlu-sv",
    pretty_name="the truncated version of the Swedish knowledge dataset MMLU-sv, "
    "translated from the English MMLU dataset",
    huggingface_id="EuroEval/mmlu-sv-mini",
    task=KNOW,
    languages=[SV],
)

HELLASWAG_SV_CONFIG = DatasetConfig(
    name="hellaswag-sv",
    pretty_name="the truncated version of the Swedish common-sense reasoning dataset "
    "HellaSwag-sv, translated from the English HellaSwag dataset",
    huggingface_id="EuroEval/hellaswag-sv-mini",
    task=COMMON_SENSE,
    languages=[SV],
)


### Unofficial datasets ###

SCHIBSTED_SV_CONFIG = DatasetConfig(
    name="schibsted-sv",
    pretty_name="the Swedish summarisation dataset Schibsted-sv",
    huggingface_id="EuroEval/schibsted-article-summaries-sv",
    task=SUMM,
    languages=[SV],
    unofficial=True,
)

ARC_SV_CONFIG = DatasetConfig(
    name="arc-sv",
    pretty_name="the truncated version of the Swedish knowledge dataset ARC-sv, "
    "translated from the English ARC dataset",
    huggingface_id="EuroEval/arc-sv-mini",
    task=KNOW,
    languages=[SV],
    unofficial=True,
)

BELEBELE_SV_CONFIG = DatasetConfig(
    name="belebele-sv",
    pretty_name="the Swedish multiple choice reading comprehension dataset "
    "BeleBele-sv, translated from the English BeleBele dataset",
    huggingface_id="EuroEval/belebele-sv-mini",
    task=MCRC,
    languages=[SV],
    unofficial=True,
)

MULTI_WIKI_QA_SV_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sv",
    pretty_name="the truncated version of the Swedish part of the reading "
    "comprehension dataset MultiWikiQA",
    huggingface_id="EuroEval/multi-wiki-qa-sv-mini",
    task=RC,
    languages=[SV],
    unofficial=True,
)

GOLDENSWAG_SV_CONFIG = DatasetConfig(
    name="goldenswag-sv",
    pretty_name="the truncated version of the Swedish common-sense reasoning "
    "dataset GoldenSwag-sv, translated from the English GoldenSwag dataset",
    huggingface_id="EuroEval/goldenswag-sv-mini",
    task=COMMON_SENSE,
    languages=[SV],
    unofficial=True,
)
