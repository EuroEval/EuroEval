"""All Spanish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import SPANISH
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

SENTIMENT_HEADLINES_CONFIG = DatasetConfig(
    name="sentiment-headlines-es",
    source="EuroEval/sentiment-headlines-es",
    task=SENT,
    languages=[SPANISH],
)

SCALA_ES_CONFIG = DatasetConfig(
    name="scala-es", source="EuroEval/scala-es", task=LA, languages=[SPANISH]
)

CONLL_ES_CONFIG = DatasetConfig(
    name="conll-es", source="EuroEval/conll-es-mini", task=NER, languages=[SPANISH]
)

MLQA_ES_CONFIG = DatasetConfig(
    name="mlqa-es", source="EuroEval/mlqa-es", task=RC, languages=[SPANISH]
)

MLSUM_ES_CONFIG = DatasetConfig(
    name="mlsum-es", source="EuroEval/mlsum-es-mini", task=SUMM, languages=[SPANISH]
)

MMLU_ES_CONFIG = DatasetConfig(
    name="mmlu-es", source="EuroEval/mmlu-es-mini", task=KNOW, languages=[SPANISH]
)

HELLASWAG_ES_CONFIG = DatasetConfig(
    name="hellaswag-es",
    source="EuroEval/hellaswag-es-mini",
    task=COMMON_SENSE,
    languages=[SPANISH],
)

EUROPEAN_VALUES_ES_CONFIG = DatasetConfig(
    name="european-values-es",
    source="EuroEval/european-values-es",
    task=EUROPEAN_VALUES,
    languages=[SPANISH],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
)


### Unofficial datasets ###

XQUAD_ES_CONFIG = DatasetConfig(
    name="xquad-es",
    source="EuroEval/xquad-es",
    task=RC,
    languages=[SPANISH],
    unofficial=True,
)

BELEBELE_ES_CONFIG = DatasetConfig(
    name="belebele-es",
    source="EuroEval/belebele-es-mini",
    task=MCRC,
    languages=[SPANISH],
    unofficial=True,
)

MULTI_WIKI_QA_ES_CONFIG = DatasetConfig(
    name="multi-wiki-qa-es",
    source="EuroEval/multi-wiki-qa-es-mini",
    task=RC,
    languages=[SPANISH],
    unofficial=True,
)

GOLDENSWAG_ES_CONFIG = DatasetConfig(
    name="goldenswag-es",
    source="EuroEval/goldenswag-es-mini",
    task=COMMON_SENSE,
    languages=[SPANISH],
    unofficial=True,
)

WINOGRANDE_ES_CONFIG = DatasetConfig(
    name="winogrande-es",
    source="EuroEval/winogrande-es",
    task=COMMON_SENSE,
    languages=[SPANISH],
    _labels=["a", "b"],
    unofficial=True,
)

EUROPEAN_VALUES_SITUATIONAL_ES_CONFIG = DatasetConfig(
    name="european-values-situational-es",
    source="EuroEval/european-values-situational-es",
    task=EUROPEAN_VALUES,
    languages=[SPANISH],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)

EUROPEAN_VALUES_COMPLETIONS_ES_CONFIG = DatasetConfig(
    name="european-values-completions-es",
    source="EuroEval/european-values-completions-es",
    task=EUROPEAN_VALUES,
    languages=[SPANISH],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
    unofficial=True,
)
