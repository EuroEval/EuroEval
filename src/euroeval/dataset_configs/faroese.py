"""All Faroese dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import FAROESE
from ..tasks import LA, NER, RC, SENT

### Official datasets ###

FOSENT_CONFIG = DatasetConfig(
    name="fosent",
    source="EuroEval/fosent",
    task=SENT,
    languages=[FAROESE],
    _num_few_shot_examples=5,
)

SCALA_FO_CONFIG = DatasetConfig(
    name="scala-fo", source="EuroEval/scala-fo", task=LA, languages=[FAROESE]
)

FONE_CONFIG = DatasetConfig(
    name="fone", source="EuroEval/fone-mini", task=NER, languages=[FAROESE]
)

FOQA_CONFIG = DatasetConfig(
    name="foqa", source="EuroEval/foqa", task=RC, languages=[FAROESE]
)

# TODO: No Faroese version of the European values dataset exists yet
# EUROPEAN_VALUES_FO_CONFIG = DatasetConfig(
#     name="european-values-fo",
#     source="EuroEval/european-values-fo",
#     task=EUROPEAN_VALUES,
#     languages=[FAROESE],
#     splits=["test"],
#     bootstrap_samples=False,
#     _instruction_prompt="{text}",
# )
#
# EUROPEAN_VALUES_SITUATIONAL_FO_CONFIG = DatasetConfig(
#     name="european-values-situational-fo",
#     source="EuroEval/european-values-situational-fo",
#     task=EUROPEAN_VALUES,
#     languages=[FAROESE],
#     splits=["test"],
#     bootstrap_samples=False,
#     _instruction_prompt="{text}",
#     unofficial=True,
# )
#
# EUROPEAN_VALUES_COMPLETIONS_FO_CONFIG = DatasetConfig(
#     name="european-values-completions-fo",
#     source="EuroEval/european-values-completions-fo",
#     task=EUROPEAN_VALUES,
#     languages=[FAROESE],
#     splits=["test"],
#     bootstrap_samples=False,
#     _instruction_prompt="{text}",
#     unofficial=True,
# )


### Unofficial datasets ###

WIKIANN_FO_CONFIG = DatasetConfig(
    name="wikiann-fo",
    source="EuroEval/wikiann-fo-mini",
    task=NER,
    languages=[FAROESE],
    unofficial=True,
)

MULTI_WIKI_QA_FO_CONFIG = DatasetConfig(
    name="multi-wiki-qa-fo",
    source="EuroEval/multi-wiki-qa-fo-mini",
    task=RC,
    languages=[FAROESE],
    unofficial=True,
)
