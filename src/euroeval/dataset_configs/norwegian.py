"""All Norwegian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import NORWEGIAN, NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK
from ..tasks import COMMON_SENSE, EUROPEAN_VALUES, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

NOREC_CONFIG = DatasetConfig(
    name="norec",
    pretty_name="NoReC",
    source="EuroEval/norec-mini",
    task=SENT,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
)

SCALA_NB_CONFIG = DatasetConfig(
    name="scala-nb",
    pretty_name="ScaLA-nb",
    source="EuroEval/scala-nb",
    task=LA,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN],
)

SCALA_NN_CONFIG = DatasetConfig(
    name="scala-nn",
    pretty_name="ScaLA-nn",
    source="EuroEval/scala-nn",
    task=LA,
    languages=[NORWEGIAN_NYNORSK],
)

NORNE_NB_CONFIG = DatasetConfig(
    name="norne-nb",
    pretty_name="NorNE-nb",
    source="EuroEval/norne-nb-mini",
    task=NER,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN],
)

NORNE_NN_CONFIG = DatasetConfig(
    name="norne-nn",
    pretty_name="NorNE-nn",
    source="EuroEval/norne-nn-mini",
    task=NER,
    languages=[NORWEGIAN_NYNORSK],
)

NORQUAD_CONFIG = DatasetConfig(
    name="norquad",
    pretty_name="NorQuAD",
    source="EuroEval/norquad-mini",
    task=RC,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    _num_few_shot_examples=2,
)

NO_SAMMENDRAG_CONFIG = DatasetConfig(
    name="no-sammendrag",
    pretty_name="NoSammendrag",
    source="EuroEval/no-sammendrag-mini",
    task=SUMM,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
)

NRK_QUIZ_QA_CONFIG = DatasetConfig(
    name="nrk-quiz-qa",
    pretty_name="NRK Quiz QA",
    source="EuroEval/nrk-quiz-qa-mini",
    task=KNOW,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
)

IDIOMS_NO_CONFIG = DatasetConfig(
    name="idioms-no",
    pretty_name="Idioms-no",
    source="EuroEval/idioms-no",
    task=KNOW,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
)

NOR_COMMON_SENSE_QA_CONFIG = DatasetConfig(
    name="nor-common-sense-qa",
    pretty_name="NorCommonSenseQA",
    source="EuroEval/nor-common-sense-qa",
    task=COMMON_SENSE,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    _labels=["a", "b", "c", "d", "e"],
)

VALEU_NO_CONFIG = DatasetConfig(
    name="valeu-no",
    pretty_name="VaLEU-no",
    source="EuroEval/european-values-no",
    task=EUROPEAN_VALUES,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    splits=["test"],
    bootstrap_samples=False,
    _instruction_prompt="{text}",
)


### Unofficial datasets ###

NO_COLA_CONFIG = DatasetConfig(
    name="no-cola",
    pretty_name="NoCoLA",
    source="EuroEval/no-cola-mini",
    task=LA,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN],
    unofficial=True,
)

NORGLM_MULTI_QA = DatasetConfig(
    name="norglm-multi-qa",
    pretty_name="NorGLM-Multi-QA",
    source="EuroEval/norglm-multi-qa",
    task=RC,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
)

NORGLM_MULTI_SUM = DatasetConfig(
    name="norglm-multi-sum",
    pretty_name="NorGLM-Multi-Sum",
    source="EuroEval/norglm-multi-sum",
    task=SUMM,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
)

SCHIBSTED_NO_CONFIG = DatasetConfig(
    name="schibsted-no",
    pretty_name="Schibsted-no",
    source="EuroEval/schibsted-article-summaries-no",
    task=SUMM,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
)

PERSONAL_SUM_CONFIG = DatasetConfig(
    name="personal-sum",
    pretty_name="Personal Sum",
    source="EuroEval/personal-sum",
    task=SUMM,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
)

MMLU_NO_CONFIG = DatasetConfig(
    name="mmlu-no",
    pretty_name="MMLU-no",
    source="EuroEval/mmlu-no-mini",
    task=KNOW,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
)

ARC_NO_CONFIG = DatasetConfig(
    name="arc-no",
    pretty_name="ARC-no",
    source="EuroEval/arc-no-mini",
    task=KNOW,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
)

HELLASWAG_NO_CONFIG = DatasetConfig(
    name="hellaswag-no",
    pretty_name="HellaSwag-no",
    source="EuroEval/hellaswag-no-mini",
    task=COMMON_SENSE,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
)

BELEBELE_NO_CONFIG = DatasetConfig(
    name="belebele-no",
    pretty_name="Belebele-no",
    source="EuroEval/belebele-no-mini",
    task=MCRC,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
)

MULTI_WIKI_QA_NB_CONFIG = DatasetConfig(
    name="multi-wiki-qa-nb",
    pretty_name="MultiWikiQA-nb",
    source="EuroEval/multi-wiki-qa-no-mini",
    task=RC,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN],
    unofficial=True,
)

MULTI_WIKI_QA_NN_CONFIG = DatasetConfig(
    name="multi-wiki-qa-nn",
    pretty_name="MultiWikiQA-nn",
    source="EuroEval/multi-wiki-qa-nn-mini",
    task=RC,
    languages=[NORWEGIAN_NYNORSK],
    unofficial=True,
)

WINOGRANDE_NO_CONFIG = DatasetConfig(
    name="winogrande-no",
    pretty_name="Winogrande-no",
    source="EuroEval/winogrande-no",
    task=COMMON_SENSE,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    _labels=["a", "b"],
    unofficial=True,
)
