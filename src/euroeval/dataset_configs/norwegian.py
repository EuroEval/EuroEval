"""All Norwegian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import NB, NN, NO
from ..tasks import COMMON_SENSE, KNOW, LA, MCRC, NER, RC, SENT, SUMM

### Official datasets ###

NOREC_CONFIG = DatasetConfig(
    name="norec",
    pretty_name="the truncated version of the Norwegian sentiment classification "
    "dataset NoReC",
    huggingface_id="EuroEval/norec-mini",
    task=SENT,
    languages=[NB, NN, NO],
)

SCALA_NB_CONFIG = DatasetConfig(
    name="scala-nb",
    pretty_name="the Bokmål part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-nb",
    task=LA,
    languages=[NB, NO],
)

SCALA_NN_CONFIG = DatasetConfig(
    name="scala-nn",
    pretty_name="the Nynorsk part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-nn",
    task=LA,
    languages=[NN],
)

NORNE_NB_CONFIG = DatasetConfig(
    name="norne-nb",
    pretty_name="the truncated version of the Bokmål part of the Norwegian named "
    "entity recognition dataset NorNE",
    huggingface_id="EuroEval/norne-nb-mini",
    task=NER,
    languages=[NB, NO],
)

NORNE_NN_CONFIG = DatasetConfig(
    name="norne-nn",
    pretty_name="the truncated version of the Nynorsk part of the Norwegian named "
    "entity recognition dataset NorNE",
    huggingface_id="EuroEval/norne-nn-mini",
    task=NER,
    languages=[NN],
)

NORQUAD_CONFIG = DatasetConfig(
    name="norquad",
    pretty_name="the truncated version of the Norwegian question answering "
    "dataset NorQuAD",
    huggingface_id="EuroEval/norquad-mini",
    task=RC,
    languages=[NB, NN, NO],
    _num_few_shot_examples=2,
)

NO_SAMMENDRAG_CONFIG = DatasetConfig(
    name="no-sammendrag",
    pretty_name="the truncated version of the Norwegian summarisation dataset "
    "Norske Sammendrag",
    huggingface_id="EuroEval/no-sammendrag-mini",
    task=SUMM,
    languages=[NB, NN, NO],
)

NRK_QUIZ_QA_CONFIG = DatasetConfig(
    name="nrk-quiz-qa",
    pretty_name="the truncated version of the Norwegian knowledge dataset NRK Quiz QA",
    huggingface_id="EuroEval/nrk-quiz-qa-mini",
    task=KNOW,
    languages=[NB, NN, NO],
)

NOR_COMMON_SENSE_QA_CONFIG = DatasetConfig(
    name="nor-common-sense-qa",
    pretty_name="the truncated version of the Norwegian common-sense reasoning dataset "
    "NorCommonSenseQA",
    huggingface_id="EuroEval/nor-common-sense-qa",
    task=COMMON_SENSE,
    languages=[NB, NN, NO],
)


### Unofficial datasets ###

NO_COLA_CONFIG = DatasetConfig(
    name="no-cola",
    pretty_name="the truncated version of the Norwegian linguistic acceptability "
    "dataset NoCoLA",
    huggingface_id="EuroEval/no-cola-mini",
    task=LA,
    languages=[NB, NO],
    unofficial=True,
)

NORGLM_MULTI_QA = DatasetConfig(
    name="norglm-multi-qa",
    pretty_name="the question answering part of the Norwegian NorGLM multi-task human "
    "annotated dataset NO-Multi-QA-Sum",
    huggingface_id="EuroEval/norglm-multi-qa",
    task=RC,
    languages=[NB, NN, NO],
    _num_few_shot_examples=2,
    unofficial=True,
)

NORGLM_MULTI_SUM = DatasetConfig(
    name="norglm-multi-sum",
    pretty_name="the summarisation part of the Norwegian NorGLM multi-task human "
    "annotated dataset NO-Multi-QA-Sum",
    huggingface_id="EuroEval/norglm-multi-sum",
    task=SUMM,
    languages=[NB, NN, NO],
    unofficial=True,
)

SCHIBSTED_NO_CONFIG = DatasetConfig(
    name="schibsted-no",
    pretty_name="the Norwegian summarisation dataset Schibsted-no",
    huggingface_id="EuroEval/schibsted-article-summaries-no",
    task=SUMM,
    languages=[NB, NN, NO],
    unofficial=True,
)

PERSONAL_SUM_CONFIG = DatasetConfig(
    name="personal-sum",
    pretty_name="the Norwegian summarisation dataset personal-sum",
    huggingface_id="EuroEval/personal-sum",
    task=SUMM,
    languages=[NB, NN, NO],
    unofficial=True,
)

MMLU_NO_CONFIG = DatasetConfig(
    name="mmlu-no",
    pretty_name="the truncated version of the Norwegian knowledge dataset MMLU-no, "
    "translated from the English MMLU dataset",
    huggingface_id="EuroEval/mmlu-no-mini",
    task=KNOW,
    languages=[NB, NN, NO],
    unofficial=True,
)

ARC_NO_CONFIG = DatasetConfig(
    name="arc-no",
    pretty_name="the truncated version of the Norwegian knowledge dataset ARC-no, "
    "translated from the English ARC dataset",
    huggingface_id="EuroEval/arc-no-mini",
    task=KNOW,
    languages=[NB, NN, NO],
    unofficial=True,
)

HELLASWAG_NO_CONFIG = DatasetConfig(
    name="hellaswag-no",
    pretty_name="the truncated version of the Norwegian common-sense reasoning dataset "
    "HellaSwag-no, translated from the English HellaSwag dataset",
    huggingface_id="EuroEval/hellaswag-no-mini",
    task=COMMON_SENSE,
    languages=[NB, NN, NO],
    unofficial=True,
)

BELEBELE_NO_CONFIG = DatasetConfig(
    name="belebele-no",
    pretty_name="the Norwegian multiple choice reading comprehension dataset "
    "BeleBele-no, translated from the English BeleBele dataset",
    huggingface_id="EuroEval/belebele-no-mini",
    task=MCRC,
    languages=[NB, NN, NO],
    unofficial=True,
)
