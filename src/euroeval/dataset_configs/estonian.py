"""All Estonian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, ESTONIAN
from ..tasks import (
    COMMON_SENSE,
    EUROPEAN_VALUES,
    HALLU,
    INSTRUCTION_FOLLOWING,
    KNOW,
    LA,
    NER,
    RC,
    SENT,
    SUMM,
    TOOL_CALLING,
    TRANSLATION,
)

# Official datasets ###

WMT24PP_EN_ET_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-et",
    pretty_name="WMT24++-en-et",
    source="EuroEval/wmt24pp-en-et",
    task=TRANSLATION,
    languages=[ESTONIAN],
    source_language=ENGLISH,
    target_language=ESTONIAN,
    revision="2f8160f117501d75f2d0d3742c9e3989bcc64769",
)

FLORES_EN_ET_CONFIG = TranslationDatasetConfig(
    name="flores-en-et",
    pretty_name="FLORES-en-et",
    source="EuroEval/flores-en-et",
    task=TRANSLATION,
    languages=[ESTONIAN],
    source_language=ENGLISH,
    target_language=ESTONIAN,
    revision="05f8684c7095ecab69ee95846bf0bf66b552881b",
)

ESTONIAN_VALENCE_CONFIG = DatasetConfig(
    name="estonian-valence",
    pretty_name="Estonian Valence",
    source="EuroEval/estonian-valence",
    task=SENT,
    languages=[ESTONIAN],
    revision="8feabbdbb2857297c756b3e1204b680138613456",
)

GRAMMAR_ET_CONFIG = DatasetConfig(
    name="grammar-et",
    pretty_name="Grammar-et",
    source="EuroEval/grammar-et",
    task=LA,
    languages=[ESTONIAN],
    revision="441748489b2a3672286b19ab94c2c43ed435f935",
)

ESTNER_CONFIG = DatasetConfig(
    name="estner",
    pretty_name="EstNER",
    source="EuroEval/estner-mini",
    task=NER,
    languages=[ESTONIAN],
    revision="ad6e535d977d35aca07c1478a0ea0883008b0e09",
)

MULTI_WIKI_QA_ET_CONFIG = DatasetConfig(
    name="multi-wiki-qa-et",
    pretty_name="MultiWikiQA-et",
    source="EuroEval/multi-wiki-qa-et-mini",
    task=RC,
    languages=[ESTONIAN],
    revision="e7eebf5c4184ac5b89745aa10355755fe3edcd9f",
)

ERR_NEWS_CONFIG = DatasetConfig(
    name="err-news",
    pretty_name="ERR News",
    source="EuroEval/err-news-mini",
    task=SUMM,
    languages=[ESTONIAN],
    revision="1c4f3315de8f6806abacdb4ac31cd36124c03b35",
)

TRIVIA_ET_CONFIG = DatasetConfig(
    name="trivia-et",
    pretty_name="Trivia-et",
    source="EuroEval/trivia-et",
    task=KNOW,
    languages=[ESTONIAN],
    revision="a3c11692a17fcc12a6ed44e566b199ed090d0567",
)

WINOGRANDE_ET_CONFIG = DatasetConfig(
    name="winogrande-et",
    pretty_name="Winogrande-et",
    source="EuroEval/winogrande-et",
    task=COMMON_SENSE,
    languages=[ESTONIAN],
    prompt_prefix="Sulle esitatakse lüngaga (_) tekstülesanded, "
    "igal ülesandel on kaks vastusevarianti (a ja b).",
    prompt_template="Tekstülesanne: {text}\nVastus: {label}",
    instruction_prompt="Tekstülesanne: {text}\n\n"
    "Sinu ülesanne on valida lünka sobiv vastusevariant. "
    "Vasta ainult {labels_str}. Muud vastused ei ole lubatud.",
    labels=["a", "b"],
    revision="633cf60575bbfdb43afa97fcbb4be0875807b358",
)

IFEVAL_ET_CONFIG = DatasetConfig(
    name="ifeval-et",
    pretty_name="IFEval-et",
    source="EuroEval/ifeval-et",
    task=INSTRUCTION_FOLLOWING,
    languages=[ESTONIAN],
    train_split=None,
    val_split=None,
    revision="f7472356783c9e2515190f7ea3dd7212f3c4c11c",
)

VALEU_ET_CONFIG = DatasetConfig(
    name="valeu-et",
    pretty_name="VaLEU-et",
    source="EuroEval/european-values-et",
    task=EUROPEAN_VALUES,
    languages=[ESTONIAN],
    train_split=None,
    val_split=None,
    bootstrap_samples=False,
    instruction_prompt="{text}",
    revision="9602fd1a5dc4fcef9a711d6b8a19bbdae8ec086c",
)

RAGTRUTH_ET_CONFIG = DatasetConfig(
    name="ragtruth-et",
    pretty_name="RAGTruth-et",
    source="EuroEval/ragtruth-translated-hallucinations-et-mini",
    task=HALLU,
    languages=[ESTONIAN],
    train_split=None,
    revision="d5b94ef3f08c9dd8a14914dcfb7dbc6b3268ae98",
)


# Unofficial datasets ###

MULTI_BFCL_ET_CONFIG = DatasetConfig(
    name="multi-bfcl-et",
    pretty_name="MultiBFCL-et",
    source="EuroEval/multi-bfcl-et-mini",
    task=TOOL_CALLING,
    languages=[ESTONIAN],
    train_split=None,
    revision="42ca76b5fd261c7a590e65e2affa70a934b42569",
)


WMT24PP_ET_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-et-en",
    pretty_name="WMT24++-et-en",
    source="EuroEval/wmt24pp-et-en",
    task=TRANSLATION,
    languages=[ESTONIAN],
    source_language=ESTONIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="bb9fc69144736bcec2dd8b4b5eac3eff78f68d8a",
)

FLORES_ET_EN_CONFIG = TranslationDatasetConfig(
    name="flores-et-en",
    pretty_name="FLORES-et-en",
    source="EuroEval/flores-et-en",
    task=TRANSLATION,
    languages=[ESTONIAN],
    source_language=ESTONIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="694b335b889238e81cc111a2988c718f4d2d72ca",
)

MULTI_IFEVAL_ET_CONFIG = DatasetConfig(
    name="multi-ifeval-et",
    pretty_name="MultiIFEval-et",
    source="EuroEval/multi-ifeval-et",
    task=INSTRUCTION_FOLLOWING,
    languages=[ESTONIAN],
    train_split=None,
    val_split=None,
    unofficial=True,
    revision="6aac2bee7c885f06443d007f6927964985185880",
)

SCALA_ET_CONFIG = DatasetConfig(
    name="scala-et",
    pretty_name="ScaLA-et",
    source="EuroEval/scala-et",
    task=LA,
    languages=[ESTONIAN],
    unofficial=True,
    revision="0bb1ce6a88c5e8516ab497bd479c35d3212089cc",
)

EXAM_ET_CONFIG = DatasetConfig(
    name="exam-et",
    pretty_name="Exam-et",
    source="EuroEval/exam-et",
    task=KNOW,
    languages=[ESTONIAN],
    labels=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"],
    unofficial=True,
    revision="391fe59c22d1028a8c75d1e23128496df92f6436",
)

MMLU_ET_CONFIG = DatasetConfig(
    name="mmlu-et",
    pretty_name="MMLU-et",
    source="EuroEval/mmlu-et-mini",
    task=KNOW,
    languages=[ESTONIAN],
    unofficial=True,
    revision="ded933ec05085c60d69616f087ff2fbe62661174",
)

INCLUDE_ET_CONFIG = DatasetConfig(
    name="include-et",
    pretty_name="INCLUDE-et",
    source="EuroEval/include-et-mini",
    task=KNOW,
    languages=[ESTONIAN],
    unofficial=True,
    revision="d0b69b4f4645b0bb43276c7c3ebd4b0c36e55fd4",
)
