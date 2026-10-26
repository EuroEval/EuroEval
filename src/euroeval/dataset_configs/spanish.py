"""All Spanish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, SPANISH
from ..tasks import (
    COMMON_SENSE,
    EUROPEAN_VALUES,
    HALLU,
    INSTRUCTION_FOLLOWING,
    KNOW,
    LA,
    MCRC,
    NER,
    RC,
    SENT,
    SUMM,
    TOOL_CALLING,
    TRANSLATION,
)

# Official datasets ###

FLORES_EN_ES_CONFIG = TranslationDatasetConfig(
    name="flores-en-es",
    pretty_name="FLORES-en-es",
    source="EuroEval/flores-en-es",
    task=TRANSLATION,
    languages=[SPANISH],
    source_language=ENGLISH,
    target_language=SPANISH,
    revision="2ff555c449cf49bc8e2582988964242a254a7382",
)

SENTIMENT_HEADLINES_CONFIG = DatasetConfig(
    name="sentiment-headlines-es",
    pretty_name="Sentiment Headlines ES",
    source="EuroEval/sentiment-headlines-es",
    task=SENT,
    languages=[SPANISH],
    revision="6357121680ccb1f20a26b946baba35451ca03510",
)

SCALA_ES_CONFIG = DatasetConfig(
    name="scala-es",
    pretty_name="ScaLA-es",
    source="EuroEval/scala-es",
    task=LA,
    languages=[SPANISH],
    revision="7af567057ea1ef89783c5723a36d3780c703cd05",
)

CONLL_ES_CONFIG = DatasetConfig(
    name="conll-es",
    pretty_name="CoNLL-es",
    source="EuroEval/conll-es-mini",
    task=NER,
    languages=[SPANISH],
    revision="a9c6360120472a0d45fd098d994d2531bc7d2f52",
)

MLQA_ES_CONFIG = DatasetConfig(
    name="mlqa-es",
    pretty_name="MLQA-es",
    source="EuroEval/mlqa-es",
    task=RC,
    languages=[SPANISH],
    revision="fac21d6e3aabb75f797ae7aa6bb3a3b5b1a69e82",
)

MLSUM_ES_CONFIG = DatasetConfig(
    name="mlsum-es",
    pretty_name="MLSUM-es",
    source="EuroEval/mlsum-es-mini",
    task=SUMM,
    languages=[SPANISH],
    revision="ff230d9c4c35ceeb59361bac250c8dce606b7949",
)

IFEVAL_ES_CONFIG = DatasetConfig(
    name="ifeval-es",
    pretty_name="IFEval-es",
    source="EuroEval/ifeval-es",
    task=INSTRUCTION_FOLLOWING,
    languages=[SPANISH],
    train_split=None,
    val_split=None,
    revision="5cf9480d57d46576fceab4245765b7b61c88f41b",
)

VALEU_ES_CONFIG = DatasetConfig(
    name="valeu-es",
    pretty_name="VaLEU-es",
    source="EuroEval/european-values-es",
    task=EUROPEAN_VALUES,
    languages=[SPANISH],
    train_split=None,
    val_split=None,
    bootstrap_samples=False,
    instruction_prompt="{text}",
    revision="1daa1db875005613d54dcf9998f0c16b9a2cdb97",
)

RAGTRUTH_ES_CONFIG = DatasetConfig(
    name="ragtruth-es",
    pretty_name="RAGTruth-es",
    source="EuroEval/ragtruth-translated-hallucinations-es-mini",
    task=HALLU,
    languages=[SPANISH],
    train_split=None,
    revision="a502044935a515bbf5725b42e1264823783786e1",
)


WINOGRANDE_ES_CONFIG = DatasetConfig(
    name="winogrande-es",
    pretty_name="Winogrande-es",
    source="EuroEval/winogrande-es",
    task=COMMON_SENSE,
    languages=[SPANISH],
    labels=["a", "b"],
    revision="0114116436fa847a85a7542bfe7c634d80e36fdb",
)

INCLUDE_ES_CONFIG = DatasetConfig(
    name="include-es",
    pretty_name="INCLUDE-es",
    source="EuroEval/include-es-mini",
    task=KNOW,
    languages=[SPANISH],
    revision="13b453aaf49c8dd316567e09497191e80d639056",
)

MULTILOKO_ES_CONFIG = DatasetConfig(
    name="multiloko-es",
    pretty_name="MultiLoKo-es",
    source="EuroEval/multiloko-es-mini",
    task=KNOW,
    languages=[SPANISH],
    val_split=None,
    revision="acd8a60cad146704a6689009b6570ef4c17d2e30",
)


# Unofficial datasets ###

MULTI_BFCL_ES_CONFIG = DatasetConfig(
    name="multi-bfcl-es",
    pretty_name="MultiBFCL-es",
    source="EuroEval/multi-bfcl-es-mini",
    task=TOOL_CALLING,
    languages=[SPANISH],
    train_split=None,
    revision="72beaf7fc32dbcbe982ce4252c6597f3fd58590f",
)


FLORES_ES_EN_CONFIG = TranslationDatasetConfig(
    name="flores-es-en",
    pretty_name="FLORES-es-en",
    source="EuroEval/flores-es-en",
    task=TRANSLATION,
    languages=[SPANISH],
    source_language=SPANISH,
    target_language=ENGLISH,
    unofficial=True,
    revision="5213433c99cf36ea27bc50354825e64c9c866bb9",
)

MMLU_ES_CONFIG = DatasetConfig(
    name="mmlu-es",
    pretty_name="MMLU-es",
    source="EuroEval/mmlu-es-mini",
    task=KNOW,
    languages=[SPANISH],
    unofficial=True,
    revision="294409cf7c7a94e085ad6e33369159ba6b609808",
)

HELLASWAG_ES_CONFIG = DatasetConfig(
    name="hellaswag-es",
    pretty_name="HellaSwag-es",
    source="EuroEval/hellaswag-es-mini",
    task=COMMON_SENSE,
    languages=[SPANISH],
    unofficial=True,
    revision="e7cac5a54d8f756f293d65a26a9576841b7a16e6",
)

MULTI_IFEVAL_ES_CONFIG = DatasetConfig(
    name="multi-ifeval-es",
    pretty_name="MultiIFEval-es",
    source="EuroEval/multi-ifeval-es",
    task=INSTRUCTION_FOLLOWING,
    languages=[SPANISH],
    train_split=None,
    val_split=None,
    unofficial=True,
    revision="69b852e733c00b50b11ea26347bd85145f0941af",
)

XQUAD_ES_CONFIG = DatasetConfig(
    name="xquad-es",
    pretty_name="XQuAD-es",
    source="EuroEval/xquad-es",
    task=RC,
    languages=[SPANISH],
    unofficial=True,
    revision="a43ceae9ba457b920a6392fb07adca88388237aa",
)

BELEBELE_ES_CONFIG = DatasetConfig(
    name="belebele-es",
    pretty_name="Belebele-es",
    source="EuroEval/belebele-es-mini",
    task=MCRC,
    languages=[SPANISH],
    unofficial=True,
    revision="7457f188180dbca56ec36e404fff41c58b070373",
)

MULTI_WIKI_QA_ES_CONFIG = DatasetConfig(
    name="multi-wiki-qa-es",
    pretty_name="MultiWikiQA-es",
    source="EuroEval/multi-wiki-qa-es-mini",
    task=RC,
    languages=[SPANISH],
    unofficial=True,
    revision="12e976eb474d0159933d687ff590f0eea65cd78a",
)

DACSA_ES_CONFIG = DatasetConfig(
    name="dacsa-es",
    pretty_name="DACSA-es",
    source="EuroEval/dacsa-es-mini",
    task=SUMM,
    languages=[SPANISH],
    unofficial=True,
    revision="649f493a19fe401f330acf08283e7a516f38979f",
)

GOLDENSWAG_ES_CONFIG = DatasetConfig(
    name="goldenswag-es",
    pretty_name="GoldenSwag-es",
    source="EuroEval/goldenswag-es-mini",
    task=COMMON_SENSE,
    languages=[SPANISH],
    unofficial=True,
    revision="1ee2ba18d2a14f964d824c8d97fb280306e418c4",
)

MULTINRC_ES_CONFIG = DatasetConfig(
    name="multinrc-es",
    pretty_name="MultiNRC-es",
    source="EuroEval/multinrc-es",
    task=KNOW,
    languages=[SPANISH],
    unofficial=True,
    revision="c5590749d47cd69b7bdcd65de79a1b89a89b1a98",
)
