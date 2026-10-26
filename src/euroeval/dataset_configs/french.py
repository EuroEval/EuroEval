"""All French dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, FRENCH
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

WMT24PP_EN_FR_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-fr",
    pretty_name="WMT24++-en-fr",
    source="EuroEval/wmt24pp-en-fr",
    task=TRANSLATION,
    languages=[FRENCH],
    source_language=ENGLISH,
    target_language=FRENCH,
    revision="8299dfc8cc17c5de1c4499b36372b1905c078045",
)

FLORES_EN_FR_CONFIG = TranslationDatasetConfig(
    name="flores-en-fr",
    pretty_name="FLORES-en-fr",
    source="EuroEval/flores-en-fr",
    task=TRANSLATION,
    languages=[FRENCH],
    source_language=ENGLISH,
    target_language=FRENCH,
    revision="0e667de7296a5d438a8f1610553c635ff02fa9de",
)

ALLOCINE_CONFIG = DatasetConfig(
    name="allocine",
    pretty_name="AlloCiné",
    source="EuroEval/allocine-mini",
    task=SENT,
    languages=[FRENCH],
    labels=["negative", "positive"],
    prompt_label_mapping=dict(positive="positif", negative="négatif"),
    revision="f8b3491d397993b09647f836a8ba5ca364854255",
)

SCALA_FR_CONFIG = DatasetConfig(
    name="scala-fr",
    pretty_name="ScaLA-fr",
    source="EuroEval/scala-fr",
    task=LA,
    languages=[FRENCH],
    revision="5af76237659d708f160a7ef65b459c06b68352fa",
)

ELTEC_CONFIG = DatasetConfig(
    name="eltec",
    pretty_name="ELTeC",
    source="EuroEval/eltec-mini",
    task=NER,
    languages=[FRENCH],
    revision="8c6211cd0273d65f794658fd58d95569adae63a4",
)

FQUAD_CONFIG = DatasetConfig(
    name="fquad",
    pretty_name="FQuAD",
    source="EuroEval/fquad-mini",
    task=RC,
    languages=[FRENCH],
    revision="63d5707cb93fc76edc709b399c5537663baa20df",
)

ORANGE_SUM_CONFIG = DatasetConfig(
    name="orange-sum",
    pretty_name="OrangeSum",
    source="EuroEval/orange-sum-mini",
    task=SUMM,
    languages=[FRENCH],
    revision="b0c4cfb276542881793f81a7ddf9f71921402d58",
)

HELLASWAG_FR_CONFIG = DatasetConfig(
    name="hellaswag-fr",
    pretty_name="HellaSwag-fr",
    source="EuroEval/hellaswag-fr-mini",
    task=COMMON_SENSE,
    languages=[FRENCH],
    revision="22ce3fbeaebca06f2eec93ebec0cb5f3b56dd79b",
)

IFEVAL_FR_CONFIG = DatasetConfig(
    name="ifeval-fr",
    pretty_name="IFEval-fr",
    source="EuroEval/ifeval-fr",
    task=INSTRUCTION_FOLLOWING,
    languages=[FRENCH],
    train_split=None,
    val_split=None,
    revision="6d6314bfad0e6f6dc50549f5ce1b28678cfa8edb",
)

VALEU_FR_CONFIG = DatasetConfig(
    name="valeu-fr",
    pretty_name="VaLEU-fr",
    source="EuroEval/european-values-fr",
    task=EUROPEAN_VALUES,
    languages=[FRENCH],
    train_split=None,
    val_split=None,
    bootstrap_samples=False,
    instruction_prompt="{text}",
    revision="6fc80d88e79f475f04ca5eeba706294adc82229f",
)

RAGTRUTH_FR_CONFIG = DatasetConfig(
    name="ragtruth-fr",
    pretty_name="RAGTruth-fr",
    source="EuroEval/ragtruth-translated-hallucinations-fr-mini",
    task=HALLU,
    languages=[FRENCH],
    train_split=None,
    revision="164f0faaa60de1212ab3722bc857884d041a90a6",
)


INCLUDE_FR_CONFIG = DatasetConfig(
    name="include-fr",
    pretty_name="INCLUDE-fr",
    source="EuroEval/include-fr-mini",
    task=KNOW,
    languages=[FRENCH],
    revision="ed08b6d79135dba0956d344997779d844922819f",
)

MULTILOKO_FR_CONFIG = DatasetConfig(
    name="multiloko-fr",
    pretty_name="MultiLoKo-fr",
    source="EuroEval/multiloko-fr-mini",
    task=KNOW,
    languages=[FRENCH],
    val_split=None,
    revision="a1b65134bc32377e8bb11f8834b88c846bc245c4",
)


# Unofficial datasets ###

MULTI_BFCL_FR_CONFIG = DatasetConfig(
    name="multi-bfcl-fr",
    pretty_name="MultiBFCL-fr",
    source="EuroEval/multi-bfcl-fr-mini",
    task=TOOL_CALLING,
    languages=[FRENCH],
    train_split=None,
    revision="a8fd89babdc446c53dfde2a2c1aa3269c8eb1026",
)


WMT24PP_FR_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-fr-en",
    pretty_name="WMT24++-fr-en",
    source="EuroEval/wmt24pp-fr-en",
    task=TRANSLATION,
    languages=[FRENCH],
    source_language=FRENCH,
    target_language=ENGLISH,
    unofficial=True,
    revision="8f6e245e644db8c5c7ccc3a76c2377dc9a6f5644",
)

FLORES_FR_EN_CONFIG = TranslationDatasetConfig(
    name="flores-fr-en",
    pretty_name="FLORES-fr-en",
    source="EuroEval/flores-fr-en",
    task=TRANSLATION,
    languages=[FRENCH],
    source_language=FRENCH,
    target_language=ENGLISH,
    unofficial=True,
    revision="f7397c5e558cfa9731308512634a4c611334bb55",
)

MMLU_FR_CONFIG = DatasetConfig(
    name="mmlu-fr",
    pretty_name="MMLU-fr",
    source="EuroEval/mmlu-fr-mini",
    task=KNOW,
    languages=[FRENCH],
    unofficial=True,
    revision="abe6e95e8abe841f7e627130baa76eca8ac63228",
)

MULTI_IFEVAL_FR_CONFIG = DatasetConfig(
    name="multi-ifeval-fr",
    pretty_name="MultiIFEval-fr",
    source="EuroEval/multi-ifeval-fr",
    task=INSTRUCTION_FOLLOWING,
    languages=[FRENCH],
    train_split=None,
    val_split=None,
    unofficial=True,
    revision="e08bd91eac4a7dd2819f8eb6d7ec477a8f028207",
)

BELEBELE_FR_CONFIG = DatasetConfig(
    name="belebele-fr",
    pretty_name="Belebele-fr",
    source="EuroEval/belebele-fr-mini",
    task=MCRC,
    languages=[FRENCH],
    unofficial=True,
    revision="4f76b1a7d25a638725d6dd4ab9377c70ed1bd831",
)

MULTI_WIKI_QA_FR_CONFIG = DatasetConfig(
    name="multi-wiki-qa-fr",
    pretty_name="MultiWikiQA-fr",
    source="EuroEval/multi-wiki-qa-fr-mini",
    task=RC,
    languages=[FRENCH],
    unofficial=True,
    revision="c247455330d14fa05d9f941c045f98374af751ea",
)

GOLDENSWAG_FR_CONFIG = DatasetConfig(
    name="goldenswag-fr",
    pretty_name="GoldenSwag-fr",
    source="EuroEval/goldenswag-fr-mini",
    task=COMMON_SENSE,
    languages=[FRENCH],
    unofficial=True,
    revision="b5614241765d3fb977e5d3ccc6d7640cce2d7124",
)

WINOGRANDE_FR_CONFIG = DatasetConfig(
    name="winogrande-fr",
    pretty_name="Winogrande-fr",
    source="EuroEval/winogrande-fr",
    task=COMMON_SENSE,
    languages=[FRENCH],
    labels=["a", "b"],
    unofficial=True,
    revision="65f72e13d6aaffe7c209de3277ef858f31ebb02b",
)

MULTINRC_FR_CONFIG = DatasetConfig(
    name="multinrc-fr",
    pretty_name="MultiNRC-fr",
    source="EuroEval/multinrc-fr",
    task=KNOW,
    languages=[FRENCH],
    unofficial=True,
    revision="d6f0fa0d1e9da368bd925ad86745eb3a19ce6583",
)

EU_MMLU_FR_CONFIG = DatasetConfig(
    name="eu-mmlu-fr",
    pretty_name="EU-MMLU-fr",
    source="EuroEval/eu-mmlu-fr",
    task=KNOW,
    languages=[FRENCH],
    unofficial=True,
    revision="1d846fb187b119d1e53b2a562041e94c11417e57",
)
