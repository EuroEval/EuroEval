"""All Faroese dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, FAROESE
from ..tasks import (
    GEC,
    GED,
    HALLU,
    INSTRUCTION_FOLLOWING,
    KNOW,
    LA,
    LOGIC,
    NER,
    RC,
    SENT,
    TOOL_CALLING,
    TRANSLATION,
)

# Official datasets ###

FLORES_EN_FO_CONFIG = TranslationDatasetConfig(
    name="flores-en-fo",
    pretty_name="FLORES-en-fo",
    source="EuroEval/flores-en-fo",
    task=TRANSLATION,
    languages=[FAROESE],
    source_language=ENGLISH,
    target_language=FAROESE,
    revision="65f72c2e17e9a698358900af17bd988b837be311",
)

FOSENT_CONFIG = DatasetConfig(
    name="fosent",
    pretty_name="FoSent",
    source="EuroEval/fosent",
    task=SENT,
    languages=[FAROESE],
    num_few_shot_examples=5,
    revision="d57f23c500505d32c298a555fbf9fa68c358b091",
)

SCALA_FO_CONFIG = DatasetConfig(
    name="scala-fo",
    pretty_name="ScaLA-fo",
    source="EuroEval/scala-fo",
    task=LA,
    languages=[FAROESE],
    revision="589bf8e142baa31881ea2f670c94acde1b4c5f5e",
)

FONE_CONFIG = DatasetConfig(
    name="fone",
    pretty_name="FoNE",
    source="EuroEval/fone-mini",
    task=NER,
    languages=[FAROESE],
    revision="215ac623bc30c385534174f0ee9446879d4cdb95",
)

FOQA_CONFIG = DatasetConfig(
    name="foqa",
    pretty_name="FoQA",
    source="EuroEval/foqa",
    task=RC,
    languages=[FAROESE],
    revision="3746906f81353d29c04214869c1e6a2e87ad35de",
)

MULTI_IFEVAL_FO_CONFIG = DatasetConfig(
    name="multi-ifeval-fo",
    pretty_name="MultiIFEval-fo",
    source="EuroEval/multi-ifeval-fo",
    task=INSTRUCTION_FOLLOWING,
    languages=[FAROESE],
    train_split=None,
    val_split=None,
    revision="38b3a0020bbc91c78678e260ffaaecd067d15751",
)

ZEBRA_PUZZLE_EASY_FO_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-fo",
    pretty_name="ZebraPuzzlesEasy-fo",
    source="EuroEval/zebra-puzzles-easy-fo",
    task=LOGIC,
    languages=[FAROESE],
    revision="e10652b11573990037c3fdb7b1e14eb5f4c79c94",
)

RAGTRUTH_FO_CONFIG = DatasetConfig(
    name="ragtruth-fo",
    pretty_name="RAGTruth-fo",
    source="EuroEval/ragtruth-translated-hallucinations-fo-mini",
    task=HALLU,
    languages=[FAROESE],
    train_split=None,
    revision="1f72f24ac1195b647b2a2e8b2f657d759bc19ac4",
)


FAROESE_GRAMMATICAL_CORRECTNESS_CONFIG = DatasetConfig(
    name="faroese-grammatical-correctness",
    pretty_name="Faroese Grammatical Correctness",
    source="EuroEval/faroese-grammatical-correctness",
    task=GEC,
    languages=[FAROESE],
    revision="3c5e305ebedf8091c1cb79ca2039c6c1637cfbe0",
)

FAROESE_SEMANTIC_RELATIONS_CONFIG = DatasetConfig(
    name="faroese-semantic-relations",
    pretty_name="Faroese Semantic Relations",
    source="EuroEval/faroese-semantic-relations",
    task=KNOW,
    languages=[FAROESE],
    labels=["a", "b", "c", "d", "e", "f"],
    revision="49c2cd3bb2d4158cbf49eb2afee54fbe0520921b",
)

FAROESE_METAPHORICAL_EXPLANATIONS_CONFIG = DatasetConfig(
    name="faroese-metaphorical-explanations",
    pretty_name="Faroese Metaphorical Explanations",
    source="EuroEval/faroese-metaphorical-explanations",
    task=KNOW,
    languages=[FAROESE],
    revision="c5eb404c5c63adaa755da968aa2fcd0657f91c50",
)


# Unofficial datasets ###

MULTI_BFCL_FO_CONFIG = DatasetConfig(
    name="multi-bfcl-fo",
    pretty_name="MultiBFCL-fo",
    source="EuroEval/multi-bfcl-fo-mini",
    task=TOOL_CALLING,
    languages=[FAROESE],
    train_split=None,
    revision="a204fd503b28a5bc522a659e0fa2b72b61c11c2e",
)


FLORES_FO_EN_CONFIG = TranslationDatasetConfig(
    name="flores-fo-en",
    pretty_name="FLORES-fo-en",
    source="EuroEval/flores-fo-en",
    task=TRANSLATION,
    languages=[FAROESE],
    source_language=FAROESE,
    target_language=ENGLISH,
    unofficial=True,
    revision="867819d3af6b9cfbbdac6814d024aa3e6d781173",
)

WIKIANN_FO_CONFIG = DatasetConfig(
    name="wikiann-fo",
    pretty_name="WikiANN-fo",
    source="EuroEval/wikiann-fo-mini",
    task=NER,
    languages=[FAROESE],
    unofficial=True,
    revision="564cc836587aa94894fffba0ed28857eb5b6f939",
)

MULTI_WIKI_QA_FO_CONFIG = DatasetConfig(
    name="multi-wiki-qa-fo",
    pretty_name="MultiWikiQA-fo",
    source="EuroEval/multi-wiki-qa-fo-mini",
    task=RC,
    languages=[FAROESE],
    unofficial=True,
    revision="f04531a36a0e9bd0641c456f86a36e87ad9cc1c8",
)

GERLANGMOD_FO_CONFIG = DatasetConfig(
    name="gerlangmod-fo",
    pretty_name="GerLangMod-fo",
    source="EuroEval/gerlangmod-fo",
    task=GED,
    languages=[FAROESE],
    unofficial=True,
    revision="028aa500e50c2d0e28fcce1fca4db698bd86aef8",
)

ZEBRA_PUZZLE_HARD_FO_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-fo",
    pretty_name="ZebraPuzzlesHard-fo",
    source="EuroEval/zebra-puzzles-hard-fo",
    task=LOGIC,
    languages=[FAROESE],
    unofficial=True,
    revision="bce38bb07e111f66c8862bd1b2a843f8f02bf493",
)
