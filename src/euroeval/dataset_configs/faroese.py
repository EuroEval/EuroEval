"""All Faroese dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import FAROESE
from ..tasks import GED, HALLU, INSTRUCTION_FOLLOWING, KNOW, LA, LOGIC, NER, RC, SENT

# Official datasets ###

FOSENT_CONFIG = DatasetConfig(
    name="fosent",
    pretty_name="FoSent",
    source="EuroEval/fosent",
    task=SENT,
    languages=[FAROESE],
    num_few_shot_examples=5,
)

SCALA_FO_CONFIG = DatasetConfig(
    name="scala-fo",
    pretty_name="ScaLA-fo",
    source="EuroEval/scala-fo",
    task=LA,
    languages=[FAROESE],
)

FONE_CONFIG = DatasetConfig(
    name="fone",
    pretty_name="FoNE",
    source="EuroEval/fone-mini",
    task=NER,
    languages=[FAROESE],
)

FOQA_CONFIG = DatasetConfig(
    name="foqa",
    pretty_name="FoQA",
    source="EuroEval/foqa",
    task=RC,
    languages=[FAROESE],
)

MULTI_IFEVAL_FO_CONFIG = DatasetConfig(
    name="multi-ifeval-fo",
    pretty_name="MultiIFEval-fo",
    source="EuroEval/multi-ifeval-fo",
    task=INSTRUCTION_FOLLOWING,
    languages=[FAROESE],
    train_split=None,
    val_split=None,
)

ZEBRA_PUZZLE_EASY_FO_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-fo",
    pretty_name="ZebraPuzzlesEasy-fo",
    source="EuroEval/zebra-puzzles-easy-fo",
    task=LOGIC,
    languages=[FAROESE],
)

RAGTRUTH_FO_CONFIG = DatasetConfig(
    name="ragtruth-fo",
    pretty_name="RAGTruth-fo",
    source="EuroEval/ragtruth-translated-hallucinations-fo-mini",
    task=HALLU,
    languages=[FAROESE],
    train_split=None,
)


# Unofficial datasets ###

FAROESE_SEMANTIC_RELATIONS_CONFIG = DatasetConfig(
    name="faroese-semantic-relations",
    pretty_name="Faroese Semantic Relations",
    source="EuroEval/faroese-semantic-relations",
    task=KNOW,
    languages=[FAROESE],
    labels=["a", "b", "c", "d", "e", "f"],
    unofficial=True,
)

WIKIANN_FO_CONFIG = DatasetConfig(
    name="wikiann-fo",
    pretty_name="WikiANN-fo",
    source="EuroEval/wikiann-fo-mini",
    task=NER,
    languages=[FAROESE],
    unofficial=True,
)

MULTI_WIKI_QA_FO_CONFIG = DatasetConfig(
    name="multi-wiki-qa-fo",
    pretty_name="MultiWikiQA-fo",
    source="EuroEval/multi-wiki-qa-fo-mini",
    task=RC,
    languages=[FAROESE],
    unofficial=True,
)

GERLANGMOD_FO_CONFIG = DatasetConfig(
    name="gerlangmod-fo",
    pretty_name="GerLangMod-fo",
    source="EuroEval/gerlangmod-fo",
    task=GED,
    languages=[FAROESE],
    unofficial=True,
)

ZEBRA_PUZZLE_HARD_FO_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-fo",
    pretty_name="ZebraPuzzlesHard-fo",
    source="EuroEval/zebra-puzzles-hard-fo",
    task=LOGIC,
    languages=[FAROESE],
    unofficial=True,
)
