"""All Dutch dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import DUTCH, ENGLISH
from ..tasks import (
    COMMON_SENSE,
    EUROPEAN_VALUES,
    GED,
    HALLU,
    INSTRUCTION_FOLLOWING,
    KNOW,
    LA,
    LOGIC,
    MCRC,
    MCSTEREO,
    NER,
    NLI,
    RC,
    SENT,
    SIMPL,
    SUMM,
    TOOL_CALLING,
    TRANSLATION,
)

# Official datasets ###

WMT24PP_EN_NL_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-nl",
    pretty_name="WMT24++-en-nl",
    source="EuroEval/wmt24pp-en-nl",
    task=TRANSLATION,
    languages=[DUTCH],
    source_language=ENGLISH,
    target_language=DUTCH,
    revision="3c555800afec97095504aa073ae9f9f40f0066a0",
)

FLORES_EN_NL_CONFIG = TranslationDatasetConfig(
    name="flores-en-nl",
    pretty_name="FLORES-en-nl",
    source="EuroEval/flores-en-nl",
    task=TRANSLATION,
    languages=[DUTCH],
    source_language=ENGLISH,
    target_language=DUTCH,
    revision="640eb2623f042509a05fdc98615e7391e8f695ee",
)

DBRD_CONFIG = DatasetConfig(
    name="dbrd",
    pretty_name="DBRD",
    source="EuroEval/dbrd-mini",
    task=SENT,
    languages=[DUTCH],
    labels=["negative", "positive"],
    revision="1e8ed43db8117b4fc5ab495ef92b52626813cda4",
)

DUTCH_COLA_CONFIG = DatasetConfig(
    name="dutch-cola",
    pretty_name="Dutch CoLA",
    source="EuroEval/dutch-cola",
    task=LA,
    languages=[DUTCH],
    revision="31c590b928f49745b7720de5cc93e3f06f7ee25f",
)

CONLL_NL_CONFIG = DatasetConfig(
    name="conll-nl",
    pretty_name="CoNLL-nl",
    source="EuroEval/conll-nl-mini",
    task=NER,
    languages=[DUTCH],
    revision="8b3624fc6bf6438c4ab30632ce4d466d6b09ab25",
)

SQUAD_NL_CONFIG = DatasetConfig(
    name="squad-nl",
    pretty_name="SQuAD-nl",
    source="EuroEval/squad-nl-v2-mini",
    task=RC,
    languages=[DUTCH],
    revision="44a840957f9a2d020c1a32938bd55a7c4bd1c718",
)

WIKI_LINGUA_NL_CONFIG = DatasetConfig(
    name="wiki-lingua-nl",
    pretty_name="WikiLingua-nl",
    source="EuroEval/wiki-lingua-nl-mini",
    task=SUMM,
    languages=[DUTCH],
    revision="0c2c7bf15cefadf521d26cae4012ea13cbea29e5",
)

DUIDELIJKE_TAAL_NL_CONFIG = DatasetConfig(
    name="duidelijke-taal",
    pretty_name="Duidelijke Taal",
    source="EuroEval/duidelijke-taal",
    task=SIMPL,
    languages=[DUTCH],
    revision="d8cedb04e85f0402c4bfaa6c7cefe28440108ffa",
)

VALEU_NL_CONFIG = DatasetConfig(
    name="valeu-nl",
    pretty_name="VaLEU-nl",
    source="EuroEval/european-values-nl",
    task=EUROPEAN_VALUES,
    languages=[DUTCH],
    train_split=None,
    val_split=None,
    bootstrap_samples=False,
    instruction_prompt="{text}",
    revision="ab426347018a8604b59f69d03b06b3037e6f4c76",
)

MULTI_IFEVAL_NL_CONFIG = DatasetConfig(
    name="multi-ifeval-nl",
    pretty_name="MultiIFEval-nl",
    source="EuroEval/multi-ifeval-nl",
    task=INSTRUCTION_FOLLOWING,
    languages=[DUTCH],
    train_split=None,
    val_split=None,
    revision="086d8549cc9541fbda30b5fb88800a7c29e1abbc",
)

MBBQ_NL_CONFIG = DatasetConfig(
    name="mbbq-nl",
    pretty_name="MBBQ-nl",
    source="EuroEval/mbbq-nl",
    task=MCSTEREO,
    languages=[DUTCH],
    train_split=None,
    revision="ee037402655e1980748224ae5320927ff3cdf509",
)

ZEBRA_PUZZLE_EASY_NL_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-nl",
    pretty_name="ZebraPuzzlesEasy-nl",
    source="EuroEval/zebra-puzzles-easy-nl",
    task=LOGIC,
    languages=[DUTCH],
    revision="0220403610e08c6803601bf3b45125de5cd27e54",
)

RAGTRUTH_NL_CONFIG = DatasetConfig(
    name="ragtruth-nl",
    pretty_name="RAGTruth-nl",
    source="EuroEval/ragtruth-translated-hallucinations-nl-mini",
    task=HALLU,
    languages=[DUTCH],
    train_split=None,
    revision="6b8bdd5b80182fa427f65ad73d8682cf0f3d1749",
)


INCLUDE_NL_CONFIG = DatasetConfig(
    name="include-nl",
    pretty_name="INCLUDE-nl",
    source="EuroEval/include-nl-mini",
    task=KNOW,
    languages=[DUTCH],
    revision="326c7e1ba91f101650bfe4d3d1df1e0174a69fca",
)

MULTILOKO_NL_CONFIG = DatasetConfig(
    name="multiloko-nl",
    pretty_name="MultiLoKo-nl",
    source="EuroEval/multiloko-nl-mini",
    task=KNOW,
    languages=[DUTCH],
    val_split=None,
    revision="4998998637b36809c7594c17d9bd399cfcb23ec3",
)

WINOGRANDE_NL_CONFIG = DatasetConfig(
    name="winogrande-nl",
    pretty_name="Winogrande-nl",
    source="EuroEval/winogrande-nl",
    task=COMMON_SENSE,
    languages=[DUTCH],
    labels=["a", "b"],
    revision="ce62fbe0de33d6a365d9d04d4bcdfd55b88a09cf",
)


# Unofficial datasets ###

MULTI_BFCL_NL_CONFIG = DatasetConfig(
    name="multi-bfcl-nl",
    pretty_name="MultiBFCL-nl",
    source="EuroEval/multi-bfcl-nl-mini",
    task=TOOL_CALLING,
    languages=[DUTCH],
    train_split=None,
    revision="393339b3a612b251e1147204e3c881198aeb7303",
)


WMT24PP_NL_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-nl-en",
    pretty_name="WMT24++-nl-en",
    source="EuroEval/wmt24pp-nl-en",
    task=TRANSLATION,
    languages=[DUTCH],
    source_language=DUTCH,
    target_language=ENGLISH,
    unofficial=True,
    revision="3ad45b7165ccf1a552c8d47c596f35f0afc7aff1",
)

FLORES_NL_EN_CONFIG = TranslationDatasetConfig(
    name="flores-nl-en",
    pretty_name="FLORES-nl-en",
    source="EuroEval/flores-nl-en",
    task=TRANSLATION,
    languages=[DUTCH],
    source_language=DUTCH,
    target_language=ENGLISH,
    unofficial=True,
    revision="2dedecfc879cecba3357bd99ecf6e053948a9252",
)

HELLASWAG_NL_CONFIG = DatasetConfig(
    name="hellaswag-nl",
    pretty_name="HellaSwag-nl",
    source="EuroEval/hellaswag-nl-mini",
    task=COMMON_SENSE,
    languages=[DUTCH],
    unofficial=True,
    revision="de22490d8479eee815216b7b0c12b44a36877679",
)

MMLU_NL_CONFIG = DatasetConfig(
    name="mmlu-nl",
    pretty_name="MMLU-nl",
    source="EuroEval/mmlu-nl-mini",
    task=KNOW,
    languages=[DUTCH],
    unofficial=True,
    revision="65eb0e893587a67eda575ac830d3b3b84d5a2f44",
)

SCALA_NL_CONFIG = DatasetConfig(
    name="scala-nl",
    pretty_name="ScaLA-nl",
    source="EuroEval/scala-nl",
    task=LA,
    languages=[DUTCH],
    unofficial=True,
    revision="f5b7b376fc2d5ae7254a34bb6b6e297c357841a4",
)

BLIMP_NL_CONFIG = DatasetConfig(
    name="blimp-nl",
    pretty_name="BLiMP-NL-mini",
    source="EuroEval/blimp-nl-mini",
    task=LA,
    languages=[DUTCH],
    unofficial=True,
)

DUTCH_COLA_FULL_CONFIG = DatasetConfig(
    name="dutch-cola-full",
    pretty_name="Dutch CoLA Full",
    source="EuroEval/dutch-cola-full",
    task=LA,
    languages=[DUTCH],
    unofficial=True,
    revision="56081cdb317ea15214ae1a4dd0e1ce020237a31d",
)

DUTCH_PROVERBS_CONFIG = DatasetConfig(
    name="dutch-proverbs",
    pretty_name="Dutch Proverbs",
    source="EuroEval/dutch-proverbs",
    task=KNOW,
    languages=[DUTCH],
    labels=["a", "b"],
    val_split=None,
    unofficial=True,
    revision="1f355e5804cba11f377a71e3476c53f8cfd87818",
)

ARC_NL_CONFIG = DatasetConfig(
    name="arc-nl",
    pretty_name="ARC-nl",
    source="EuroEval/arc-nl-mini",
    task=KNOW,
    languages=[DUTCH],
    unofficial=True,
    revision="acef2ce9afe4a385a9d94f378d155bf78e203f54",
)

BELEBELE_NL_CONFIG = DatasetConfig(
    name="belebele-nl",
    pretty_name="Belebele-nl",
    source="EuroEval/belebele-nl-mini",
    task=MCRC,
    languages=[DUTCH],
    unofficial=True,
    revision="dd7540e8291e838150a8d3d8d1d80b68b5f69a6f",
)

MULTI_WIKI_QA_NL_CONFIG = DatasetConfig(
    name="multi-wiki-qa-nl",
    pretty_name="MultiWikiQA-nl",
    source="EuroEval/multi-wiki-qa-nl-mini",
    task=RC,
    languages=[DUTCH],
    unofficial=True,
    revision="4c37a46962ee15e13f6885002a9f963060a53a66",
)

COPA_NL_CONFIG = DatasetConfig(
    name="copa-nl",
    pretty_name="COPA-nl",
    source="EuroEval/copa-nl",
    task=COMMON_SENSE,
    languages=[DUTCH],
    unofficial=True,
    labels=["a", "b"],
    revision="7b45234c7776a1a2591eb2d7d1fe1cbc3a004265",
)

GOLDENSWAG_NL_CONFIG = DatasetConfig(
    name="goldenswag-nl",
    pretty_name="GoldenSwag-nl",
    source="EuroEval/goldenswag-nl-mini",
    task=COMMON_SENSE,
    languages=[DUTCH],
    unofficial=True,
    revision="4e4c688b2e5b3324d69d17b9111e339f3cfe53eb",
)

GERLANGMOD_NL_CONFIG = DatasetConfig(
    name="gerlangmod-nl",
    pretty_name="GerLangMod-nl",
    source="EuroEval/gerlangmod-nl",
    task=GED,
    languages=[DUTCH],
    unofficial=True,
    revision="3ddbb367f8ddb611e51856a1ea6fdfbbdc74c8c1",
)

SICK_NL_CONFIG = DatasetConfig(
    name="sick-nl",
    pretty_name="The Dutch SICK-NL Entailment Dataset",
    source="EuroEval/sick-nl",
    task=NLI,
    languages=[DUTCH],
    unofficial=True,
    revision="a28162feda3ec4888edcdff0fce8ec8b87da48ee",
)

ZEBRA_PUZZLE_HARD_NL_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-nl",
    pretty_name="ZebraPuzzlesHard-nl",
    source="EuroEval/zebra-puzzles-hard-nl",
    task=LOGIC,
    languages=[DUTCH],
    unofficial=True,
    revision="038a85b0e2a62ca692b3fd3131cafce85b265d9c",
)

EU_MMLU_NL_CONFIG = DatasetConfig(
    name="eu-mmlu-nl",
    pretty_name="EU-MMLU-nl",
    source="EuroEval/eu-mmlu-nl",
    task=KNOW,
    languages=[DUTCH],
    unofficial=True,
    revision="c89814371cf1bd601061331226891d7e34ebddd2",
)

MKQA_NL_CONFIG = DatasetConfig(
    name="mkqa-nl",
    pretty_name="MKQA-nl",
    source="EuroEval/mkqa-nl-mini",
    task=KNOW,
    languages=[DUTCH],
    unofficial=True,
    revision="0464f7cbfe10e791e25a8e6253a07666fec26a86",
)
