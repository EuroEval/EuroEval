"""All Luxembourgish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import LUXEMBOURGISH
from ..tasks import INSTRUCTION_FOLLOWING, LA, NER, NLI, RC, SENT, TEXT_CLASSIFICATION

# Official datasets ###

LTZGLUE_SA_CONFIG = DatasetConfig(
    name="ltzglue-sa",
    pretty_name="ltzGLUE-SA",
    source="EuroEval/ltzglue-sa",
    task=SENT,
    languages=[LUXEMBOURGISH],
)

LTZGLUE_LA_BINARY_CONFIG = DatasetConfig(
    name="ltzglue-la-binary",
    pretty_name="ltzGLUE-LA",
    source="EuroEval/ltzglue-la",
    task=LA,
    languages=[LUXEMBOURGISH],
)

LTZGLUE_NER_CONFIG = DatasetConfig(
    name="ltzglue-ner",
    pretty_name="ltzGLUE-NER",
    source="EuroEval/ltzglue-ner",
    task=NER,
    languages=[LUXEMBOURGISH],
)

MULTI_WIKI_QA_LB_CONFIG = DatasetConfig(
    name="multi-wiki-qa-lb",
    pretty_name="MultiWikiQA-lb",
    source="EuroEval/multi-wiki-qa-lb-mini",
    task=RC,
    languages=[LUXEMBOURGISH],
)

LTZGLUE_RTE_CONFIG = DatasetConfig(
    name="ltzglue-rte",
    pretty_name="ltzGLUE-RTE",
    source="EuroEval/ltzglue-rte",
    task=NLI,
    languages=[LUXEMBOURGISH],
)

LTZGLUE_HC_CONFIG = DatasetConfig(
    name="ltzglue-hc",
    pretty_name="ltzGLUE-HC",
    source="EuroEval/ltzglue-hc",
    task=TEXT_CLASSIFICATION,
    languages=[LUXEMBOURGISH],
    labels=["no", "yes"],
    prompt_prefix=(
        "Folgend sinn Iwwerschrëften an ob den Artikel dës Behaaptung bestätegt."
    ),
    prompt_template="Iwwerschrëft: {text}\nBestätegt: {label}",
    instruction_prompt="Iwwerschrëft: {text}\n\nBestëmmt ob den Artikel d'Behaaptung "
    "an der Iwwerschrëft bestätegt. Äntwert nëmme mat 'yes' oder 'no'.",
)

LTZGLUE_TC_CONFIG = DatasetConfig(
    name="ltzglue-tc",
    pretty_name="ltzGLUE-TC",
    source="EuroEval/ltzglue-tc",
    task=TEXT_CLASSIFICATION,
    languages=[LUXEMBOURGISH],
    labels=["technology", "business", "culture", "animals", "sports"],
    prompt_prefix="Folgend sinn Noriichten-Artikelen an hir Themen.",
    prompt_template="Artikel: {text}\nThema: {label}",
    instruction_prompt="Artikel: {text}\n\nKlassifizéiert d'Thema vum Artikel. "
    "Äntwert nëmme mat engem vun dësen Etiketten: {labels_str}.",
)

LTZGLUE_ID_CONFIG = DatasetConfig(
    name="ltzglue-id",
    pretty_name="ltzGLUE-ID",
    source="EuroEval/ltzglue-id",
    task=TEXT_CLASSIFICATION,
    languages=[LUXEMBOURGISH],
    train_split=None,
    val_split=None,
    num_few_shot_examples=0,
    labels=[
        "addtoplaylist",
        "alarm cancel alarm",
        "alarm set alarm",
        "alarm show alarms",
        "bookrestaurant",
        "playmusic",
        "ratebook",
        "reminder set reminder",
        "searchcreativework",
        "searchscreeningevent",
        "weather find",
    ],
    prompt_prefix="Folgend si Benotzerufruffen an hir Intentiounen.",
    prompt_template="Ufruff: {text}\nIntentioun: {label}",
    instruction_prompt="Ufruff: {text}\n\nIdentifizéiert d'Intentioun vum Benotzer. "
    "Äntwert nëmme mat engem vun dësen Etiketten: {labels_str}.",
)

MULTI_IFEVAL_LB_CONFIG = DatasetConfig(
    name="multi-ifeval-lb",
    pretty_name="MultiIFEval-lb",
    source="EuroEval/multi-ifeval-lb",
    task=INSTRUCTION_FOLLOWING,
    languages=[LUXEMBOURGISH],
    train_split=None,
    val_split=None,
)

# Unofficial datasets ###

LTZGLUE_LA_MULTI_CONFIG = DatasetConfig(
    name="ltzglue-la-multi",
    pretty_name="ltzGLUE-LA (Multi-class)",
    source="EuroEval/ltzglue-la-multi",
    task=LA,
    languages=[LUXEMBOURGISH],
    unofficial=True,
    labels=["correct", "word_order", "agreement", "morphology", "other"],
    prompt_prefix="Folgend sinn Sätz. Bestëmmt ob se grammatesch korrekt "
    "sinn, oder identifizéiert de Feeler Typ.",
    prompt_template="Saz: {text}\nÄntwert: {label}",
    instruction_prompt="Saz: {text}\n\nBestëmmt ob de Saz grammatesch korrekt ass "
    "(äntwert 'correct'), oder identifizéiert de Feeler Typ: 'word_order' (falsch "
    "Wuert-Reiefolleg), 'agreement' (Subject-Verb oder Determiner-Noun Stëmmung net "
    "korrekt), 'morphology' (falsch Wortform), oder 'other'. Äntwert nëmme mat engem "
    "vun dësen Etiketten: {labels_str}.",
)
