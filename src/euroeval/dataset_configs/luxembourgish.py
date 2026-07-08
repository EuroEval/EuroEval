"""All Luxembourgish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import LUXEMBOURGISH
from ..tasks import LA, NER, NLI, RC, SENT, TEXT_CLASSIFICATION

# Official datasets ###

LTZGLUE_SA_CONFIG = DatasetConfig(
    name="ltzglue-sa",
    pretty_name="ltzGLUE-SA",
    source="EuroEval/ltzglue-sa",
    task=SENT,
    languages=[LUXEMBOURGISH],
    labels=["negative", "neutral", "positive"],
    prompt_prefix="Fir dësen Text ass uginn ob de Sentiment negativ, neutral oder "
    "positiv ass.",
    prompt_template="Text: {text}\nSentiment: {label}",
    instruction_prompt="Text: {text}\n\nBestëmmt de Sentiment vum Text. Äntwert "
    "nëmme mat 'negativ', 'neutral' oder 'positiv'.",
    prompt_label_mapping="auto",
)

LTZGLUE_LA_BINARY_CONFIG = DatasetConfig(
    name="ltzglue-la-binary",
    pretty_name="ltzGLUE-LA",
    source="EuroEval/ltzglue-la",
    task=LA,
    languages=[LUXEMBOURGISH],
    labels=["correct", "incorrect"],
)

LTZGLUE_NER_CONFIG = DatasetConfig(
    name="ltzglue-ner",
    pretty_name="ltzGLUE-NER",
    source="EuroEval/ltzglue-ner",
    task=NER,
    languages=[LUXEMBOURGISH],
    labels=[
        "o",
        "b-per",
        "i-per",
        "b-loc",
        "i-loc",
        "b-org",
        "i-org",
        "b-misc",
        "i-misc",
    ],
    prompt_prefix="Fir dëse Satz gëtt et eng JSON-Liste mat den Entitéiten.",
    prompt_template="Saz: {text}\nEntitéiten: {label}",
    instruction_prompt="Saz: {text}\n\nIdentifizéiert déi benannt Entitéiten am Satz. "
    "Output als JSON-Liste mat den Schlësselen 'persoun', 'plaz', "
    "'organisatioun' an 'divers'.",
    prompt_label_mapping={
        "b-per": "persoun",
        "i-per": "persoun",
        "b-loc": "plaz",
        "i-loc": "plaz",
        "b-org": "organisatioun",
        "i-org": "organisatioun",
        "b-misc": "divers",
        "i-misc": "divers",
    },
)

MULTI_WIKI_QA_LB_CONFIG = DatasetConfig(
    name="multi-wiki-qa-lb",
    pretty_name="MultiWikiQA-lb",
    source="EuroEval/multi-wiki-qa-lb",
    task=RC,
    languages=[LUXEMBOURGISH],
    labels=["answer"],
    prompt_prefix="Fir dës Fro gëtt et eng Äntwert.",
    prompt_template="Text: {text}\nFro: {question}\nÄntwert: {label}",
    instruction_prompt="Text: {text}\nFro: {question}\n\nBeäntwert d'Fro "
    "baséiert op dem Text.",
    prompt_label_mapping="auto",
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
    prompt_label_mapping="auto",
)

LTZGLUE_RTE_CONFIG = DatasetConfig(
    name="ltzglue-rte",
    pretty_name="ltzGLUE-RTE",
    source="EuroEval/ltzglue-rte",
    task=NLI,
    languages=[LUXEMBOURGISH],
    labels=["entailment", "neutral", "contradiction"],
    prompt_prefix="Fir dës Puer ass uginn ob d'Hypothees d'Premisse follegt "
    "oder widderleet.",
    prompt_template="Text: {text}\nRelatioun: {label}",
    instruction_prompt="Text: {text}\n\nBestëmmt ob déi zweet Aussag aus der "
    "éischter follegt, hir widdersträit, oder keng logesch Verbindung mat hir huet. "
    "Äntwert mat 'folgerung', 'neutral', oder 'widdersträit'.",
    prompt_label_mapping={
        "entailment": "folgerung",
        "neutral": "neutral",
        "contradiction": "widdersträit",
    },
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
    prompt_label_mapping="auto",
)

LTZGLUE_ID_CONFIG = DatasetConfig(
    name="ltzglue-id",
    pretty_name="ltzGLUE-ID",
    source="EuroEval/ltzglue-id",
    task=TEXT_CLASSIFICATION,
    languages=[LUXEMBOURGISH],
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
    prompt_label_mapping="auto",
)

# Unofficial datasets ###

LTZGLUE_LA_MULTI_CONFIG = DatasetConfig(
    name="ltzglue-la-multi",
    pretty_name="ltzGLUE-LA (Multi-class)",
    source="EuroEval/ltzglue-la-multi",
    task=TEXT_CLASSIFICATION,
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
    prompt_label_mapping="auto",
)
