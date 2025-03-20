"""Templates for the Named Entity Recognition task."""

from ..data_models import PromptConfig
from ..languages import DA, DE, EN, FO, FR, IS, IT, NB, NL, NN, NO, SV
from ..types import TemplateDict

NER_TEMPLATES: TemplateDict = {
    DA: PromptConfig(
        prompt_prefix="Følgende er sætninger og JSON-ordbøger med de navngivne "
        "enheder, som forekommer i den givne sætning.",
        prompt_template="Sætning: {text}\nNavngivne enheder: {label}",
        instruction_prompt="Sætning: {text}\n\nIdentificér de navngivne enheder i "
        "sætningen. Du skal outputte dette som en JSON-ordbog med nøglerne 'person', "
        "'sted', 'organisation' og 'diverse'. Værdierne skal være lister over de "
        "navngivne enheder af den type, præcis som de forekommer i sætningen.",
    ),
    DE: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    EN: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    FO: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    FR: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    IS: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    IT: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    NB: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    NL: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    NN: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    NO: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    SV: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
}

NER_TAG_MAPPING = {
    "per": {
        DA: "person",
        DE: "person",
        EN: "person",
        FO: "persónur",
        FR: "personne",
        IS: "einstaklingur",
        IT: "persona",
        NB: "person",
        NL: "persoon",
        NN: "person",
        NO: "person",
        SV: "person",
    },
    "loc": {
        DA: "sted",
        DE: "ort",
        EN: "location",
        FO: "staður",
        FR: "lieu",
        IS: "staðsetning",
        IT: "posizione",
        NB: "sted",
        NL: "locatie",
        NN: "sted",
        NO: "sted",
        SV: "plats",
    },
    "org": {
        DA: "organisation",
        DE: "organisation",
        EN: "organization",
        FO: "felagsskapur",
        FR: "organisation",
        IS: "stofnun",
        IT: "organizzazione",
        NB: "organisasjon",
        NL: "organisatie",
        NN: "organisasjon",
        NO: "organisasjon",
        SV: "organisation",
    },
    "misc": {
        DA: "diverse",
        DE: "verschiedenes",
        EN: "miscellaneous",
        FO: "ymiskt",
        FR: "divers",
        IS: "ýmislegt",
        IT: "varie",
        NB: "diverse",
        NL: "diversen",
        NN: "diverse",
        NO: "diverse",
        SV: "diverse",
    },
}
