"""Constants used for the template configurations."""

NER_TAG_MAPPING = {
    "per": {
        "da": "person",
        "de": "person",
        "en": "person",
        "es": "persona",
        "fo": "persónur",
        "fr": "personne",
        "is": "einstaklingur",
        "it": "persona",
        "nb": "person",
        "nl": "persoon",
        "nn": "person",
        "no": "person",
        "sv": "person",
    },
    "loc": {
        "da": "sted",
        "de": "ort",
        "en": "location",
        "es": "lugar",
        "fo": "staður",
        "fr": "lieu",
        "is": "staðsetning",
        "it": "posizione",
        "nb": "sted",
        "nl": "locatie",
        "nn": "sted",
        "no": "sted",
        "sv": "plats",
    },
    "org": {
        "da": "organisation",
        "de": "organisation",
        "en": "organization",
        "es": "organización",
        "fo": "felagsskapur",
        "fr": "organisation",
        "is": "stofnun",
        "it": "organizzazione",
        "nb": "organisasjon",
        "nl": "organisatie",
        "nn": "organisasjon",
        "no": "organisasjon",
        "sv": "organisation",
    },
    "misc": {
        "da": "diverse",
        "de": "verschiedenes",
        "en": "miscellaneous",
        "es": "misceláneo",
        "fo": "ymiskt",
        "fr": "divers",
        "is": "ýmislegt",
        "it": "varie",
        "nb": "diverse",
        "nl": "diversen",
        "nn": "diverse",
        "no": "diverse",
        "sv": "diverse",
    },
}

LA_LABEL_MAPPING = {
    "correct": {
        "da": "ja",
        "de": "ja",
        "en": "yes",
        "es": "sí",
        "fo": "ja",
        "fr": "oui",
        "is": "já",
        "it": "si",
        "nb": "ja",
        "nl": "ja",
        "nn": "ja",
        "no": "ja",
        "sv": "ja",
    },
    "incorrect": {
        "da": "nej",
        "de": "nein",
        "en": "no",
        "es": "no",
        "fo": "nei",
        "fr": "non",
        "is": "nei",
        "it": "no",
        "nb": "nei",
        "nl": "nee",
        "nn": "nei",
        "no": "nei",
        "sv": "nej",
    },
}

SENT_LABEL_MAPPING = {
    "positive": {
        "da": "positiv",
        "de": "positiv",
        "en": "positive",
        "es": "positivo",
        "fo": "positivt",
        "fr": "positif",
        "is": "jákvætt",
        "it": "positivo",
        "nb": "positiv",
        "nl": "positief",
        "nn": "positiv",
        "no": "positiv",
        "sv": "positiv",
    },
    "neutral": {
        "da": "neutral",
        "de": "neutral",
        "en": "neutral",
        "es": "neutral",
        "fo": "neutralt",
        "fr": "neutre",
        "is": "hlutlaust",
        "it": "neutro",
        "nb": "nøytral",
        "nl": "neutraal",
        "nn": "nøytral",
        "no": "nøytral",
        "sv": "neutral",
    },
    "negative": {
        "da": "negativ",
        "de": "negativ",
        "en": "negative",
        "es": "negativo",
        "fo": "negativt",
        "fr": "négatif",
        "is": "neikvætt",
        "it": "negativo",
        "nb": "negativ",
        "nl": "negatief",
        "nn": "negativ",
        "no": "negativ",
        "sv": "negativ",
    },
}

LANG_TO_OR = {
    "da": "eller",
    "de": "oder",
    "en": "or",
    "es": "o",
    "fo": "ella",
    "fr": "ou",
    "is": "eða",
    "it": "o",
    "nb": "eller",
    "nl": "of",
    "nn": "eller",
    "no": "eller",
    "sv": "eller",
}

LANG_TO_AND = {
    "da": "og",
    "de": "und",
    "en": "and",
    "es": "y",
    "fo": "og",
    "fr": "et",
    "is": "og",
    "it": "e",
    "nb": "og",
    "nl": "en",
    "nn": "og",
    "no": "og",
    "sv": "och",
}
