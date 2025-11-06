"""Constants used in the dataset creation scripts."""

# Bounds on the size of texts in sequence classification datasets
MIN_NUM_CHARS_IN_DOCUMENT = 2
MAX_NUM_CHARS_IN_DOCUMENT = 5000


# Bounds on the size of texts in question answering datasets
MIN_NUM_CHARS_IN_CONTEXT = 30
MAX_NUM_CHARS_IN_CONTEXT = 5000
MIN_NUM_CHARS_IN_QUESTION = 10
MAX_NUM_CHARS_IN_QUESTION = 150


# Bounds on the size of texts in summarisation datasets
MIN_NUM_CHARS_IN_ARTICLE = 30
MAX_NUM_CHARS_IN_ARTICLE = 6000


# Bounds on the size of texts in multiple choice datasets
MIN_NUM_CHARS_IN_INSTRUCTION = 30
MAX_NUM_CHARS_IN_INSTRUCTION = 2000
MIN_NUM_CHARS_IN_OPTION = 1
MAX_NUM_CHARS_IN_OPTION = 1000
MAX_REPETITIONS = 50


CHOICES_MAPPING = {
    "bg": "Възможности",
    "cs": "Výběr",
    "da": "Svarmuligheder",
    "de": "Antwortmöglichkeiten",
    "el": "Επιλογές",
    "en": "Choices",
    "es": "Opciones",
    "et": "Vastusevariandid",
    "fi": "Vastausvaihtoehdot",
    "fo": "Svarmøguleikar",
    "fr": "Choix",
    "hr": "Izbori",
    "is": "Svarmöguleikar",
    "it": "Scelte",
    "lt": "Pasirinkimai",
    "lv": "Izvēles",
    "nl": "Antwoordopties",
    "no": "Svaralternativer",
    "pl": "Opcje",
    "pt": "Opções",
    "sk": "Možnosti",
    "sl": "Možnosti",
    "sr": "Opcije",
    "sv": "Svarsalternativ",
    "uk": "Варіанти",
}
