"""Template for dataset configs for new languages."""

import logging

from huggingface_hub import HfApi

from .. import languages
from ..data_models import DatasetConfig
from ..tasks import RC

logger = logging.getLogger("euroeval")


LANGUAGES_NOT_SUPPORTED = [
    # Not in Wikipedia
    "aa",
    "ae",
    "hz",
    "kr",
    "kj",
    "lu",
    "mh",
    "na",
    "nd",
    "nr",
    "ng",
    "nb",
    "oj",
    "pt",  # Covered by pt-pt and pt-br
    # Too few samples
    "bm",
    "bi",
    "ch",
    "cu",
    "cr",
    "ee",
    "fj",
    "kl",
    "iu",
    "ik",
    "ki",
    "kg",
    "nv",
    "pi",
    "rn",
    "sm",
    "sg",
    "st",
    "ty",
    "ti",
    "to",
    "ve",
    "za",
    "ady",
    "arc",
    "atj",
    "bug",
    "chr",
    "chy",
    "fon",
    "gan",
    "got",
    "jbo",
    "kcg",
    "lbe",
    "nov",
    "nso",
    "pag",
    "pnt",
    "rmy",
    "srn",
    "tly",
    "tpi",
    "xal",
]


api = HfApi()
for language, cfg in languages.get_all_languages().items():
    # Skip if the language is not supported by MultiWikiQA
    if language in LANGUAGES_NOT_SUPPORTED:
        continue

    # Create the dataset config
    dataset_config = DatasetConfig(
        name=f"multi-wiki-qa-{language}",
        pretty_name=f"the truncated version of the {cfg.name} part of the "
        "reading comprehension dataset MultiWikiQA",
        huggingface_id=f"EuroEval/multi-wiki-qa-{language}-mini",
        task=RC,
        languages=[getattr(languages, language.upper().replace("-", "_"))],
        _num_few_shot_examples=1,
    )

    # Add the config to the globals
    globals().update({f"MULTI_WIKI_QA_{language.upper()}_CONFIG": dataset_config})
