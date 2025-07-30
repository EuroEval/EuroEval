"""Template for dataset configs for new languages."""

from .. import languages
from ..data_models import DatasetConfig
from ..tasks import RC

for language, cfg in languages.get_all_languages().items():
    try:
        globals().update(
            {
                f"MULTI_WIKI_QA_{language.upper()}_CONFIG": DatasetConfig(
                    name=f"multi-wiki-qa-{language}",
                    pretty_name=f"the truncated version of the {cfg.name} part of the "
                    "reading comprehension dataset MultiWikiQA",
                    huggingface_id=f"EuroEval/multi-wiki-qa-{language}-mini",
                    task=RC,
                    languages=[getattr(languages, language.upper())],
                )
            }
        )
    except AttributeError:
        # If the language is not supported, we skip it
        pass
