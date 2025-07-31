"""Template for dataset configs for new languages."""

from huggingface_hub import HfApi

from .. import languages
from ..data_models import DatasetConfig
from ..tasks import RC

api = HfApi()
for language, cfg in languages.get_all_languages().items():
    # Skip if the language is not supported by MultiWikiQA
    if not api.repo_exists(f"EuroEval/multi-wiki-qa-{language}-mini"):
        continue

    # Create the dataset config
    dataset_config = DatasetConfig(
        name=f"multi-wiki-qa-{language}",
        pretty_name=f"the truncated version of the {cfg.name} part of the "
        "reading comprehension dataset MultiWikiQA",
        huggingface_id=f"EuroEval/multi-wiki-qa-{language}-mini",
        task=RC,
        languages=[getattr(languages, language.upper())],
        _num_few_shot_examples=2,
    )

    # Add the config to the globals
    globals().update({f"MULTI_WIKI_QA_{language.upper()}_CONFIG": dataset_config})
