"""Template for dataset configs for new languages."""

from huggingface_hub import HfApi

from .. import languages
from ..data_models import DatasetConfig
from ..tasks import RC

globals().update(
    {
        f"MULTI_WIKI_QA_{language.upper()}_CONFIG": DatasetConfig(
            name=f"multi-wiki-qa-{language}",
            pretty_name=f"the truncated version of the {language} part of the reading "
            "comprehension dataset MultiWikiQA",
            huggingface_id=f"EuroEval/multi-wiki-qa-{language}-mini",
            task=RC,
            languages=[getattr(languages, language.upper())],
        )
        for language in [
            cfg["config_name"].split(".")[-1]
            for cfg in HfApi()
            .repo_info(repo_id="alexandrainst/multi-wiki-qa", repo_type="dataset")
            .card_data.configs
        ]
    }
)
