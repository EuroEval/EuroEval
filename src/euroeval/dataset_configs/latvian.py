"""All Latvian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import LV
from ..tasks import RC

### Unofficial datasets ###


MULTI_WIKI_QA_LV_CONFIG = DatasetConfig(
    name="multi-wiki-qa-lv",
    pretty_name="the truncated version of the Latvian part of the reading "
    "comprehension dataset MultiWikiQA",
    huggingface_id="EuroEval/multi-wiki-qa-lv-mini",
    task=RC,
    languages=[LV],
    unofficial=True,
)
