"""All Slovak dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import UK
from ..tasks import SENT

### Official datasets ###

CROSS_DOMAIN_UK_REVIEWS_CONFIG = DatasetConfig(
    name="cross-domain-uk-reviews",
    pretty_name="the truncated version of the Ukrainian sentiment classification "
    "dataset Cross-Domain UK Reviews",
    huggingface_id="EuroEval/cross-domain-uk-reviews-mini",
    task=SENT,
    languages=[UK],
)
