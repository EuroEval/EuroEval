"""Machine translation dataset configurations."""

from ..data_models import DatasetConfig
from ..language_pairs_mt import DANISH_TO_UKRAINIAN
from ..languages import DANISH, UKRAINIAN
from ..tasks import MT

MT_DA_UK_CONFIG = DatasetConfig(
    name="mt-da-uk",
    pretty_name="MT-DA-UK",
    source="oliverkinch/mt_da_uk",
    task=MT,
    languages=[DANISH, UKRAINIAN],
    language_pair=DANISH_TO_UKRAINIAN,
)
