"""Machine translation dataset configurations."""

from ..data_models import DatasetConfig
from ..dataset_preprocessors import preprocess_wmt24pp_en_da
from ..languages import DANISH, ENGLISH, UKRAINIAN
from ..tasks import MT

MT_DA_UK_CONFIG = DatasetConfig(
    name="mt-da-uk",
    pretty_name="MT-DA-UK",
    source="oliverkinch/mt_da_uk",
    task=MT,
    languages=[DANISH, UKRAINIAN],
)

WMT24PP_EN_DA_CONFIG = DatasetConfig(
    name="wmt24pp-en-da",
    pretty_name="WMT24++ en-da",
    source="google/wmt24pp::en-da_DK",
    task=MT,
    languages=[ENGLISH, DANISH],
    train_split="train",
    val_split=None,
    test_split="test",
    num_few_shot_examples=1,
    preprocess_fn=preprocess_wmt24pp_en_da,
    split_seed=42,
    split_sizes={"train": 1, "test": None},
    unofficial=True,
)
