"""All dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import get_all_languages
from ..tasks import SPEED
from ..utils import load_custom_datasets_module
from .bulgarian import *  # noqa: F403
from .croatian import *  # noqa: F403
from .czech import *  # noqa: F403
from .danish import *  # noqa: F403
from .dutch import *  # noqa: F403
from .english import *  # noqa: F403
from .estonian import *  # noqa: F403
from .faroese import *  # noqa: F403
from .finnish import *  # noqa: F403
from .french import *  # noqa: F403
from .german import *  # noqa: F403
from .greek import *  # noqa: F403
from .icelandic import *  # noqa: F403
from .italian import *  # noqa: F403
from .latvian import *  # noqa: F403
from .lithuanian import *  # noqa: F403
from .norwegian import *  # noqa: F403
from .polish import *  # noqa: F403
from .portuguese import *  # noqa: F403
from .serbian import *  # noqa: F403
from .slovak import *  # noqa: F403
from .slovenian import *  # noqa: F403
from .spanish import *  # noqa: F403
from .swedish import *  # noqa: F403
from .ukrainian import *  # noqa: F403


def get_all_dataset_configs() -> dict[str, DatasetConfig]:
    """Get a mapping of all the dataset configurations.

    Returns:
        A mapping between names of datasets and their configurations.
    """
    globals_dict = globals()
    module = load_custom_datasets_module()
    if module is not None:
        globals_dict |= vars(module)
    dataset_configs = [
        cfg
        for cfg in globals_dict.values()
        if isinstance(cfg, DatasetConfig) and cfg.task != SPEED
    ]
    assert len(dataset_configs) == len({cfg.name for cfg in dataset_configs}), (
        "There are duplicate dataset configurations. Please ensure that each dataset "
        "has a unique name."
    )
    mapping = {cfg.name: cfg for cfg in dataset_configs}
    return mapping


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get the dataset configuration for a dataset.

    Args:
        dataset_name:
            The name of the dataset.

    Returns:
        The dataset configuration.

    Raises:
        ValueError:
            If the dataset is not found.
    """
    dataset_configs = get_all_dataset_configs()
    if dataset_name not in dataset_configs:
        raise ValueError(f"No dataset config found for dataset {dataset_name}.")
    return dataset_configs[dataset_name]


SPEED_CONFIG = DatasetConfig(
    name="speed",
    pretty_name="",
    source="",
    task=SPEED,
    languages=list(get_all_languages().values()),
    _logging_string="the speed estimation benchmark",
)
