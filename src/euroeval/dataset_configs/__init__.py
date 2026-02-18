"""All dataset configurations used in EuroEval."""

import collections.abc as c
import logging
from pathlib import Path

from ..custom_dataset_configs import (
    load_custom_datasets_module,
    try_get_dataset_config_from_repo,
)
from ..data_models import DatasetConfig
from ..languages import get_all_languages
from ..logging_utils import log_once
from ..tasks import SPEED
from .albanian import *  # noqa: F403
from .bosnian import *  # noqa: F403
from .bulgarian import *  # noqa: F403
from .catalan import *  # noqa: F403
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
from .hungarian import *  # noqa: F403
from .icelandic import *  # noqa: F403
from .italian import *  # noqa: F403
from .latvian import *  # noqa: F403
from .lithuanian import *  # noqa: F403
from .norwegian import *  # noqa: F403
from .polish import *  # noqa: F403
from .portuguese import *  # noqa: F403
from .romanian import *  # noqa: F403
from .serbian import *  # noqa: F403
from .slovak import *  # noqa: F403
from .slovene import *  # noqa: F403
from .spanish import *  # noqa: F403
from .swedish import *  # noqa: F403
from .translation_configs import *  # noqa: F403
from .ukrainian import *  # noqa: F403


def get_all_dataset_configs(
    custom_datasets_file: Path,
    dataset_ids: c.Sequence[str],
    api_key: str | None,
    cache_dir: Path,
) -> dict[str, DatasetConfig]:
    """Get a mapping of all the dataset configurations.

    Args:
        custom_datasets_file:
            A path to a Python file containing custom dataset configurations.
        dataset_ids:
            The IDs of the datasets to include in the mapping.
        api_key:
            The Hugging Face API key to use to check if the repositories have custom
            dataset configs.
        cache_dir:
            The directory to store the cache in.

    Returns:
        A mapping between names of datasets and their configurations.
    """
    globals_dict = globals()

    # If any of the dataset IDs are referring to Hugging Face dataset IDs, then we check
    # if the repositories have custom dataset configs and if they do, we add them to the
    # globals dict.
    for dataset_id in dataset_ids:
        dataset_config_or_none = try_get_dataset_config_from_repo(
            dataset_id=dataset_id, api_key=api_key, cache_dir=cache_dir
        )
        if dataset_config_or_none is not None:
            globals_dict[dataset_id] = dataset_config_or_none
            msg = f"Loaded external dataset {dataset_id}"
            split_strings = []
            if dataset_config_or_none.train_split is not None:
                split_strings.append(
                    f"train split '{dataset_config_or_none.train_split}'"
                )
            if dataset_config_or_none.val_split is not None:
                split_strings.append(f"val split '{dataset_config_or_none.val_split}'")
            if dataset_config_or_none.test_split is not None:
                split_strings.append(
                    f"test split '{dataset_config_or_none.test_split}'"
                )
            if split_strings:
                msg += f" with {', '.join(split_strings[:-1])} and {split_strings[-1]}"
            msg += "."
            log_once(msg, level=logging.INFO)

    # Add the custom datasets from the custom datasets file to the globals dict
    module = load_custom_datasets_module(custom_datasets_file=custom_datasets_file)
    if module is not None:
        globals_dict |= vars(module)

    # Extract the dataset configs from the globals dict
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


SPEED_CONFIG = DatasetConfig(
    name="speed",
    pretty_name="",
    source="",
    task=SPEED,
    languages=list(get_all_languages().values()),
)
