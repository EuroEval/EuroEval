"""All dataset configurations used in EuroEval."""

import collections.abc as c
import logging
from pathlib import Path

from huggingface_hub import HfApi

from ..data_models import DatasetConfig
from ..languages import get_all_languages
from ..logging_utils import log_once
from ..tasks import SPEED
from ..utils import get_hf_token, load_custom_datasets_module
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
    # globals dict. If they don't, we warn the user.
    token = get_hf_token(api_key=api_key)
    hf_api = HfApi(token=token)
    for dataset_id in dataset_ids:
        if hf_api.repo_exists(repo_id=dataset_id, repo_type="dataset"):
            repo_files = hf_api.list_repo_files(
                repo_id=dataset_id, repo_type="dataset", revision="main"
            )
            if "euroeval_config.py" not in repo_files:
                log_once(
                    f"Dataset {dataset_id} does not have a euroeval_config.py file, "
                    "so we cannot load it. Skipping.",
                    level=logging.WARNING,
                )
                continue
            external_config_path = cache_dir / "external_dataset_configs" / dataset_id
            external_config_path.mkdir(parents=True, exist_ok=True)
            hf_api.hf_hub_download(
                repo_id=dataset_id,
                repo_type="dataset",
                filename="euroeval_config.py",
                local_dir=external_config_path,
                local_dir_use_symlinks=False,
            )
            module = load_custom_datasets_module(
                custom_datasets_file=external_config_path / "euroeval_config.py"
            )
            if module is None:
                continue
            repo_dataset_configs = [
                cfg for cfg in vars(module).values() if isinstance(cfg, DatasetConfig)
            ]
            match len(repo_dataset_configs):
                case 0:
                    continue  # Already warned the user in this case, so we just skip
                case 1:
                    repo_dataset_config = repo_dataset_configs[0]
                    repo_dataset_config.name = dataset_id
                    repo_dataset_config.pretty_name = dataset_id
                    repo_dataset_config.source = dataset_id
                    globals_dict[dataset_id] = repo_dataset_config
                    log_once(
                        f"Loaded external dataset configuration for {dataset_id}.",
                        level=logging.INFO,
                    )
                case _:
                    log_once(
                        f"Dataset {dataset_id} has multiple dataset configurations. "
                        "Please ensure that only a single DatasetConfig is defined in "
                        "the `euroeval_config.py` file.",
                        level=logging.WARNING,
                    )

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
