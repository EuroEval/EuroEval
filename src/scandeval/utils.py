"""Utility functions to be used in other scripts."""

import gc
import importlib
import logging
import os
import random
import re
import sys
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Type

import numpy as np
import pkg_resources
import requests
import torch
from datasets.utils import disable_progress_bar
from huggingface_hub import HfApi, ModelFilter
from huggingface_hub.hf_api import ModelInfo
from requests.exceptions import RequestException
from transformers import GenerationConfig, PreTrainedModel
from transformers import logging as tf_logging

from .config import Language
from .enums import Framework
from .languages import DA, NB, NN, NO, SV, get_all_languages
from .protocols import GenerativeModel, Tokenizer

logger = logging.getLogger(__package__)


GENERATIVE_MODEL_TASKS = [
    "text-generation",
    "conversational",
    "text2text-generation",
]


GENERATIVE_DATASET_SUPERTASKS = [
    "multiple-choice",
    "text-to-text",
    "text-modelling",
]


def create_model_cache_dir(cache_dir: str, model_id: str) -> str:
    """Create cache directory for a model.

    Args:
        cache_dir:
            The cache directory.
        model_id:
            The model ID.

    Returns:
        The path to the cache directory.
    """
    # to avoid nesting due to models name containing '/'
    _model_id = model_id.replace("/", "--")
    cache_dir_path = Path(cache_dir) / "model_cache" / _model_id
    return str(cache_dir_path)


def clear_memory():
    """Clears the memory of unused items."""

    # Clear the Python cache
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def enforce_reproducibility(framework: Framework | str, seed: int = 4242):
    """Ensures reproducibility of experiments.

    Args:
        framework:
            The framework used for the benchmarking.
        seed:
            Seed for the random number generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    if framework in (Framework.PYTORCH, Framework.JAX):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    return rng


def is_module_installed(module: str) -> bool:
    """Check if a module is installed.

    This is used when dealing with spaCy models, as these are installed as separate
    Python packages.

    Args:
        module:
            The name of the module.

    Returns:
        Whether the module is installed or not.
    """
    # Get list of all modules, including their versions
    installed_modules_with_versions = list(pkg_resources.working_set)

    # Strip the module versions from the list of modules. Also make the modules lower
    # case and replace dashes with underscores
    installed_modules = [
        re.sub("[0-9. ]", "", str(module)).lower().replace("-", "_")
        for module in installed_modules_with_versions
    ]

    # Check if the module is installed by checking if the module name is in the list
    return module.lower() in installed_modules


def block_terminal_output():
    """Blocks libraries from writing output to the terminal.

    This filters warnings from some libraries, sets the logging level to ERROR for some
    libraries, disabled tokeniser progress bars when using Hugging Face tokenisers, and
    disables most of the logging from the `transformers` library.
    """

    # Ignore miscellaneous warnings
    warnings.filterwarnings(
        "ignore",
        module="torch.nn.parallel*",
        message="Was asked to gather along dimension 0, but all input tensors were "
        "scalars; will instead unsqueeze and return a vector.",
    )
    warnings.filterwarnings("ignore", module="seqeval*")

    # Up the logging level, to disable outputs
    logging.getLogger("filelock").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.ERROR)
    logging.getLogger("torch.nn.parallel.distributed").setLevel(logging.ERROR)
    logging.getLogger("accelerate").setLevel(logging.ERROR)

    # Disable the tokeniser progress bars
    disable_progress_bar()

    # Disable most of the `transformers` logging
    tf_logging._default_log_level = logging.CRITICAL
    tf_logging.set_verbosity(logging.CRITICAL)
    logging.getLogger("transformers.trainer").setLevel(logging.CRITICAL)

    # Disable `deepspeed` logging
    try:
        import deepspeed

        deepspeed.utils.logging.logger.setLevel(logging.CRITICAL)
        deepspeed.utils.logging.LoggerFactory.create_logger = (
            lambda *args, **kwargs: logging.getLogger("deepspeed").setLevel(
                logging.CRITICAL
            )
        )
        deepspeed.runtime.zero.parameter_offload.print_rank_0 = (
            lambda *args, **kwargs: None
        )
    except ImportError:
        pass


def get_class_by_name(
    class_name: str | list[str],
    module_name: str | None = None,
) -> Type | None:
    """Get a class by its name.

    Args:
        class_name:
            The name of the class, written in kebab-case. The corresponding class name
            must be the same, but written in PascalCase, and lying in a module with the
            same name, but written in snake_case. If a list of strings is passed, the
            first class that is found is returned.
        module_name:
            The name of the module where the class is located. If None then the module
            name is assumed to be the same as the class name, but written in
            snake_case. Defaults to None.

    Returns:
        The class. If the class is not found, None is returned.
    """
    # Ensure that `class_name` is a sequence
    if isinstance(class_name, str):
        class_name = [class_name]

    # Loop over the class names
    for name in class_name:
        # Get the snake_case and PascalCase version of the class name
        name_snake = name.replace("-", "_")
        name_pascal = kebab_to_pascal(name)

        # Import the module
        try:
            if not module_name:
                module_name = f"scandeval.{name_snake}"
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            module_name = None
            continue

        # Get the class from the module
        try:
            class_: Type = getattr(module, name_pascal)
        except AttributeError:
            module_name = None
            continue

        # Return the class
        return class_

    # If the class could not be found, return None
    return None


def kebab_to_pascal(kebab_string: str) -> str:
    """Converts a kebab-case string to PascalCase.

    Args:
        kebab_string:
            The kebab-case string.

    Returns:
        The PascalCase string.
    """
    return "".join(word.title() for word in kebab_string.split("-"))


def internet_connection_available() -> bool:
    """Checks if internet connection is available by pinging google.com.

    Returns:
        Whether or not internet connection is available.
    """
    try:
        requests.get("https://www.google.com")
        return True
    except RequestException:
        return False


def get_special_token_metadata(tokenizer: Tokenizer) -> dict:
    """Get the special token metadata for a tokenizer.

    Args:
        tokenizer:
            The tokenizer.

    Returns:
        The special token metadata.
    """
    # Create some test input IDs, to check if the tokenizer is adding special tokens
    test_input_ids = tokenizer("Test").input_ids

    # Extract the CLS token IDs from the tokenizer, if it's using them
    has_cls_token = True
    if tokenizer.cls_token_id in test_input_ids:
        cls_token_id = tokenizer.cls_token_id
        cls_token = tokenizer.cls_token
    elif tokenizer.bos_token_id in test_input_ids:
        cls_token_id = tokenizer.bos_token_id
        cls_token = tokenizer.bos_token
    elif tokenizer.cls_token is not None:
        cls_token_id = tokenizer.cls_token_id
        cls_token = tokenizer.cls_token
        has_cls_token = False
    else:
        cls_token_id = tokenizer.bos_token_id
        cls_token = tokenizer.bos_token
        has_cls_token = False

    # Extract the SEP token IDs from the tokenizer, if it's using them
    has_sep_token = True
    if tokenizer.sep_token_id in test_input_ids:
        sep_token = tokenizer.sep_token
    elif tokenizer.eos_token_id in test_input_ids:
        sep_token = tokenizer.eos_token
    elif tokenizer.sep_token is not None:
        sep_token = tokenizer.sep_token
        has_sep_token = False
    else:
        sep_token = tokenizer.eos_token
        has_sep_token = False

    return dict(
        cls_token_id=cls_token_id,
        cls_token=cls_token,
        sep_token=sep_token,
        has_cls_token=has_cls_token,
        has_sep_token=has_sep_token,
    )


def get_huggingface_model_lists(
    languages: list[Language] | None,
    token: bool | str,
    is_main_process: bool,
) -> dict[str, list[str]]:
    """Fetches up-to-date model lists from the Hugging Face Hub.

    Args:
        languages:
            The language codes of the language to consider. If None then the models
            will not be filtered on language.
        token:
            The authentication token for the Hugging Face Hub. If a boolean value is
            specified then the token will be fetched from the Hugging Face CLI, where
            the user has logged in through `huggingface-cli login`. If a string is
            specified then it will be used as the token. Defaults to False.
        is_main_process:
            Whether this is the main process. Only relevant for distributed training.

    Returns:
        The keys are filterings of the list, which includes all language codes,
        including 'multilingual', as well as 'all'. The values are lists of model IDs.
    """
    # Get list of all languages
    all_languages = list(get_all_languages().values())

    # If no languages are specified, then include all languages
    language_list = all_languages if languages is None else languages

    # Form string of languages
    if len(language_list) == 1:
        language_string = language_list[0].name
    else:
        language_list = sorted(language_list, key=lambda x: x.name)
        if {lang.code for lang in language_list} == {
            lang.code for lang in all_languages
        }:
            language_string = "all"
        else:
            # Remove generic 'Norwegian' from the list of languages if both 'Bokmål'
            # and 'Nynorsk' already exist in the list
            if all([lang in language_list for lang in [NO, NB, NN]]):
                language_list = [lang for lang in language_list if lang != NO]

            language_string = (
                f"{', '.join(lang.name for lang in language_list[:-1])} and "
                f"{language_list[-1].name}"
            )

    # Log fetching message
    if is_main_process:
        logger.info(
            f"Fetching list of {language_string} models from the Hugging Face Hub."
        )

    # Initialise the API
    api: HfApi = HfApi()

    # Initialise model lists
    model_lists = defaultdict(list)

    # Do not iterate over all the languages if we are not filtering on language
    language_itr: list[Language | None]
    if {lang.code for lang in language_list} == {lang.code for lang in all_languages}:
        language_itr = [None]
    else:
        language_itr = deepcopy(language_list)  # type: ignore[arg-type]

    for language in language_itr:
        # Extract the language code
        language_str: str | None
        if language is not None:
            language_str = language.code
        else:
            language_str = None

        # Fetch the model list
        models: list[ModelInfo] = api.list_models(
            filter=ModelFilter(language=language_str), token=token
        )

        # Filter the models to only keep the ones with the specified language
        models = [
            model
            for model in models
            if (language is None or language.code in model.tags)
        ]

        # Only keep the models which are not finetuned
        models = [
            model
            for model in models
            if model.pipeline_tag is None
            or model.pipeline_tag
            in {"fill-mask", "sentence-similarity", "feature-extraction"}
            | set(GENERATIVE_MODEL_TASKS)
        ]

        # Extract the model IDs
        model_ids: list[str] = [model.modelId for model in models if model.modelId]

        # Remove models that have "finetuned" in their name
        model_ids = [
            model_id for model_id in model_ids if "finetuned" not in model_id.lower()
        ]

        # Store the model IDs
        model_lists["all"].extend(model_ids)
        if language is not None:
            model_lists[language.code].extend(model_ids)

    # Add multilingual models manually
    multi_models = [
        "bert-base-multilingual-cased",
        "bert-base-multilingual-uncased",
        "distilbert-base-multilingual-cased",
        "cardiffnlp/twitter-xlm-roberta-base",
        "microsoft/infoxlm-base",
        "microsoft/infoxlm-large",
        "microsoft/xlm-align-base",
        "microsoft/mdeberta-v3-base",
        "setu4993/LaBSE",
        "sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking",
        "sentence-transformers/distiluse-base-multilingual-cased",
        "sentence-transformers/distiluse-base-multilingual-cased-v1",
        "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
        "sentence-transformers/quora-distilbert-multilingual",
        "sentence-transformers/stsb-xlm-r-multilingual",
        "sentence-transformers/use-cmlm-multilingual",
        "studio-ousia/mluke-base",
        "studio-ousia/mluke-large",
        "xlm-roberta-base",
        "xlm-roberta-large",
        "dbmdz/bert-tiny-historic-multilingual-cased",
        "dbmdz/bert-mini-historic-multilingual-cased",
        "dbmdz/bert-base-historic-multilingual-cased",
        "dbmdz/bert-medium-historic-multilingual-cased",
    ]
    model_lists["multilingual"] = multi_models
    model_lists["all"].extend(multi_models)

    # Add fresh models
    fresh_models = ["fresh-xlm-roberta-base", "fresh-electra-small"]
    model_lists["fresh"].extend(fresh_models)
    model_lists["all"].extend(fresh_models)

    # Add some multilingual Danish models manually that have not marked 'da' as their
    # language
    if DA in language_itr:
        multi_da_models: list[str] = [
            "Geotrend/bert-base-en-da-cased",
            "Geotrend/bert-base-25lang-cased",
            "Geotrend/bert-base-en-fr-de-no-da-cased",
            "Geotrend/distilbert-base-en-da-cased",
            "Geotrend/distilbert-base-25lang-cased",
            "Geotrend/distilbert-base-en-fr-de-no-da-cased",
        ]
        model_lists["da"].extend(multi_da_models)
        model_lists["all"].extend(multi_da_models)

    # Add some multilingual Swedish models manually that have not marked 'sv' as their
    # language
    if SV in language_itr:
        multi_sv_models: list[str] = []
        model_lists["sv"].extend(multi_sv_models)
        model_lists["all"].extend(multi_sv_models)

    # Add some multilingual Norwegian models manually that have not marked 'no', 'nb'
    # or 'nn' as their language
    if any(lang in language_itr for lang in [NO, NB, NN]):
        multi_no_models: list[str] = [
            "Geotrend/bert-base-en-no-cased",
            "Geotrend/bert-base-25lang-cased",
            "Geotrend/bert-base-en-fr-de-no-da-cased",
            "Geotrend/distilbert-base-en-no-cased",
            "Geotrend/distilbert-base-25lang-cased",
            "Geotrend/distilbert-base-en-fr-de-no-da-cased",
        ]
        model_lists["no"].extend(multi_no_models)
        model_lists["all"].extend(multi_no_models)

    # Remove duplicates from the lists
    for lang, model_list in model_lists.items():
        model_lists[lang] = list(set(model_list))

    # Remove banned models
    BANNED_MODELS = [
        r"TransQuest/siamesetransquest-da.*",
        r"M-CLIP/.*",
        r".*/.*CTRL.*",  # TEMP
    ]
    for lang, model_list in model_lists.items():
        model_lists[lang] = [
            model
            for model in model_list
            if not any(re.search(regex, model) is not None for regex in BANNED_MODELS)
        ]

    return dict(model_lists)


class HiddenPrints:
    """Context manager which removes all terminal output."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def model_is_generative(model: PreTrainedModel | GenerativeModel) -> bool:
    """Check if a model is generative or not.

    Args:
        model:
            The model to check.

    Returns:
        Whether the model is generative or not.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            dummy_inputs = torch.tensor([[1]], device=model.device, dtype=torch.long)
            generation_config = GenerationConfig(
                max_new_tokens=1,
                pad_token_id=model.config.pad_token_id,
                eos_token_id=model.config.eos_token_id,
            )
            model.generate(inputs=dummy_inputs, generation_config=generation_config)
            return True
    except (NotImplementedError, TypeError):
        return False
