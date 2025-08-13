"""EuroEval - A benchmarking framework for language models."""

### STAGE 1 ###
### Block unwanted terminal output that happens on importing external modules ###

import importlib.util
import logging
import os
import sys
import warnings

from termcolor import colored

# Block specific warnings before importing anything else, as they can be noisy
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("datasets").setLevel(logging.CRITICAL)
logging.getLogger("vllm").setLevel(logging.CRITICAL)
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

# Set up logging
fmt = colored("%(asctime)s", "light_blue") + " ⋅ " + colored("%(message)s", "green")
logging.basicConfig(
    level=logging.CRITICAL if hasattr(sys, "_called_from_test") else logging.INFO,
    format=fmt,
    datefmt="%Y-%m-%d %H:%M:%S",
)


### STAGE 2 ###
### Check for incompatible packages ###

# Throw informative error if `flash_attn` is installed ###
if importlib.util.find_spec("flash_attn") is not None:
    logging.critical(
        "The `flash_attn` package is not supported by EuroEval, as it is now built "
        "into the other packages and it conflicts with the other implementations. "
        "Please uninstall it using `pip uninstall flash_attn` and try again."
    )
    sys.exit(1)


### STAGE 3 ###
### Set the rest up ###

import importlib.metadata  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

from .benchmarker import Benchmarker  # noqa: E402
from .utils import block_terminal_output  # noqa: E402

# Block unwanted terminal outputs. This blocks way more than the above, but since it
# relies on importing from the `utils` module, external modules are already imported
# before this is run, necessitating the above block as well
block_terminal_output()


# Fetches the version of the package as defined in pyproject.toml
__version__ = importlib.metadata.version("euroeval")


# Loads environment variables
load_dotenv()


# Disable parallelisation when tokenizing, as that can lead to errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Enable MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# Set amount of threads per GPU - this is the default and is only set to prevent a
# warning from showing
os.environ["OMP_NUM_THREADS"] = "1"


# Disable a warning from Ray regarding the detection of the number of CPUs
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"


# Avoid the "Cannot re-initialize CUDA in forked subprocess" error - see
# https://github.com/vllm-project/vllm/issues/6152 for more
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


# Allow long max model length in vLLM. This happens when vLLM registers that the model
# has a shorter context length than the value we are inserting. But since we do a
# thorough check of the model's config before setting the context length, we trust our
# own checks and ignore the internal vLLM check.
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"


# Avoid the "Unclosed client session" error when evaluating Ollama models with LiteLLM.
# The error comes from the `aiohttp` package, and this environment variable forces the
# use of `httpx` instead.
# Link: https://github.com/BerriAI/litellm/issues/11657#issuecomment-3038984975
os.environ["DISABLE_AIOHTTP_TRANSPORT"] = "True"


# Use older version v0 of vLLM, as the newer one requires XGrammar as decoding backend,
# but XGrammar does not support having a maximal amount of elements in lists
os.environ["VLLM_USE_V1"] = "0"


# Set the HF_TOKEN env var to copy the HUGGINGFACE_API_KEY env var, as vLLM uses the
# former and LiteLLM uses the latter
if os.getenv("HUGGINGFACE_API_KEY"):
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_API_KEY"]
