"""EuroEval - A benchmarking framework for language models."""

### STAGE 1 ###
### Block unwanted terminal output that happens on importing external modules ###

import logging
import sys
import warnings

from termcolor import colored

# Block specific warnings before importing anything else, as they can be noisy
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("datasets").setLevel(logging.CRITICAL)
logging.getLogger("vllm").setLevel(logging.CRITICAL)

# Set up logging
fmt = colored("%(asctime)s", "light_blue") + " ⋅ " + colored("%(message)s", "green")
logging.basicConfig(
    level=logging.CRITICAL if hasattr(sys, "_called_from_test") else logging.INFO,
    format=fmt,
    datefmt="%Y-%m-%d %H:%M:%S",
)


### STAGE 2 ###
### Set the rest up ###

import importlib.metadata  # noqa: E402
import os  # noqa: E402

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


# Set the HF_TOKEN env var to copy the HUGGINGFACE_API_KEY env var, as vLLM uses the
# former and LiteLLM uses the latter
if os.getenv("HUGGINGFACE_API_KEY"):
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_API_KEY"]
