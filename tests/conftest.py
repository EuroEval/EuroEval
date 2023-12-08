"""General fixtures used throughout test modules."""

import os
from typing import Generator

import pytest
import torch

from scandeval.config import BenchmarkConfig
from scandeval.dataset_tasks import LA, NER, QA, SENT
from scandeval.languages import DA, NO, SV


@pytest.fixture(scope="session")
def benchmark_config() -> Generator[BenchmarkConfig, None, None]:
    # Get the authentication token to the Hugging Face Hub
    auth = os.environ.get("HUGGINGFACE_HUB_TOKEN", True)

    # Ensure that the token does not contain quotes or whitespace
    if isinstance(auth, str):
        auth = auth.strip(" \"'")

    yield BenchmarkConfig(
        model_languages=[DA, SV, NO],
        dataset_languages=[DA, SV, NO],
        dataset_tasks=[NER, QA, SENT, LA],
        framework=None,
        batch_size=32,
        raise_errors=False,
        cache_dir=".scandeval_cache",
        evaluate_train=False,
        token=auth,
        openai_api_key=None,
        progress_bar=False,
        save_results=False,
        device=torch.device("cpu"),
        verbose=False,
        trust_remote_code=False,
        load_in_4bit=None,
        use_flash_attention=False,
        testing=True,
    )


@pytest.fixture(scope="session")
def model_id():
    yield "jonfd/electra-small-nordic"
