"""Generating links for models."""

from __future__ import annotations

import logging
import re
import time
from functools import cache

import httpx
import openai
from huggingface_hub import HfApi
from huggingface_hub.errors import (
    GatedRepoError,
    HFValidationError,
    LocalTokenNotFoundError,
    RepositoryNotFoundError,
)
from requests.exceptions import RequestException
from yaml import safe_dump, safe_load

from .paths import MODELS_WITHOUT_URLS_CACHE
from .utils import log_once

logger = logging.getLogger(__name__)


KNOWN_MODELS_WITHOUT_URLS = [
    "fresh-electra-small",
    "fresh-xlm-roberta-base",
    "skole-gpt-mixtral",
    "danish-foundation-models/munin-7b-v0.1dev0",
    "mhenrichsen/danskgpt-chat-v2.1",
    "syvai/danskgpt-chat-llama3-70b",
    "syvai/llama3-da-base",
    "xai/grok-3-beta",
    "xai/grok-3-mini-beta",
    "danish-foundation-models/munin-7b-core-pt",
    "danish-foundation-models/munin-7b-core-pt-2",
    "danish-foundation-models/munin-7b-core-pt-3",
    "danish-foundation-models/munin-7b-core-it",
    "danish-foundation-models/munin-7b-open-pt",
    "danish-foundation-models/munin-7b-open-it",
]


@cache
def generate_task_link(id: int, label: str) -> str:
    """Generate a link to a EuroEval task.

    Args:
        id:
            A unique task ID.
        label:
            The task ID, in kebab-case.

    Returns:
        The anchor tag of the task, linking to the EuroEval task description.
    """
    styling = (
        "style='"
        "font-size: 12px; "
        "font-weight: normal; "
        "color: Grey; "
        "text-decoration: underline;"
        "'"
    )
    return (
        f"<a id={id} href='https://euroeval.com/tasks/{label}/' {styling}>"
        f"{label.replace('-', ' ').capitalize()}"
        "</a>"
    )


def generate_model_url(model_id: str) -> str | None:
    """Generate a URL for a model.

    Args:
        model_id:
            The model ID.

    Returns:
        The URL for the model, or None if no URL can be generated.
    """
    model_id_without_extras = model_id.split("@")[0].split("#")[0]

    # Skip URL generation for models without hosted pages
    if model_id_without_extras in KNOWN_MODELS_WITHOUT_URLS:
        return None

    url = generate_ollama_url(model_id=model_id_without_extras)
    if url is None:
        url = generate_hf_hub_url(model_id=model_id_without_extras)
    if url is None:
        url = generate_openai_url(model_id=model_id_without_extras)
    if url is None:
        url = generate_anthropic_url(model_id=model_id_without_extras)
    if url is None:
        url = generate_google_url(model_id=model_id_without_extras)
    if url is None:
        url = generate_xai_url(model_id=model_id_without_extras)
    if url is None:
        url = generate_ordbogen_url(model_id=model_id_without_extras)

    return url


@cache
def generate_anchor_tag(model_id: str) -> str | None:
    """Generate an anchor tag for a model.

    Args:
        model_id:
            The model ID.

    Returns:
        The anchor tag for the model, or the model ID if the URL cannot be generated.
        Can also return None if the model should be removed from the results.
    """
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)

    # Skip URL generation for already-annotated models
    if re.match(r"^<a href='.*'>.*</a>$", model_id):
        return model_id

    model_id_without_extras = model_id.split("@")[0].split("#")[0]

    # Skip URL generation for models without hosted pages
    if model_id_without_extras in KNOWN_MODELS_WITHOUT_URLS:
        return model_id

    # Check cached decision (remove or keep without URL)
    cached_decision = _load_model_url_decision(model_id=model_id_without_extras)
    if cached_decision is True:
        # Cached decision: remove this model
        log_once(
            f"Removing model {model_id_without_extras} from results (cached decision).",
            logging_level=logging.INFO,
        )
        return None
    if cached_decision is False:
        # Cached decision: keep without URL
        return model_id

    url = generate_model_url(model_id=model_id_without_extras)
    if url is None:
        remove_model = ask_user_to_remove_model(model_id=model_id_without_extras)
        if remove_model:
            log_once(
                f"Removing model {model_id_without_extras} from results.",
                logging_level=logging.INFO,
            )
            return None

    return model_id if url is None else f"<a href='{url}'>{model_id}</a>"


@cache
def ask_user_to_remove_model(model_id: str) -> bool:
    """Ask the user if they want to remove a model from the results.

    Args:
        model_id:
            The model ID.

    Returns:
        True if the user wants to remove the model from the results, False otherwise.
    """
    # Check persistent cache first
    cached_decision = _load_model_url_decision(model_id=model_id)
    if cached_decision is not None:
        return cached_decision

    while True:
        user_input = input(
            f"Could not find a URL for model {model_id}. Do you want to remove it from "
            "the results? (y/n): "
        )
        if user_input not in ["y", "n"]:
            print("Invalid input. Please enter 'y' or 'n'.")
            continue
        remove = user_input == "y"
        _remember_model_url_decision(model_id=model_id, remove=remove)
        return remove


@cache
def _load_model_url_decisions() -> dict[str, bool]:
    """Load cached model URL decisions (remove or keep).

    Returns:
        A dict mapping model IDs to whether they should be removed (True)
        or kept without a URL (False). Returns an empty dict if the cache
        file does not exist.
    """
    if not MODELS_WITHOUT_URLS_CACHE.exists():
        return {}
    with MODELS_WITHOUT_URLS_CACHE.open("r") as f:
        data = safe_load(f) or {}
    # Backwards compatibility: old format was a list of model IDs to keep
    if isinstance(data, list):
        return {model_id: False for model_id in data}
    return data


def _load_model_url_decision(model_id: str) -> bool | None:
    """Load a cached decision for a specific model.

    Args:
        model_id:
            The model ID to look up.

    Returns:
        True if the model should be removed, False if it should be kept
        without a URL, or None if no cached decision exists.
    """
    decisions = _load_model_url_decisions()
    return decisions.get(model_id)


def _remember_model_url_decision(model_id: str, remove: bool) -> None:
    """Persist a model URL decision to the cache.

    Args:
        model_id:
            The model ID.
        remove:
            True if the model should be removed, False if it should be
            kept without a URL.
    """
    decisions = _load_model_url_decisions()
    if model_id in decisions:
        return
    decisions[model_id] = remove
    with MODELS_WITHOUT_URLS_CACHE.open("w") as f:
        safe_dump(dict(sorted(decisions.items())), f)
    _load_model_url_decisions.cache_clear()
    generate_anchor_tag.cache_clear()


def _check_model_exists_with_retry(model_id: str, hf_api: HfApi) -> None:
    """Check if a model exists on the Hugging Face Hub with retry logic.

    Retries only on connection-related errors (not repository errors).

    Args:
        model_id:
            The Hugging Face model ID.
        hf_api:
            The Hugging Face API client.

    Raises:
        RepositoryNotFoundError:
            If the repository does not exist (not retried).
        GatedRepoError:
            If the repository is gated (not retried).
        HFValidationError:
            If the model ID is invalid (not retried).
        httpx.RemoteProtocolError:
            If the server disconnects without sending a response (retried).
        ConnectionError:
            If there is a network connection error (retried).
    """
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            hf_api.model_info(repo_id=model_id)
            return
        except (httpx.RemoteProtocolError, ConnectionError) as e:
            if attempt == max_attempts - 1:
                raise  # Re-raise on last attempt
            wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
            logger.warning(
                f"Connection error checking {model_id}: {e}. "
                f"Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)
        except (RepositoryNotFoundError, GatedRepoError, HFValidationError):
            raise  # Don't retry these errors


@cache
def generate_hf_hub_url(model_id: str) -> str | None:
    """Generate a model URL for a model hosted on the Hugging Face Hub.

    Args:
        model_id:
            The Hugging Face model ID.

    Returns:
        The URL for the model on the Hugging Face Hub, or None if the model does not
        exist on the Hugging Face Hub.
    """
    hf_api = HfApi()
    try:
        _check_model_exists_with_retry(model_id=model_id, hf_api=hf_api)
        return f"https://hf.co/{model_id}"
    except (
        GatedRepoError,
        LocalTokenNotFoundError,
        RepositoryNotFoundError,
        HFValidationError,
        RequestException,
        OSError,
    ):
        return None


@cache
def generate_openai_url(model_id: str) -> str | None:
    """Generate a model URL for a model hosted on OpenAI.

    Args:
        model_id:
            The OpenAI model ID.

    Returns:
        The URL for the model on OpenAI, or None if the model does not exist on OpenAI.
    """
    model_id = model_id.replace("openai/", "")

    available_openai_models = [
        model_info.id for model_info in openai.models.list().data
    ]

    if model_id == "gpt-4-1106-preview":
        model_id_without_version_id = "gpt-4-turbo"
    else:
        model_id_without_version_id_parts: list[str] = []
        for part in model_id.split("-"):
            if re.match(r"^\d{2,}$", part):
                break
            model_id_without_version_id_parts.append(part)
        model_id_without_version_id = "-".join(model_id_without_version_id_parts)

    if (
        model_id in available_openai_models
        or model_id_without_version_id in available_openai_models
    ):
        return f"https://platform.openai.com/docs/models/{model_id_without_version_id}"
    return None


@cache
def generate_anthropic_url(model_id: str) -> str | None:
    """Generate a model URL for a model hosted on Anthropic.

    Checks if the model ID matches Anthropic's naming patterns (claude-*).
    Does not validate against the API list, as older models may not appear there
    even though they're still valid.

    Args:
        model_id:
            The Anthropic model ID.

    Returns:
        The URL for the model on Anthropic, or None if the model does not match
        Anthropic's naming pattern.
    """
    model_id = model_id.replace("anthropic/", "")
    # Match Anthropic model naming patterns:
    # - claude-3-7-sonnet-20250219
    # - claude-sonnet-4-5-20250929
    # - claude-opus-4-20250514
    # - claude-haiku-4-5-20251001
    if model_id.startswith("claude-"):
        return "https://docs.anthropic.com/en/docs/about-claude"
    return None


@cache
def generate_ollama_url(model_id: str) -> str | None:
    """Generate a model URL for a model hosted on Ollama.

    Args:
        model_id:
            The Ollama model ID.

    Returns:
        The URL for the model on Ollama, or None if the model does not exist on Ollama.
    """
    if not model_id.startswith("ollama/") and not model_id.startswith("ollama_chat/"):
        return None
    model_id = model_id.replace("ollama/", "").replace("ollama_chat/", "")
    return f"https://ollama.com/library/{model_id}"


@cache
def generate_google_url(model_id: str) -> str | None:
    """Generate a model URL for a model hosted on Google.

    Args:
        model_id:
            The Google model ID.

    Returns:
        The URL for the model on Google, or None if the model does not exist on Google.
    """
    if not model_id.startswith("gemini/"):
        return None
    model_id = model_id.replace("gemini/", "")
    return f"https://ai.google.dev/gemini-api/docs/models#{model_id}"


@cache
def generate_xai_url(model_id: str) -> str | None:
    """Generate a model URL for a model hosted on xAI.

    Args:
        model_id:
            The xAI model ID.

    Returns:
        The URL for the model on xAI, or None if the model does not exist on xAI.
    """
    if not model_id.startswith("xai/"):
        return None
    model_id = model_id.replace("xai/", "")
    return f"https://docs.x.ai/developers/models/{model_id}"


@cache
def generate_ordbogen_url(model_id: str) -> str | None:
    """Generate a model URL for a model hosted on Ordbogen.

    Args:
        model_id:
            The Ordbogen model ID.

    Returns:
        The URL for the model on Ordbogen, or None if the model does not exist on
        Ordbogen.
    """
    if not model_id.startswith("ordbogen/"):
        return None
    model_id = model_id.replace("ordbogen/", "")
    return f"https://www.ordbogen.ai/docs/models/{model_id}"
