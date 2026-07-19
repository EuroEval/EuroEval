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

from euroeval.string_utils import split_model_id

from .constants import MODELS_WITHOUT_URLS_CACHE
from .records import plain_model_id

# Matches the href of an anchored model name, e.g.
# `<a href='https://hf.co/org/repo'>org/repo</a>`.
_ANCHOR_HREF_RE = re.compile(r"<a [^>]*href=['\"](?P<href>[^'\"]+)['\"]")

logger = logging.getLogger(__name__)


@cache
def generate_task_link(task_id: int, label: str) -> str:
    """Generate a link to a EuroEval task.

    Args:
        task_id:
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
        f"<a id={task_id} href='https://euroeval.com/tasks/{label}/' {styling}>"
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
    # If the id is an anchored name whose href already holds the model URL, use
    # it directly. This avoids resolving the whole `<a ...>...</a>` string as a
    # model id (which always fails) and the spurious "remove model?" prompt that
    # follows.
    anchor_href_match = _ANCHOR_HREF_RE.search(model_id)
    if anchor_href_match is not None:
        return anchor_href_match.group("href")

    # Strip any anchor and variant suffix so the URL generators see the canonical
    # `org/repo` slug rather than e.g. `org/repo (zero-shot)`.
    model_id_without_extras = split_model_id(model_id=plain_model_id(model_id)).model_id

    # Any model with a cached decision (remove or keep-without-url) never gets
    # a URL, so the model_url field stays None for these ids.
    if _load_model_url_decision(model_id=model_id_without_extras) is not None:
        return None

    url_generators = (
        generate_ollama_url,
        generate_hf_hub_url,
        generate_openai_url,
        generate_anthropic_url,
        generate_google_url,
        generate_xai_url,
        generate_ordbogen_url,
        generate_alx_url,
    )
    for url_generator in url_generators:
        url = url_generator(model_id=model_id_without_extras)
        if url is not None:
            return url

    return None


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


@cache
def generate_alx_url(model_id: str) -> str | None:
    """Generate a model URL for a model hosted on ALX.

    Args:
        model_id:
            The Ordbogen model ID.

    Returns:
        The ALX platform URL for the provider, or None if the model ID is not an ALX
        model.
    if not model_id.startswith("alx/"):
        return None
    return "https://platform.alexandra.dk/pricing/"


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
