"""On-disk cache for Hugging Face Hub model metadata lookups.

The queue processor consults ``HfApi.model_info`` once per candidate to
size-check the model and to detect gated repos. Without a cache, repeated
cron invocations re-hit the API for the same models and risk rate limits.
Negative results (missing repos) are not cached so typo corrections take
effect on the next run.
"""

from __future__ import annotations

import json
import logging
import sys
import time

from huggingface_hub import HfApi, ModelInfo
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from .constants import HF_CACHE_PATH, HF_CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)


def cached_model_summary(model_id: str) -> dict | None:
    """Return a ``{param_count, gated, gguf}`` summary for a model id.

    Looks up :data:`HF_CACHE_PATH` first and falls back to
    ``HfApi.model_info`` only on cache miss or stale entry. Negative
    results (model not found) are not cached so that typo corrections take
    effect immediately.

    Args:
        model_id:
            The Hugging Face repo id to look up.

    Returns:
        A dict with keys:

        - ``param_count`` (int, possibly ``sys.maxsize`` for unknown).
        - ``gated`` (bool, True when the configured assignee does not have
          read access to a gated repo -- i.e. ``model_info`` raised
          ``GatedRepoError``).
        - ``gguf`` (bool, True when the repo is a GGUF model, which the
          evaluation queue cannot run).

        Returns None when the model is not on the Hub.
    """
    cache = _load_hf_cache()
    entry = cache.get(model_id)
    if entry is not None and "param_count" in entry:
        return {
            "param_count": int(entry["param_count"]),
            "gated": False,
            "gguf": bool(entry.get("gguf", False)),
        }

    try:
        info = HfApi().model_info(repo_id=model_id)
    except GatedRepoError:
        # Don't cache: access can be granted at any time, and we want to
        # pick that up on the next run.
        return {"param_count": sys.maxsize, "gated": True, "gguf": False}
    except RepositoryNotFoundError:
        # Expected for typo-d / since-deleted repos; just drop the candidate.
        return None
    except Exception as e:  # noqa: BLE001
        # Best-effort network lookup: any other failure (transient HTTP, timeout,
        # unexpected payload) should just drop the candidate, never crash the run.
        logger.warning(f"HF model lookup failed for {model_id}: {e}")
        return None

    safetensors = getattr(info, "safetensors", None)
    total = getattr(safetensors, "total", None) if safetensors else None
    param_count = total if isinstance(total, int) and total > 0 else sys.maxsize

    gguf = is_gguf_model(info=info)

    cache[model_id] = {
        "timestamp": time.time(),
        "param_count": param_count,
        "gguf": gguf,
    }
    _write_hf_cache(cache=cache)
    return {"param_count": param_count, "gated": False, "gguf": gguf}


def is_gguf_model(info: ModelInfo) -> bool:
    """Return whether a model_info result describes a GGUF-only repo.

    The evaluation queue cannot load ``.gguf`` weights, but many repos ship
    GGUF quants *alongside* safetensors (e.g. ``norallm/normistral-11b-warm``)
    and are perfectly runnable from the safetensors. So a repo counts as GGUF
    only when it has ``.gguf`` weights and *no* safetensors.

    The GGUF signal uses three independent indicators so it survives the many
    ways GGUF repos are laid out (per-quant subfolders, sharded files, names
    without a quant suffix): the ``gguf`` tag the Hub sets automatically on any
    repo containing a ``.gguf`` file, a ``library_name`` of ``gguf``, and the
    presence of any ``.gguf`` file in ``siblings`` (which enumerates the repo
    recursively).

    Args:
        info:
            The model info returned by ``HfApi.model_info``.

    Returns:
        Whether the repo is a GGUF-only model.
    """
    filenames = [(s.rfilename or "").lower() for s in (info.siblings or [])]
    tags = info.tags or []
    has_gguf = (
        any(tag.lower() == "gguf" for tag in tags)
        or (info.library_name or "").lower() == "gguf"
        or any(name.endswith(".gguf") for name in filenames)
    )
    has_safetensors = getattr(info, "safetensors", None) is not None or any(
        name.endswith(".safetensors") for name in filenames
    )
    return has_gguf and not has_safetensors


def _load_hf_cache() -> dict[str, dict]:
    """Read the on-disk HF Hub lookup cache, or return an empty dict.

    Stale or malformed entries are silently dropped so callers always see
    a well-formed mapping of model id -> cache entry.

    Returns:
        The cached entries, keyed by model id.
    """
    try:
        data = json.loads(HF_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    now = time.time()
    fresh: dict[str, dict] = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        ts = value.get("timestamp")
        if not isinstance(ts, int | float) or now - ts > HF_CACHE_TTL_SECONDS:
            continue
        fresh[key] = value
    return fresh


def _write_hf_cache(cache: dict[str, dict]) -> None:
    """Persist ``cache`` to :data:`HF_CACHE_PATH` atomically.

    Args:
        cache:
            The mapping of model id to cache entry to write to disk.
    """
    try:
        HF_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = HF_CACHE_PATH.with_suffix(HF_CACHE_PATH.suffix + ".tmp")
        tmp.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        tmp.replace(HF_CACHE_PATH)
    except OSError as e:
        logger.warning(f"Could not write HF cache to {HF_CACHE_PATH}: {e}")
