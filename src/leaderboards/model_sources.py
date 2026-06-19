"""Model-source helpers: EU-org matching and parameter-count resolution.

These functions identify EU-built models from a regex list and resolve a
best-effort parameter count for a model, either from a size token in its
id or from its Hugging Face safetensors manifest.
"""

from __future__ import annotations

import collections.abc as c
import logging
import re
from functools import cache

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.hf_api import RepositoryNotFoundError

from .constants import PARAMS_FROM_ID_RE
from .records import plain_model_id

logger = logging.getLogger(__name__)


def eu_models(model_ids: c.Iterable[str], eu_patterns: list[str]) -> set[str]:
    """Return the subset of `model_ids` matched by an EU-org regex.

    Args:
        model_ids:
            Candidate model identifiers (HuggingFace-style).
        eu_patterns:
            Regex patterns from `core_models.yaml::eu_model_patterns`.

    Returns:
        Set of model_ids that match at least one pattern.
    """
    compiled = [re.compile(p) for p in eu_patterns]
    eu_model_ids = {
        model_id
        for model_id in model_ids
        if any(p.search(plain_model_id(model_id).split("#")[0]) for p in compiled)
    }
    logger.info(f"Fetched {len(eu_model_ids)} EU models.")
    return eu_model_ids


def params_from_model_id(model_id: str) -> float:
    """Best-effort parameter count parsed from a model identifier.

    Some entries (typically ones we haven't evaluated yet, or with broken
    HF metadata) reach the core-model list with NaN parameters, which
    would otherwise default them to the ``xlarge`` bucket. When the id
    itself encodes the size (``EuroLLM-22B``, ``Ministral-3-14B``,
    ``SmolLM2-135M``, ...), use that as a fallback.

    Args:
        model_id:
            HuggingFace-style id, optionally with `org/` prefix.

    Returns:
        Parameter count, or NaN if no size token is present.
    """
    m = PARAMS_FROM_ID_RE.search(model_id)
    if not m:
        return float("nan")
    value = float(m.group(1))
    unit = m.group(2).lower()
    return value * (1_000_000_000 if unit == "b" else 1_000_000)


@cache
def params_from_hf_safetensors(model_id: str) -> float:
    """Best-effort parameter count from a model's safetensors manifest.

    HF's model-info endpoint exposes a ``safetensors.total`` field for
    repos that ship safetensors weights; that's a sum across every
    parameter tensor in the manifest. We use it as a last-resort
    fallback for models where neither the EuroEval metadata nor the id
    itself encodes the size (`yulan-team/YuLan-Mini` and friends).

    Network failures, missing repos, and missing safetensors data all
    degrade to NaN — the caller will then bucket the model as ``xlarge``
    which is the existing behaviour.

    Args:
        model_id:
            HuggingFace ``org/repo`` slug, optionally with an
            ``@revision`` suffix. Anything that doesn't look like
            ``a/b`` is skipped without a network call.

    Returns:
        Total parameter count, or NaN.
    """
    repo_id, _, revision = model_id.partition("@")
    if "/" not in repo_id or repo_id.count("/") > 1:
        return float("nan")
    try:
        info = HfApi().model_info(
            repo_id=repo_id, revision=revision or None, files_metadata=False
        )
    except (RepositoryNotFoundError, HfHubHTTPError, OSError, ValueError) as exc:
        # 404s for IDs we synthesised locally (renamed repos, transient
        # variants, etc.) aren't actionable — log at debug level so the
        # standard run output stays clean.
        logger.debug(f"HF safetensors lookup failed for {model_id!r}: {exc}")
        return float("nan")
    safetensors = getattr(info, "safetensors", None)
    if safetensors is None:
        return float("nan")
    total = getattr(safetensors, "total", None)
    return float(total) if isinstance(total, int) and total > 0 else float("nan")
