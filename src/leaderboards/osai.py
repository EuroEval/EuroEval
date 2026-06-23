"""Scrape and rank the OSAI 'truly open' model index (osai-index.eu).

The OSAI site is a Nuxt SPA that exposes its model database via a rotating
JS bundle. These helpers locate that bundle from the homepage, parse the
model entries, and rank the qualifying 'truly open' text models by their
openness count. If the scrape fails, callers fall back to a manually
curated override list.
"""

from __future__ import annotations

import logging
import re
import typing as t
import urllib.error
import urllib.request

from .constants import (
    OSAI_BASE,
    OSAI_BLOCK_LINK_RE,
    OSAI_BUNDLE_MARKER,
    OSAI_FIELD_RE,
    OSAI_RANKING_FIELDS,
    OSAI_REQUIRED_OPEN,
    OSAI_SYSTEM_RE,
    OSAI_WEIGHT_CRITERIA,
    OSAI_WEIGHTS_BLOCK_RE,
)

logger = logging.getLogger(__name__)


class _OsaiEntry(t.TypedDict):
    """A single parsed OSAI model entry.

    Attributes:
        name:
            The model's system name.
        endmodelname:
            The end-model name.
        type:
            The model type (e.g. ``"text"``).
        open_count:
            Number of openness ranking fields marked open.
        required_open:
            Whether trainingcode and datasources_basemodel are both open.
        weight_links:
            Mapping from each open weights field name to its HF URL.
    """

    name: str
    endmodelname: str
    type: str
    open_count: int
    required_open: bool
    weight_links: dict[str, str]


def osai_top_models(
    limit: int = 10, overrides: list[str] | None = None
) -> list[tuple[str, int]]:
    """Return the top OSAI 'truly open' text models as (model_id, rank) pairs.

    The OSAI index is filtered to text models with open base weights,
    training code, and data sources. Models are ranked by total open-field
    count (descending). We scrape the live JS bundle; on any failure (no
    bundle reachable, regex match yields zero entries, etc.) we return
    the override list from the YAML config so the pipeline stays
    deterministic.

    Args:
        limit (optional):
            Number of top models to return. Defaults to 10.
        overrides (optional):
            Fallback list of `org/repo` model IDs in rank order, used
            verbatim when the scrape fails. Defaults to None.

    Returns:
        List of `(model_id, rank)` pairs in 1-based rank order. Empty
        list if both the scrape and overrides yield nothing.
    """
    logger.info(f"Fetching the top-{limit} OSAI open models...")

    bundle = _osai_bundle()
    if bundle is None:
        logger.warning("OSAI: bundle not found; using overrides.")
        return _overrides_to_ranked(overrides=overrides, limit=limit)

    entries = _parse_osai_models(bundle=bundle)
    if not entries:
        logger.warning("OSAI: parsed zero model entries; using overrides.")
        return _overrides_to_ranked(overrides=overrides, limit=limit)

    # A model qualifies if training code and base-model data sources are
    # open and at least one of base/end-model weights is open. When both
    # weights are open, both are added to the eval target list — they're
    # different checkpoints worth scoring independently.
    qualifying = [
        e
        for e in entries
        if e["type"] == "text" and e["required_open"] and e["weight_links"]
    ]
    qualifying.sort(key=lambda e: (-e["open_count"], e["endmodelname"].lower()))

    seen: set[str] = set()
    ranked: list[tuple[str, int]] = []
    for entry in qualifying:
        # Stable ordering: prefer the base-model checkpoint first, then end.
        for field in OSAI_WEIGHT_CRITERIA:
            link = entry["weight_links"].get(field)
            if not link:
                continue
            model_id = _hf_url_to_model_id(url=link)
            if model_id is None:
                logger.info(
                    f"OSAI: skipping {entry['endmodelname']!r} ({field}): "
                    f"cannot map {link!r} to org/repo."
                )
                continue
            if model_id in seen:
                continue
            seen.add(model_id)
            ranked.append((model_id, len(ranked) + 1))
            if len(ranked) >= limit:
                return ranked
    logger.info(f"Fetched {len(ranked)} OSAI models.")
    return ranked


def _osai_bundle() -> str | None:
    """Discover the current Nuxt JS bundle that contains the model database.

    OSAI ships a Nuxt SPA. The model entries are bundled into a single
    `/_nuxt/*.js` chunk whose filename rotates on every deploy. We
    enumerate the chunks referenced from the homepage, probe them in
    largest-first order, and return the first that contains the
    distinctive `system:{name:` marker that begins each model entry.

    Returns:
        Raw JS source of the model-data chunk, or None if discovery failed.
    """
    try:
        html = _fetch(url=OSAI_BASE)
    except (urllib.error.URLError, OSError) as exc:
        logger.warning(f"OSAI: cannot reach homepage: {exc}")
        return None
    chunks = re.findall(r'href="(/_nuxt/[^"]+\.js)"', html)
    if not chunks:
        logger.warning("OSAI: no /_nuxt/*.js chunks found on homepage.")
        return None
    sized: list[tuple[int, str]] = []
    for chunk in set(chunks):
        url = OSAI_BASE + chunk
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as resp:
                length = int(resp.headers.get("Content-Length") or 0)
        except (urllib.error.URLError, OSError, ValueError):
            continue
        sized.append((length, url))
    sized.sort(reverse=True)
    # Skip the obviously-too-small chunks; the model bundle is hundreds
    # of KB. Of the remaining candidates, return the first one whose
    # contents contain the model-entry marker.
    for length, url in sized:
        if length < 100_000:
            break
        try:
            body = _fetch(url=url, timeout=30.0)
        except (urllib.error.URLError, OSError):
            continue
        if OSAI_BUNDLE_MARKER in body:
            logger.info(f"OSAI: model data found in {url}")
            return body
    logger.warning("OSAI: no chunk contained the model-entry marker.")
    return None


def _parse_osai_models(bundle: str) -> list[_OsaiEntry]:
    """Parse model entries out of an OSAI JS bundle.

    Each model entry begins with a `system:{...}` block and is followed by
    openness-criteria fields. We slice the bundle on `system:{name:` to
    delimit entries, parse each slice, and stop at the next `system:{`
    (or end of bundle).

    Args:
        bundle:
            Raw JavaScript source of the Nuxt chunk that holds the model
            database.

    Returns:
        List of parsed entries, one per model.
    """
    matches = list(OSAI_SYSTEM_RE.finditer(bundle))
    entries: list[_OsaiEntry] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(bundle)
        slice_ = bundle[start:end]

        classes = dict(OSAI_FIELD_RE.findall(slice_))
        open_count = sum(
            1 for field in OSAI_RANKING_FIELDS if classes.get(field) == "open"
        )
        required_open = all(
            classes.get(field) == "open" for field in OSAI_REQUIRED_OPEN
        )

        # Collect an HF URL for each weights field that is open. If both
        # base and end weights are open we'll emit both downstream so the
        # eval covers them individually.
        weight_links: dict[str, str] = {}
        for block in OSAI_WEIGHTS_BLOCK_RE.finditer(slice_):
            field = block.group(1)
            if classes.get(field) != "open":
                continue
            link_match = OSAI_BLOCK_LINK_RE.search(block.group("body"))
            if not link_match:
                continue
            link = link_match.group("single") or link_match.group("first") or ""
            if "huggingface.co/" in link:
                weight_links[field] = link

        entries.append(
            _OsaiEntry(
                name=m.group("name"),
                endmodelname=m.group("endmodelname"),
                type=m.group("type"),
                open_count=open_count,
                required_open=required_open,
                weight_links=weight_links,
            )
        )
    return entries


def _hf_url_to_model_id(url: str) -> str | None:
    """Convert an HF URL to a `org/repo` model identifier.

    Args:
        url:
            A URL like ``https://huggingface.co/org/repo`` (with optional
            trailing path, blob link, etc.).

    Returns:
        The ``org/repo`` slug, or None if the URL doesn't match that shape.
    """
    m = re.match(r"https://huggingface\.co/([^/]+)/([^/?#]+)", url)
    if not m:
        return None
    org, repo = m.group(1), m.group(2)
    if org in {"datasets", "spaces", "docs", "blog", "collections"}:
        return None
    return f"{org}/{repo}"


def _overrides_to_ranked(
    overrides: list[str] | None, limit: int
) -> list[tuple[str, int]]:
    """Convert a flat override list into ranked tuples.

    Args:
        overrides:
            Manually-curated list of model IDs in rank order.
        limit:
            Max length of the returned list.

    Returns:
        `[(model_id, rank), ...]` up to `limit` entries.
    """
    if not overrides:
        return []
    return [(model_id, rank) for rank, model_id in enumerate(overrides[:limit], 1)]


def _fetch(url: str, timeout: float = 20.0) -> str:
    """Fetch a URL as text.

    Args:
        url:
            The URL to fetch.
        timeout (optional):
            Socket timeout in seconds. Defaults to 20.0.

    Returns:
        The response body as text.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "EuroEval-CoreModels/1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")
