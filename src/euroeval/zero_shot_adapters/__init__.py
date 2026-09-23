"""Adapters wrapping non-generative zero-shot "decision" models (e.g. Laya).

Each adapter in `ADAPTERS` wraps a model loaded through its own package, and is
tried in order by `ZeroShotClassifierModel` (see
`euroeval.benchmark_modules.zero_shot_classifier`). Adding a new adapter (e.g. for
NLI or GLiClass models) requires no changes to that module -- just append the new
adapter class to `ADAPTERS`.
"""

import typing as t

from ..exceptions import NeedsExtraInstalled
from .base import ZeroShotClassifierAdapter

if t.TYPE_CHECKING:
    from ..data_models import BenchmarkConfig

# The registered zero-shot classifier adapters, tried in order. Empty by default;
# concrete adapters (e.g. `LayaAdapter`) register themselves here.
ADAPTERS: list[type[ZeroShotClassifierAdapter]] = []


def get_adapter(
    model_id: str, benchmark_config: "BenchmarkConfig"
) -> type[ZeroShotClassifierAdapter] | None:
    """Find the first registered adapter that matches the given model ID.

    Args:
        model_id:
            The model ID, without revision or parameter suffixes.
        benchmark_config:
            The benchmark configuration.

    Returns:
        The first matching adapter class, or None if no adapter matches.

    Raises:
        NeedsExtraInstalled:
            If an adapter matches the model ID but its required extra is not
            installed.
    """
    needs_extras: list[str] = list()
    for adapter_cls in ADAPTERS:
        matches_or_err = adapter_cls.matches(
            model_id=model_id, benchmark_config=benchmark_config
        )
        if isinstance(matches_or_err, NeedsExtraInstalled):
            needs_extras.append(matches_or_err.extra)
        elif matches_or_err is True:
            return adapter_cls
    if needs_extras:
        raise NeedsExtraInstalled(extra=needs_extras[0])
    return None


__all__ = ["ADAPTERS", "ZeroShotClassifierAdapter", "get_adapter"]
