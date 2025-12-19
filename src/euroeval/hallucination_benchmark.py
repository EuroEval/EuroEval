"""Benchmarking model hallucination."""

import collections.abc as c
import typing as t

from .data_loading import load_raw_data

# `hallucination_toolkit` is copied from https://github.com/leochlon/hallbayes/tree/main/hallbayes
# Would be nice if it was simply a pypi package.
from .hallucination_toolkit import OpenAIBackend, OpenAIItem, OpenAIPlanner

if t.TYPE_CHECKING:
    from .benchmark_modules import BenchmarkModule
    from .data_models import BenchmarkConfig, DatasetConfig


def hallucination_benchmark(
    model: "BenchmarkModule",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
) -> c.Sequence[dict[str, float]]:
    """Benchmark model hallucination.

    Args:
        model:
            Model to use.
    """
    dataset = load_raw_data(
        dataset_config=dataset_config, cache_dir=benchmark_config.cache_dir
    )

    backend = OpenAIBackend(model="gpt-4o-mini")
    planner = OpenAIPlanner(backend, temperature=0.3)

    items = [
        OpenAIItem(
            prompt=dataset["train"][i]["question"],
            n_samples=7,
            m=6,
            skeleton_policy="closed_book",
        )
        for i in range(len(dataset["train"]))
        if dataset["train"][i]["question"] is not None
    ]
    items = items[:2]

    metrics = planner.run(
        items,
        h_star=0.05,  # Target 5% hallucination max
        isr_threshold=1.0,  # Standard ISR gate
        margin_extra_bits=0.2,  # Safety margin
        B_clip=12.0,  # Clipping bound
        clip_mode="one-sided",  # Conservative mode
    )

    scores: list[dict[str, float]] = [
        {"test_hallucination": m.roh_bound} for m in metrics
    ]
    return scores
