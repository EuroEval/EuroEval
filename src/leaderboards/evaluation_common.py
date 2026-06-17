"""Shared helpers for the queue processor and the core-model runner.

Both ``process_evaluation_queue.py`` and ``run_core_model_evaluations.py``
need to:

* invoke the ``euroeval`` CLI with the same pty-streamed/captured stdout,
* size-check open-weight models against the local GPU/RAM budget,
* enumerate the official ``(dataset, language)`` pairs, and
* resolve a HuggingFace token for the subprocess env.

This module is the single home for that code so the two scripts stay in
sync.
"""

from __future__ import annotations

import collections.abc as c
import json
import logging
import os
import pty
import re
import select
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from huggingface_hub import get_safetensors_metadata, get_token
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    NotASafetensorsRepoError,
    RepositoryNotFoundError,
    SafetensorsParsingError,
)

from .constants import DTYPE_BYTES, GPU_FIT_OVERHEAD, LANGUAGE_GROUP_CODES

# Heavy imports (``torch``, ``psutil``, and anything under ``euroeval`` -- which
# transitively loads ``nltk`` and ``sklearn``) are deferred to the functions
# that actually need them, so lightweight callers like the results collector
# don't pay a multi-second import cost they never use.

logger = logging.getLogger(__name__)


def run_euroeval(
    model_id: str,
    languages: c.Sequence[str],
    datasets: c.Sequence[str] | None = None,
    evaluate_test_split: bool = True,
    zero_shot: bool = False,
    trust_remote_code: bool = True,
    clear_model_cache: bool = True,
    gpu_memory_utilization: float | None = None,
) -> tuple[int, str]:
    """Run the euroeval CLI for the given model, languages, and datasets.

    Output is streamed live to stderr (so the operator can follow progress
    on long evaluations) while also being captured for post-run inspection.

    Args:
        model_id:
            The model identifier to evaluate.
        languages:
            ISO codes to pass via repeated ``--language`` flags.
        datasets (optional):
            Dataset ids to pass via repeated ``--dataset`` flags. When
            None or empty, no ``--dataset`` flag is passed and the CLI
            uses its language-driven default. Defaults to None.
        evaluate_test_split (optional):
            When True pass ``--evaluate-test-split``; when False pass
            ``--evaluate-val-split``. Defaults to True.
        zero_shot (optional):
            When True pass ``--zero-shot``; otherwise omit (CLI default
            is few-shot). Defaults to False.
        trust_remote_code (optional):
            Pass ``--trust-remote-code``. Defaults to True.
        clear_model_cache (optional):
            Pass ``--clear-model-cache``. Defaults to True.
        gpu_memory_utilization (optional):
            When set, pass ``--gpu-memory-utilization VALUE``. When None,
            omit the flag so the euroeval CLI's default applies. Defaults
            to None.

    Returns:
        A ``(returncode, combined_output)`` pair. A returncode of 127
        signals that the CLI was not found on PATH.
    """
    cmd: list[str] = ["euroeval", "--model", model_id]
    if clear_model_cache:
        cmd.append("--clear-model-cache")
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.append(
        "--evaluate-test-split" if evaluate_test_split else "--evaluate-val-split"
    )
    if zero_shot:
        cmd.append("--zero-shot")
    for lang in languages:
        cmd += ["--language", lang]
    for dataset in datasets or []:
        cmd += ["--dataset", dataset]
    if gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    logger.info(f"Running: {' '.join(cmd)}")

    env = os.environ.copy()
    env["FULL_LOG"] = "1"
    token = resolve_hf_token()
    if token:
        env.setdefault("HF_TOKEN", token)
        env.setdefault("HUGGINGFACE_API_KEY", token)

    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
            env=env,
        )
    except FileNotFoundError:
        os.close(master_fd)
        os.close(slave_fd)
        logger.error("`euroeval` CLI not found on PATH. Is it installed?")
        return 127, "`euroeval` CLI not found on PATH."
    os.close(slave_fd)

    captured: list[bytes] = []
    try:
        while True:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                sys.stderr.buffer.write(chunk)
                sys.stderr.buffer.flush()
                captured.append(chunk)
            elif proc.poll() is not None:
                try:
                    while True:
                        chunk = os.read(master_fd, 4096)
                        if not chunk:
                            break
                        sys.stderr.buffer.write(chunk)
                        sys.stderr.buffer.flush()
                        captured.append(chunk)
                except OSError:
                    pass
                break
    finally:
        os.close(master_fd)
    proc.wait()
    output = b"".join(captured).decode("utf-8", errors="replace")
    return proc.returncode, output


def model_fits_locally(model_id: str, gpu_bytes: int | None) -> tuple[bool, int | None]:
    """Return whether an open-weight model fits the local memory budget.

    Args:
        model_id:
            The HuggingFace repo id to size-check.
        gpu_bytes:
            The local memory budget in bytes, or None when unknown
            (in which case the caller should skip the check).

    Returns:
        A ``(fits, needed_bytes)`` pair. ``fits`` is True when either
        ``gpu_bytes`` is None, the model's safetensors footprint can't
        be measured, or ``needed * GPU_FIT_OVERHEAD <= gpu_bytes``.
        ``needed_bytes`` is the measured footprint, or None when it
        couldn't be measured.
    """
    if gpu_bytes is None:
        return True, None
    needed = estimated_model_bytes(model_id=model_id)
    if needed is None:
        return True, None
    return int(needed * GPU_FIT_OVERHEAD) <= gpu_bytes, needed


def estimated_model_bytes(model_id: str) -> int | None:
    """Estimate the weight footprint of a model in bytes.

    Reads the safetensors header(s) so quantised checkpoints are sized
    correctly.

    Args:
        model_id:
            The HuggingFace repo id, optionally suffixed with ``@<revision>``.

    Returns:
        The total weight bytes across all tensors, or None when the repo
        is not a safetensors repo or metadata cannot be parsed.
    """
    from euroeval.string_utils import split_model_id  # noqa: PLC0415

    components = split_model_id(model_id=model_id)
    repo = components.model_id
    revision = components.revision
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    try:
        meta = get_safetensors_metadata(repo_id=repo, revision=revision, token=token)
    except (
        NotASafetensorsRepoError,
        SafetensorsParsingError,
        RepositoryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
    ) as e:
        logger.debug(f"safetensors metadata unavailable for {model_id}: {e}")
        return None

    total = 0
    for dtype, count in meta.parameter_count.items():
        dtype_bits = _dtype_bits(dtype=dtype)
        if dtype_bits is None:
            logger.debug(
                f"Unknown safetensors dtype {dtype!r} for {model_id}; "
                "assuming 2 bytes/param."
            )
            dtype_bits = 16
        total += (count * dtype_bits + 7) // 8
    return total


def _dtype_bits(dtype: str) -> int | None:
    """Infer the number of bits used by a safetensors dtype string.

    Args:
        dtype:
            The safetensors dtype string.

    Returns:
        The bit-width, or None when it cannot be inferred.
    """
    normalised_dtype = dtype.upper()
    if normalised_dtype in DTYPE_BYTES:
        return DTYPE_BYTES[normalised_dtype] * 8

    for match in re.findall(pattern=r"\d+", string=normalised_dtype):
        bits = int(match)
        if bits > 0:
            return bits
    return None


def gpu_total_memory_bytes() -> int | None:
    """Return the total memory available for model weights on this host.

    Honours ``EUROEVAL_GPU_MEMORY_BYTES`` for overrides. Otherwise: when CUDA
    is available and the host is not flagged as ``UNIFIED_MEMORY=1``, report
    the largest CUDA device's total memory; otherwise report total system
    RAM via ``psutil``.

    Returns:
        The total memory in bytes, or None if the override is unparseable.
    """
    override = os.environ.get("EUROEVAL_GPU_MEMORY_BYTES")
    if override:
        try:
            return int(override)
        except ValueError:
            logger.warning(f"Ignoring invalid EUROEVAL_GPU_MEMORY_BYTES={override!r}.")

    import psutil  # noqa: PLC0415
    import torch  # noqa: PLC0415

    unified = os.environ.get("UNIFIED_MEMORY", "0") == "1"
    if torch.cuda.is_available() and not unified:
        sizes = [
            torch.cuda.get_device_properties(i).total_memory
            for i in range(torch.cuda.device_count())
        ]
        if sizes:
            return max(sizes)
    return int(psutil.virtual_memory().total)


@lru_cache(maxsize=1)
def official_dataset_language_pairs() -> set[tuple[str, str]]:
    """Return all official dataset/language pairs used by EuroEval.

    Returns:
        The ``(dataset_name, language_code)`` pairs for every official
        (i.e. non-unofficial) dataset in :mod:`euroeval.dataset_configs`.
    """
    from euroeval.dataset_configs import get_all_dataset_configs  # noqa: PLC0415

    all_dataset_configs = get_all_dataset_configs(
        custom_datasets_file=Path(""),
        dataset_ids=[],
        api_key=None,
        cache_dir=Path(".cache"),
        trust_remote_code=False,
        run_with_cli=False,
    )
    return {
        (dataset_config.name, language.code)
        for dataset_config in all_dataset_configs.values()
        if not dataset_config.unofficial
        for language in dataset_config.languages
    }


def extract_language_groups(body: str | None) -> list[str]:
    """Return the language-group labels that are ticked in an issue body.

    Args:
        body:
            The markdown body of a model-evaluation-request issue, or None.

    Returns:
        The labels (keys of ``LANGUAGE_GROUP_CODES``) whose checkbox is ticked.
    """
    if not body:
        return []
    selected: list[str] = []
    for group in LANGUAGE_GROUP_CODES:
        pattern = re.compile(rf"-\s*\[[xX]\]\s*{re.escape(group)}")
        if pattern.search(body):
            selected.append(group)
    return selected


def result_dataset_language_pairs(lines: c.Iterable[str]) -> set[tuple[str, str]]:
    """Return dataset/language pairs parsed from benchmark-result JSONL lines.

    Args:
        lines:
            Benchmark-result JSONL lines to parse.

    Returns:
        The set of (dataset, language) pairs found in the lines.
    """
    from euroeval.data_models import BenchmarkResult  # noqa: PLC0415

    pairs: set[tuple[str, str]] = set()
    for line in lines:
        try:
            parsed = BenchmarkResult.from_dict(config=json.loads(line))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        pairs.update((parsed.dataset, language) for language in parsed.languages)
    return pairs


def missing_official_dataset_language_pairs(
    lines: c.Iterable[str], requested_languages: c.Iterable[str]
) -> set[tuple[str, str]]:
    """Return missing official dataset/language pairs for the requested languages.

    Args:
        lines:
            Benchmark-result JSONL lines that have already been produced.
        requested_languages:
            The language codes the leaderboard requested.

    Returns:
        The set of official (dataset, language) pairs that have no result yet.
    """
    requested = set(requested_languages)
    expected_pairs = {
        pair for pair in official_dataset_language_pairs() if pair[1] in requested
    }
    observed_pairs = result_dataset_language_pairs(lines=lines)
    return expected_pairs - observed_pairs


def resolve_hf_token() -> str | None:
    """Return the HF token from env or the ``hf auth login`` cache.

    Returns:
        The token string, or None if neither env nor cache yields one.
    """
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_API_KEY")
        or get_token()
    )
