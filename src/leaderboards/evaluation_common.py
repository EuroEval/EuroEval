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
import fcntl
import json
import logging
import os
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import time
import typing as t
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
    stream_output: bool = True,
    log_file: Path | t.IO[bytes] | None = None,
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
        stream_output (optional):
            When True, stream subprocess output live to stderr and force
            ``FULL_LOG=1`` for maximum verbosity. When False, suppress
            terminal output (subprocess writes go to /dev/null) and leave
            verbosity at the CLI default. Defaults to True.
        log_file (optional):
            When provided, write subprocess output to this file (or file-like
            object) as it arrives. Accepts a :class:`Path` (opened in append
            binary mode) or a binary file-like object with a ``write()`` method.
            Terminal output behaviour is still controlled by ``stream_output``.
            Defaults to None.

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
    if stream_output:
        logger.info(f"Running: {' '.join(cmd)}")
    else:
        logger.debug(f"Running: {' '.join(cmd)}")

    env = os.environ.copy()
    if stream_output:
        env["FULL_LOG"] = "1"
    token = resolve_hf_token()
    if token:
        env.setdefault("HF_TOKEN", token)
        env.setdefault("HUGGINGFACE_API_KEY", token)

    # Import here to preserve deferred import pattern for lightweight callers
    from euroeval.logging_utils import no_terminal_output  # noqa: PLC0415

    parent_fd, child_fd = pty.openpty()
    _set_pty_window_size(fd=child_fd)
    spawned_pgid: int | None = None
    try:
        # Safe: ``cmd`` is a fixed argument list run without a shell; only the
        # known euroeval CLI flags and operator-controlled model/language ids
        # are interpolated, never untrusted shell input.
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdin=child_fd,
            stdout=child_fd,
            stderr=child_fd,
            close_fds=True,
            env=env,
            preexec_fn=os.setsid,  # New process group for scoped cleanup
        )
        spawned_pgid = os.getpgid(proc.pid)
    except FileNotFoundError:
        os.close(parent_fd)
        os.close(child_fd)
        logger.error("`euroeval` CLI not found on PATH. Is it installed?")
        return 127, "`euroeval` CLI not found on PATH."
    os.close(child_fd)

    # Open log file if a Path was provided
    log_fh: t.IO[bytes] | None = None
    if isinstance(log_file, Path):
        log_fh = open(log_file, "ab")
    elif log_file is not None:
        # Assume it's already a file-like object
        log_fh = log_file

    captured: list[bytes] = []
    MAX_DRAIN_TIME_AFTER_EXIT = 2.0  # Fixed deadline after parent exit
    with no_terminal_output(disable=stream_output):
        parent_exit_time: float | None = None
        try:
            while True:
                # Check deadline first: once parent exits, bound the drain time
                # regardless of whether bytes are available. This prevents a
                # chatty descendant from keeping the loop alive forever.
                if parent_exit_time is not None:
                    elapsed = time.monotonic() - parent_exit_time
                    if elapsed > MAX_DRAIN_TIME_AFTER_EXIT:
                        break

                ready, _, _ = select.select([parent_fd], [], [], 0.1)
                try:
                    chunk = os.read(parent_fd, 4096) if ready else b""
                except OSError:
                    break

                if chunk:
                    if stream_output:
                        sys.stderr.buffer.write(chunk)
                        sys.stderr.buffer.flush()
                    if log_fh is not None:
                        log_fh.write(chunk)
                        log_fh.flush()
                    captured.append(chunk)
                    # Parent may have exited while we read; start deadline
                    if proc.poll() is not None and parent_exit_time is None:
                        parent_exit_time = time.monotonic()
                elif proc.poll() is not None:
                    # Parent exited and no more bytes immediately available.
                    # Start deadline timer if not already set.
                    if parent_exit_time is None:
                        parent_exit_time = time.monotonic()
                # else: process still running, continue waiting
        finally:
            os.close(parent_fd)
            # Kill process group for scoped cleanup (no broad pkill/killall)
            if spawned_pgid is not None:
                try:
                    os.killpg(spawned_pgid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    pass
            if log_fh is not None and isinstance(log_file, Path):
                # Only close if we opened it
                log_fh.close()
        proc.wait()
    output = b"".join(captured).decode("utf-8", errors="replace")
    note = _killed_by_signal_note(proc.returncode)
    if note:
        output += f"\n{note}\n"
    return proc.returncode, output


def _killed_by_signal_note(returncode: int | None) -> str | None:
    """Return a diagnostic note when a subprocess was killed by a signal.

    A negative ``returncode`` (as reported by ``subprocess.Popen``) means the
    process was terminated by a signal rather than exiting normally. Such
    deaths leave no Python traceback, so without this note the failure
    comment falls back to "(no output captured)". Resolving the signal to a
    name gives the operator a cause; SIGKILL is flagged as a likely
    out-of-memory kill, the common failure mode for large model evaluations.

    Args:
        returncode:
            The ``Popen.returncode`` to interpret.

    Returns:
        A diagnostic line, or None when ``returncode`` is not a signal death.
    """
    if returncode is None or returncode >= 0:
        return None
    signum = abs(returncode)
    try:
        name = signal.Signals(signum).name
    except ValueError:
        return (
            f"Process was killed by signal {signum} - likely out of memory."
            if signum == signal.SIGKILL
            else f"Process was killed by signal {signum}."
        )
    if signum == signal.SIGKILL:
        return f"Process was killed by signal {signum} ({name}) - likely out of memory."
    return f"Process was killed by signal {signum} ({name})."


def _set_pty_window_size(fd: int) -> None:
    """Give a freshly-opened pty a sensible window size.

    ``pty.openpty()`` creates the slave with a default window size of 0 rows by
    0 columns. Progress bars (tqdm, driven by vLLM) can't render a bar in 0
    columns, so they fall back to emitting a bare newline on every refresh --
    flooding the captured output and the tmux panes with blank lines. Setting a
    real window size makes tqdm render a single in-place-updating line instead.

    The size is inherited from this process's own controlling terminal when it
    has one (so the bar matches the operator's pane), falling back to a roomy
    80x200 default for non-interactive runs (cron, redirected output).

    Args:
        fd:
            The pty file descriptor whose window size should be set.
    """
    rows, cols = 50, 200
    for stream in (sys.stderr, sys.stdout, sys.stdin):
        try:
            packed = fcntl.ioctl(
                stream.fileno(), termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0)
            )
            src_rows, src_cols, _, _ = struct.unpack("HHHH", packed)
        except (OSError, ValueError, AttributeError):
            continue
        if src_rows > 0 and src_cols > 0:
            rows, cols = src_rows, src_cols
            break
    try:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    except OSError as e:
        logger.debug(f"Could not set pty window size: {e}")


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
    needed = int(needed * GPU_FIT_OVERHEAD)
    return needed <= gpu_bytes, needed


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
    # Deferred: see the module-level note on avoiding heavy euroeval import cost.
    from euroeval.string_utils import split_model_id  # noqa: PLC0415

    components = split_model_id(model_id=model_id)
    repo = components.model_id
    revision = components.revision
    token = os.environ.get("HF_TOKEN")
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
    """Return the memory available for model weights on this host.

    Honours ``EUROEVAL_GPU_MEMORY_BYTES`` for overrides. Otherwise: when CUDA
    is available and the host is not flagged as ``UNIFIED_MEMORY=1``, report
    the largest CUDA device's free memory; otherwise report available system
    RAM via ``psutil``.

    Returns:
        The available memory in bytes, or None if the override is unparseable.
    """
    override = os.environ.get("EUROEVAL_GPU_MEMORY_BYTES")
    if override:
        try:
            return int(override)
        except ValueError:
            logger.warning(f"Ignoring invalid EUROEVAL_GPU_MEMORY_BYTES={override!r}.")

    # Deferred: see the module-level note on avoiding heavy import cost.
    import psutil  # noqa: PLC0415
    import torch  # noqa: PLC0415

    unified = os.environ.get("UNIFIED_MEMORY", "0") == "1"
    if torch.cuda.is_available() and not unified:
        total_free: int = 0
        for device_id in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(device_id)
            total_free += free
        return total_free

    # Use available (not total) system RAM for CPU-only or unified memory hosts.
    return int(psutil.virtual_memory().available)


@lru_cache(maxsize=1)
def official_dataset_language_pairs() -> set[tuple[str, str]]:
    """Return all official dataset/language pairs used by EuroEval.

    Returns:
        The ``(dataset_name, language_code)`` pairs for every official
        (i.e. non-unofficial) dataset in :mod:`euroeval.dataset_configs`.
    """
    # Deferred: see the module-level note on avoiding heavy euroeval import cost.
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


def result_dataset_language_pairs(
    lines: c.Iterable[str],
) -> set[tuple[str, str, bool | None]]:
    """Return dataset/language/validation_split triples from benchmark results.

    Args:
        lines:
            Benchmark-result JSONL lines to parse.

    Returns:
        The set of (dataset, language, validation_split) triples found in the lines.
    """
    # Deferred: see the module-level note on avoiding heavy euroeval import cost.
    from euroeval.data_models import BenchmarkResult  # noqa: PLC0415

    pairs: set[tuple[str, str, bool | None]] = set()
    for line in lines:
        try:
            parsed = BenchmarkResult.from_dict(config=json.loads(line))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        pairs.update(
            (parsed.dataset, language, parsed.validation_split)
            for language in parsed.languages
        )
    return pairs


def missing_official_dataset_language_pairs(
    lines: c.Iterable[str], requested_languages: c.Iterable[str]
) -> set[tuple[str, str]]:
    """Return missing official dataset/language pairs for the requested languages.

    Only counts results from the validation split (or validation_split=None for
    backwards compatibility) as "complete" for queue purposes. Test-split results
    do not count towards completion.

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
    observed_triples = result_dataset_language_pairs(lines=lines)
    # Only count validation-split (True) or None (backwards compatibility) as complete
    observed_pairs: set[tuple[str, str]] = {
        (dataset, lang)
        for dataset, lang, validation_split in observed_triples
        if validation_split is True or validation_split is None
    }
    return expected_pairs - observed_pairs


def resolve_hf_token() -> str | None:
    """Return the HF token from env or the ``hf auth login`` cache.

    Returns:
        The token string, or None if neither env nor cache yields one.
    """
    return os.environ.get("HF_TOKEN") or get_token()
