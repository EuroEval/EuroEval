"""Pick up open model-evaluation-request issues and run EuroEval on them.

This script is meant to run on the compute server. For each open
``model evaluation request`` issue that is **not yet assigned** to anyone, it:

1. Verifies that the requested model exists on the Hugging Face Hub.
2. Assigns the issue to ``saattrupdan`` so the queue UI flips to "Evaluating".
3. Runs ``euroeval`` for each requested language group.
4. Posts the new ``euroeval_benchmark_results.jsonl`` lines as a comment on
   the issue, wrapped in a ``jsonl`` code fence so the local merge script
   can pick them up.

Required env vars
-----------------
GITHUB_TOKEN          A PAT with ``issues: write`` for the EuroEval repo
                      (saattrupdan's PAT).
EUROEVAL_RESULTS_PATH Optional override for the path to
                      ``euroeval_benchmark_results.jsonl``. Defaults to
                      ``./euroeval_benchmark_results.jsonl``.
"""

from __future__ import annotations

import fcntl
import importlib.metadata
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

import psutil
import torch
from huggingface_hub import HfApi, get_safetensors_metadata, get_token
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    NotASafetensorsRepoError,
    RepositoryNotFoundError,
    SafetensorsParsingError,
)

from euroeval.data_models import BenchmarkResult
from euroeval.dataset_configs import get_all_dataset_configs
from euroeval.string_utils import split_model_id

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("process_evaluation_queue")

REPO = "EuroEval/EuroEval"
LABEL = "model evaluation request"
FAILED_LABEL = "evaluation-failed"
TITLE_PREFIX = "[MODEL EVALUATION REQUEST]"
ASSIGNEE = "saattrupdan"
RESULTS_PATH = Path(
    os.environ.get("EUROEVAL_RESULTS_PATH", "euroeval_benchmark_results.jsonl")
)
LOCK_PATH = Path(os.environ.get("EUROEVAL_QUEUE_LOCK", "/tmp/euroeval_queue.lock"))
# Keep the lock file descriptor alive for the whole process lifetime.
_LOCK_FD: int | None = None

# On-disk cache for Hugging Face Hub lookups so that repeated runs of the
# queue processor (e.g. on a cron) do not re-hit the API for the same models
# and trip rate limits.
HF_CACHE_PATH = Path(
    os.environ.get(
        "EUROEVAL_QUEUE_HF_CACHE",
        str(Path.home() / ".cache" / "euroeval" / "queue_hf_cache.json"),
    )
)
HF_CACHE_TTL_SECONDS = 6 * 60 * 60

_BALTIC = "Baltic languages (Latvian, Lithuanian)"
_FINNIC = "Finnic languages (Estonian, Finnish)"
_ROMANCE = "Romance languages (Catalan, French, Italian, Portuguese, Romanian, Spanish)"
_SCANDI = "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)"
_SLAVIC = (
    "Slavic languages (Belarusian, Bulgarian, Bosnian, Croatian, Czech, Polish,"
    " Serbian, Slovak, Slovenian, Ukrainian)"
)
_WGERMANIC = "West Germanic languages (Dutch, English, German)"

# Matches the "Model ID" section in an issue body (the form template renders
# the model id as the line immediately following a "### Model ID" heading).
MODEL_ID_BODY_RE = re.compile(r"(?:^|\n)#{1,6}\s*Model ID\s*\n+([^\n]+)")

# The frontend parses this marker on issue bodies to display "Error" /
# "Waiting for bug fix" statuses.
ERROR_MARKER_RE = re.compile(r"<!--\s*errored-on:\s*v([^\s>-]+)\s*-->")

# The frontend parses this marker on issue bodies to display the
# "Gated model" status, which means the requested HF repo is gated and
# saattrupdan does not yet have read access.
GATED_MARKER_RE = re.compile(r"<!--\s*gated-model\s*-->")

# euroeval emits a summary line like "errored 3 benchmarks" when individual
# (dataset, language) combinations fail without crashing the whole run. We use
# this to detect partial failures even when the subprocess exited 0.
ERRORED_BENCHMARKS_RE = re.compile(r"errored\s+(\d+)\s+benchmarks?", re.IGNORECASE)

# Phrase euroeval prints when it cannot load a model because the repo is gated
# and the subprocess lacks the necessary HF token / accepted access terms.
# We treat this as "gated, please grant access" rather than as a code error.
GATED_OUTPUT_RE = re.compile(r"is a gated repository", re.IGNORECASE)
# Bytes-per-parameter for each safetensors dtype string. Used to compute the
# weight footprint of a model from its safetensors header before deciding
# whether it can fit on the local GPU.
DTYPE_BYTES: dict[str, int] = {
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}

# Multiplier applied to weight bytes to leave room for activations, KV cache,
# and framework overhead when judging whether a model fits in GPU memory.
GPU_FIT_OVERHEAD = 1.2


LANGUAGE_GROUP_CODES: dict[str, list[str]] = {
    _BALTIC: ["lv", "lt"],
    _FINNIC: ["et", "fi"],
    _ROMANCE: ["ca", "fr", "it", "pt", "ro", "es"],
    _SCANDI: ["da", "fo", "is", "no", "sv"],
    _SLAVIC: ["be", "bg", "bs", "hr", "cs", "pl", "sr", "sk", "sl", "uk"],
    _WGERMANIC: ["nl", "en", "de"],
    "Albanian": ["sq"],
    "Greek": ["el"],
    "Hungarian": ["hu"],
}


@lru_cache(maxsize=1)
def official_dataset_language_pairs() -> set[tuple[str, str]]:
    """Return all official dataset/language pairs used by EuroEval."""
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


def ensure_credentials() -> None:
    """Verify that ``GITHUB_TOKEN`` is set and the user is logged into HF, or exit."""
    if not os.environ.get("GITHUB_TOKEN"):
        logger.error(
            f"GITHUB_TOKEN env var is required (a PAT with `issues: write` for {REPO})."
        )
        sys.exit(1)
    try:
        HfApi().whoami()
    except Exception as e:  # noqa: BLE001
        logger.error(
            "Not logged in to Hugging Face. Run `huggingface-cli login` "
            f"(or set HF_TOKEN) and re-run. Underlying error: {e}"
        )
        sys.exit(1)


def acquire_single_instance_lock() -> None:
    """Acquire an exclusive flock on ``LOCK_PATH`` or exit with an error.

    The lock is held for the lifetime of the process-level file descriptor (i.e.
    the lifetime of the process) and is released automatically by the kernel
    when the process exits, so no stale lock file is left behind.
    """
    global _LOCK_FD
    try:
        # Open (or create) the lock file as read/write so we can both lock it
        # and update its contents with the current process PID.
        fd = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as e:
        logger.error(f"Failed to open lock file {LOCK_PATH}: {e}")
        sys.exit(1)
    try:
        # Request a non-blocking exclusive lock: fail immediately instead of
        # waiting if another process already holds the queue lock.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        existing_pid = "unknown"
        try:
            existing_pid = LOCK_PATH.read_text(encoding="utf-8").strip() or "unknown"
        except OSError:
            pass
        os.close(fd)
        logger.error(
            f"Another process_evaluation_queue.py is already running "
            f"(pid {existing_pid}); aborting. Set EUROEVAL_QUEUE_LOCK to a "
            f"different path if you need to run a second instance."
        )
        sys.exit(1)
    # Truncate the lock file first so stale bytes from a longer old PID are
    # removed before writing the current PID.
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    # Flush file contents to disk so other processes can reliably read the PID
    # associated with the lock holder.
    os.fsync(fd)
    _LOCK_FD = fd


def main() -> None:
    """Process every unassigned model-evaluation-request issue once.

    Issues are sorted by (status priority asc, parameter count asc,
    num-language-groups asc). Status priority is 0 for fresh issues, 1 for
    gated repos (cheap marker refresh), and 2 for retries of previously
    errored evaluations, so that quicker work is picked up first and gated
    items are surfaced ahead of errored ones.
    """
    ensure_credentials()

    # Held for the lifetime of the process; released by the kernel on exit.
    acquire_single_instance_lock()

    try:
        issues = list_open_unassigned_issues()
    except urllib.error.HTTPError as e:
        logger.error(f"Failed to list issues: {e}")
        sys.exit(1)

    current_version = euroeval_version()
    current_v = version_tuple(v=current_version)

    candidates: list[tuple[int, int, int, dict, str, list[str]]] = []
    for issue in issues:
        number = issue["number"]
        title = issue.get("title", "")
        body = issue.get("body") or ""

        model_id = extract_model_id(title=title, body=body)
        if not model_id:
            logger.info(f"#{number}: skipping -- could not parse model id.")
            continue

        groups = extract_language_groups(body=body)
        if not groups:
            logger.info(f"#{number}: skipping -- no language groups selected.")
            continue

        summary = cached_model_summary(model_id=model_id)
        if summary is None:
            continue

        errored_v = errored_on_version(body=body)
        body_gated = gated_marker_present(body=body)
        is_gated_repo = bool(summary.get("gated") or summary.get("gated_repo"))

        # One-off migration: issues stuck with an errored-on marker on the
        # current version whose repo is actually gated on the Hub. Those
        # errored markers were almost certainly written by a pre-fix run that
        # mis-classified a gating failure as a generic error. Add the gated
        # marker so the UI reflects reality; keep the errored marker so the
        # script doesn't waste a fresh euroeval invocation on every cron tick.
        if (
            is_gated_repo
            and not body_gated
            and errored_v is not None
            and version_tuple(v=errored_v) >= current_v
        ):
            set_gated_with_errored_block(number=number, body=body, version=errored_v)
            logger.info(
                f"#{number}: migrated stuck errored-on marker to gated for "
                f"gated repo {model_id!r}."
            )
            continue

        if (
            errored_v is not None
            and version_tuple(v=errored_v) >= current_v
            and not body_gated
        ):
            logger.info(
                f"#{number}: skipping -- errored on v{errored_v} and current "
                f"version is v{current_version}."
            )
            continue

        param_count = summary["param_count"]
        if summary.get("gated"):
            status_priority = 1
        elif errored_v is not None:
            status_priority = 2
        else:
            status_priority = 0
        candidates.append(
            (status_priority, param_count, len(groups), issue, model_id, groups)
        )

    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    logger.info(f"Found {len(candidates)} processable issue(s).")

    gpu_bytes = gpu_total_memory_bytes()
    if gpu_bytes is None:
        logger.info(
            "Could not determine local memory budget; skipping the fit pre-check."
        )
    else:
        logger.info(f"Local memory budget: {gpu_bytes / (1024**3):.1f} GiB.")

    for status_priority, param_count, num_groups, issue, model_id, groups in candidates:
        status = {0: "fresh", 1: "gated", 2: "retry of errored eval"}[status_priority]
        logger.info(
            f"#{issue['number']}: queueing {model_id!r} ({param_count} params, "
            f"{num_groups} group(s), {status})."
        )
        if gpu_bytes is not None:
            needed = estimated_model_bytes(model_id=model_id)
            if needed is not None and int(needed * GPU_FIT_OVERHEAD) > gpu_bytes:
                logger.info(
                    f"#{issue['number']}: skipping -- model {model_id!r} needs "
                    f"~{needed / (1024**3):.1f} GiB of weights "
                    f"(× {GPU_FIT_OVERHEAD} overhead), which exceeds the local "
                    f"GPU memory of {gpu_bytes / (1024**3):.1f} GiB. Leaving the "
                    "issue unassigned so a larger machine can pick it up."
                )
                continue
        try:
            process_issue(issue=issue, model_id=model_id, groups=groups)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Error while processing issue #{issue['number']}: {e}")
        time.sleep(1)


def list_open_unassigned_issues() -> list[dict]:
    """Return open model-evaluation-request issues with no assignee.

    Returns:
        The list of open unassigned issues, with pull requests filtered out.
    """
    issues = gh_request(
        path=f"/repos/{REPO}/issues",
        params={
            "state": "open",
            "labels": LABEL,
            "per_page": "100",
            "assignee": "none",
        },
    )
    assert isinstance(issues, list)
    return [i for i in issues if "pull_request" not in i]


def extract_model_id(title: str, body: str | None = None) -> str | None:
    """Return the model id for an issue, preferring the body's Model ID section.

    Mirrors the frontend's ``extractModelId`` so the queue processor reads the
    same source of truth as the UI: the issue body's ``### Model ID`` section
    (rendered by the form template), falling back to the title prefix for
    legacy issues that lack the section.

    Args:
        title:
            The full GitHub issue title.
        body:
            The markdown body of the issue, or None.

    Returns:
        The parsed model id, or None if neither the body nor the title yield
        a usable value.
    """
    if body:
        m = MODEL_ID_BODY_RE.search(body)
        if m:
            candidate = m.group(1).strip().strip("`*_").strip()
            if candidate and candidate != "<model-name>":
                return candidate
    prefix = f"{TITLE_PREFIX} "
    if not title.startswith(prefix):
        return None
    rest = title[len(prefix) :].strip()
    return rest if rest and rest != "<model-name>" else None


def extract_language_groups(body: str | None) -> list[str]:
    """Return the language-group labels that are ticked in an issue body.

    Args:
        body:
            The markdown body of the issue, or None.

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


def errored_on_version(body: str | None) -> str | None:
    """Return the EuroEval version recorded in the issue body, if any.

    Args:
        body:
            The markdown body of the issue, or None.

    Returns:
        The version string from the ``errored-on`` marker, or None if no
        marker is present.
    """
    if not body:
        return None
    m = ERROR_MARKER_RE.search(body)
    return m.group(1) if m else None


def _load_hf_cache() -> dict[str, dict]:
    """Read the on-disk HF Hub lookup cache, or return an empty dict.

    Stale or malformed entries are silently dropped so callers always see a
    well-formed mapping of model id -> cache entry.

    Returns:
        A dict mapping model id to a cache entry containing at least
        ``timestamp`` and ``param_count``.
    """
    try:
        data = json.loads(HF_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict()
    if not isinstance(data, dict):
        return dict()
    now = time.time()
    fresh: dict[str, dict] = dict()
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        ts = value.get("timestamp")
        if not isinstance(ts, (int, float)) or now - ts > HF_CACHE_TTL_SECONDS:
            continue
        fresh[key] = value
    return fresh


def _write_hf_cache(cache: dict[str, dict]) -> None:
    """Persist ``cache`` to ``HF_CACHE_PATH`` atomically.

    Args:
        cache:
            The mapping of model id to cache entry to persist.
    """
    try:
        HF_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = HF_CACHE_PATH.with_suffix(HF_CACHE_PATH.suffix + ".tmp")
        tmp.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        tmp.replace(HF_CACHE_PATH)
    except OSError as e:
        logger.warning(f"Could not write HF cache to {HF_CACHE_PATH}: {e}")


def cached_model_summary(model_id: str) -> dict | None:
    """Return a cached ``{param_count, gated, gated_repo}`` summary for a HF model id.

    Looks up ``HF_CACHE_PATH`` first and falls back to ``HfApi.model_info``
    only on cache miss or stale entry. Negative results (model not found) are
    not cached so that typo corrections take effect immediately.

    Args:
        model_id:
            The Hugging Face repo id to look up.

    Returns:
        A dict with keys:

        - ``param_count`` (int, possibly ``sys.maxsize`` for unknown).
        - ``gated`` (bool, True when ``saattrupdan`` does not have read
          access to a gated repo -- i.e. ``model_info`` raised
          ``GatedRepoError``).
        - ``gated_repo`` (bool, True when ``model_info`` reports the repo as
          gated even though we *do* have read access -- the token used by
          the euroeval subprocess may still lack download permission for
          this specific repo).

        Returns None when the model is not on the Hub.
    """
    cache = _load_hf_cache()
    entry = cache.get(model_id)
    if entry is not None and "param_count" in entry:
        return {
            "param_count": int(entry["param_count"]),
            "gated": False,
            "gated_repo": bool(entry.get("gated_repo", False)),
        }

    try:
        info = HfApi().model_info(repo_id=model_id)
    except GatedRepoError:
        # Don't cache: access can be granted at any time, and we want to
        # pick that up on the next run.
        return {"param_count": sys.maxsize, "gated": True, "gated_repo": True}
    except RepositoryNotFoundError:
        # Expected for typo-d / since-deleted repos; just drop the candidate.
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"HF model lookup failed for {model_id}: {e}")
        return None

    safetensors = getattr(info, "safetensors", None)
    total = getattr(safetensors, "total", None) if safetensors else None
    param_count = total if isinstance(total, int) and total > 0 else sys.maxsize
    # ``info.gated`` is ``False`` for public repos and ``"auto"`` / ``"manual"``
    # for gated ones; coerce to a plain bool.
    gated_repo = bool(getattr(info, "gated", False))

    cache[model_id] = {
        "timestamp": time.time(),
        "param_count": param_count,
        "gated_repo": gated_repo,
    }
    _write_hf_cache(cache=cache)
    return {"param_count": param_count, "gated": False, "gated_repo": gated_repo}


def gpu_total_memory_bytes() -> int | None:
    """Return the total memory available for model weights on this host, in bytes.

    Honors the ``EUROEVAL_GPU_MEMORY_BYTES`` env var for overrides (useful in
    tests). Otherwise mirrors ``alexandra_llm_inference.memory_utils``: when
    CUDA is available and the host is not flagged as ``UNIFIED_MEMORY=1``,
    report the largest CUDA device's total memory; otherwise (CPU-only host,
    or unified-memory boxes like DGX Spark whose Grace-Blackwell pool is not
    exposed via ``nvidia-smi --query-gpu=memory.total``) report total system
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

    unified = os.environ.get("UNIFIED_MEMORY", "0") == "1"
    if torch.cuda.is_available() and not unified:
        sizes = [
            torch.cuda.get_device_properties(i).total_memory
            for i in range(torch.cuda.device_count())
        ]
        if sizes:
            return max(sizes)
    return int(psutil.virtual_memory().total)


def estimated_model_bytes(model_id: str) -> int | None:
    """Estimate the weight footprint of a model in bytes.

    Reads the safetensors header(s) so that quantised checkpoints (int8, fp8,
    int4) are sized correctly rather than assumed to be fp16/bf16.

    Args:
        model_id:
            The Hugging Face repo id, optionally suffixed with ``@<revision>``.

    Returns:
        The total weight bytes across all tensors, or None when the repo is
        not a safetensors repo or the metadata cannot be parsed.
    """
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
    normalized_dtype = dtype.upper()
    if normalized_dtype in DTYPE_BYTES:
        return DTYPE_BYTES[normalized_dtype] * 8

    for match in re.findall(pattern=r"\d+", string=normalized_dtype):
        bits = int(match)
        if bits > 0:
            return bits
    return None


def process_issue(issue: dict, model_id: str, groups: list[str]) -> None:
    """Claim, evaluate, and report back on a single queue issue.

    Args:
        issue:
            The GitHub issue object returned by the API.
        model_id:
            The Hugging Face model id to evaluate.
        groups:
            The selected language-group labels for this issue.
    """
    number = issue["number"]
    languages: list[str] = []
    for g in groups:
        languages.extend(LANGUAGE_GROUP_CODES[g])
    languages = sorted(set(languages))

    if not issue_is_still_claimable(number=number):
        logger.info(
            f"#{number}: skipping -- no longer open and unassigned at claim time."
        )
        return

    # Re-check gated status here so a stale snapshot from main() doesn't make
    # us run a doomed evaluation, and so we can also pick up newly granted
    # access when the body marker says gated but HF now says otherwise.
    live_summary = cached_model_summary(model_id=model_id)
    body = issue.get("body")
    body_gated = gated_marker_present(body=body)
    if live_summary is not None and live_summary.get("gated"):
        if not body_gated:
            set_gated_marker(number=number, body=body)
            logger.info(f"#{number}: marked gated -- saattrupdan lacks read access.")
        else:
            logger.info(f"#{number}: still gated -- leaving marker in place.")
        return
    if body_gated:
        clear_gated_marker(number=number, body=body)
        logger.info(f"#{number}: access granted, cleared gated marker.")

    logger.info(f"#{number}: claiming issue for {model_id!r}, languages={languages}")
    assign_issue(number=number)

    before = set(read_jsonl_lines(path=RESULTS_PATH))
    returncode, output = run_euroeval(model_id=model_id, languages=languages)
    after = read_jsonl_lines(path=RESULTS_PATH)
    new_lines = [line for line in after if line not in before]

    num_errored = num_errored_benchmarks(output=output)
    missing_official = missing_official_dataset_language_pairs(
        lines=new_lines, requested_languages=languages
    )
    crashed = returncode != 0
    produced_nothing = not new_lines
    partial = num_errored > 0
    incomplete_official = bool(missing_official)
    failed = crashed or produced_nothing or partial or incomplete_official

    if failed and GATED_OUTPUT_RE.search(output):
        version = euroeval_version()
        set_gated_with_errored_block(
            number=number, body=issue.get("body"), version=version
        )
        add_failed_label(number=number)
        unassign_issue(number=number)
        logger.info(
            f"#{number}: euroeval reported a gated repo for {model_id!r}; "
            f"marked gated and errored-on v{version} to avoid retry loops."
        )
        return

    if failed:
        version = euroeval_version()
        if crashed:
            reason = f"euroeval exited with code {returncode}"
        elif produced_nothing:
            reason = "euroeval produced no new result lines"
        elif incomplete_official:
            reason = (
                "missing results for "
                f"{len(missing_official)} official dataset-language pair(s): "
                f"{format_dataset_language_pairs(dataset_language_pairs=missing_official)}"
            )
        else:
            reason = f"euroeval reported {num_errored} errored benchmark(s)"
        tail = output[-6000:].strip() or "(no output captured)"
        comment = (
            f"Error encountered during evaluation ({reason}):\n\n"
            f"```bash\n{tail}\n```\n\n"
            f"EuroEval version: v{version}\n"
        )
        comment_on_issue(number=number, comment=comment)
        set_errored_marker(number=number, body=issue.get("body"), version=version)
        add_failed_label(number=number)
        unassign_issue(number=number)
        logger.info(f"#{number}: marked errored on v{version}, returned to queue.")
        return

    payload = "\n".join(new_lines)
    comment = f"Results for `{model_id}`:\n\n```jsonl\n{payload}\n```\n"
    comment_on_issue(number=number, comment=comment)
    remove_failed_label(number=number)
    logger.info(f"#{number}: posted {len(new_lines)} result line(s) as comment.")


def missing_official_dataset_language_pairs(
    lines: list[str], requested_languages: list[str]
) -> set[tuple[str, str]]:
    """Return missing official dataset/language pairs for the requested languages."""
    expected_pairs = {
        pair
        for pair in official_dataset_language_pairs()
        if pair[1] in set(requested_languages)
    }
    observed_pairs = result_dataset_language_pairs(lines=lines)
    return expected_pairs - observed_pairs


def result_dataset_language_pairs(lines: list[str]) -> set[tuple[str, str]]:
    """Return dataset/language pairs parsed from benchmark-result JSONL lines."""
    pairs: set[tuple[str, str]] = set()
    for line in lines:
        try:
            parsed = BenchmarkResult.from_dict(config=json.loads(line))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        pairs.update((parsed.dataset, language) for language in parsed.languages)
    return pairs


def format_dataset_language_pairs(dataset_language_pairs: set[tuple[str, str]]) -> str:
    """Return a compact stable string representation of dataset/language pairs."""
    sorted_pairs = sorted(dataset_language_pairs)
    preview = [f"{dataset}/{language}" for dataset, language in sorted_pairs[:10]]
    suffix = ""
    if len(sorted_pairs) > 10:
        suffix = f" (+{len(sorted_pairs) - 10} more)"
    return ", ".join(preview) + suffix


def issue_is_still_claimable(number: int) -> bool:
    """Return True if the issue is still open with no assignees.

    Re-fetches the issue at claim time so that issues which were closed or
    assigned between the initial snapshot and now are not double-processed.

    Args:
        number:
            The issue number to verify.

    Returns:
        True if the issue is currently open and has no assignees; False
        otherwise (including when the lookup fails).
    """
    try:
        current = gh_request(path=f"/repos/{REPO}/issues/{number}")
    except urllib.error.HTTPError as e:
        logger.warning(f"#{number}: could not re-check issue state: {e}")
        return False
    if not isinstance(current, dict):
        return False
    if current.get("state") != "open":
        return False
    return not current.get("assignees")


def assign_issue(number: int) -> None:
    """Assign the issue to the configured ``ASSIGNEE``.

    Args:
        number:
            The issue number to claim.
    """
    gh_request(
        path=f"/repos/{REPO}/issues/{number}/assignees",
        method="POST",
        body={"assignees": [ASSIGNEE]},
    )


def unassign_issue(number: int) -> None:
    """Remove ``ASSIGNEE`` so the issue returns to the unassigned pool.

    Args:
        number:
            The issue number to release.
    """
    gh_request(
        path=f"/repos/{REPO}/issues/{number}/assignees",
        method="DELETE",
        body={"assignees": [ASSIGNEE]},
    )


def add_failed_label(number: int) -> None:
    """Attach the ``evaluation-failed`` label to an issue.

    Args:
        number:
            The issue number to label.
    """
    try:
        gh_request(
            path=f"/repos/{REPO}/issues/{number}/labels",
            method="POST",
            body={"labels": [FAILED_LABEL]},
        )
    except urllib.error.HTTPError as e:
        logger.warning(f"#{number}: could not add {FAILED_LABEL!r} label: {e}")


def remove_failed_label(number: int) -> None:
    """Remove the ``evaluation-failed`` label from an issue if present.

    Args:
        number:
            The issue number to unlabel.
    """
    try:
        gh_request(
            path=f"/repos/{REPO}/issues/{number}/labels/"
            + urllib.parse.quote(FAILED_LABEL),
            method="DELETE",
        )
    except urllib.error.HTTPError as e:
        # 404 just means the label wasn't applied; anything else is worth noting.
        if e.code != 404:
            logger.warning(f"#{number}: could not remove {FAILED_LABEL!r} label: {e}")


def read_jsonl_lines(path: Path) -> list[str]:
    """Return the non-empty lines of a JSONL file, or an empty list if absent.

    Args:
        path:
            Path to the JSONL file.

    Returns:
        The stripped lines that contain at least one non-whitespace character.
    """
    if not path.is_file():
        return []
    return [
        line.rstrip("\n")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_hf_token() -> str | None:
    """Return the HF token from env or the ``hf auth login`` cache.

    The queue script's parent process is authenticated via ``HfApi().whoami()``
    in :func:`ensure_credentials`, so a token is reachable somewhere -- but
    the euroeval subprocess only picks it up if ``HF_TOKEN`` /
    ``HUGGINGFACE_API_KEY`` are set in its env, not from the cached login
    file. Resolve it here so the caller can inject it into the subprocess.

    Returns:
        The token string, or None if neither env nor cache yields one.
    """
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_API_KEY")
        or get_token()
    )


def run_euroeval(model_id: str, languages: list[str]) -> tuple[int, str]:
    """Run the euroeval CLI for the given model and language list.

    Output is streamed live to stderr (so the operator can follow progress on
    long evaluations) while also being captured for post-run inspection.

    Args:
        model_id:
            The Hugging Face model id to evaluate.
        languages:
            ISO codes of the languages to pass via ``--language``.

    Returns:
        A ``(returncode, combined_output)`` pair. The output combines stdout
        and stderr in interleaved order. A returncode of 127 signals that the
        CLI was not found on PATH.
    """
    cmd = [
        "euroeval",
        "--model",
        model_id,
        "--clear-model-cache",
        "--trust-remote-code",
    ]
    for lang in languages:
        cmd += ["--language", lang]
    logger.info(f"Running: {' '.join(cmd)}")

    env = os.environ.copy()
    token = resolve_hf_token()
    if token:
        env.setdefault("HF_TOKEN", token)
        env.setdefault("HUGGINGFACE_API_KEY", token)

    try:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except FileNotFoundError:
        logger.error("`euroeval` CLI not found on PATH. Is it installed?")
        return 127, "`euroeval` CLI not found on PATH."

    assert proc.stdout is not None
    captured: list[str] = []
    for line in proc.stdout:
        sys.stderr.write(line)
        sys.stderr.flush()
        captured.append(line)
    proc.wait()
    return proc.returncode, "".join(captured)


def num_errored_benchmarks(output: str) -> int:
    """Return the number of errored benchmarks parsed from euroeval output.

    Args:
        output:
            The full captured combined-output of the euroeval subprocess.

    Returns:
        The integer reported by the last ``errored N benchmarks`` line, or 0
        if no such line is present.
    """
    last = 0
    for m in ERRORED_BENCHMARKS_RE.finditer(output):
        last = int(m.group(1))
    return last


def comment_on_issue(number: int, comment: str) -> None:
    """Post ``comment`` to the issue with the given number.

    Args:
        number:
            The issue number to comment on.
        comment:
            The markdown body of the new comment.
    """
    gh_request(
        path=f"/repos/{REPO}/issues/{number}/comments",
        method="POST",
        body={"body": comment},
    )


def set_errored_marker(number: int, body: str | None, version: str) -> None:
    """Append/replace the ``errored-on`` marker in the issue body.

    Also strips any ``gated-model`` marker so the two states stay mutually
    exclusive.

    Args:
        number:
            The issue number to update.
        body:
            The current issue body, or None.
        version:
            The EuroEval version that produced the failure.
    """
    cleaned = ERROR_MARKER_RE.sub("", body or "").rstrip()
    cleaned = GATED_MARKER_RE.sub("", cleaned).rstrip()
    new_body = f"{cleaned}\n\n<!-- errored-on: v{version} -->\n"
    gh_request(
        path=f"/repos/{REPO}/issues/{number}", method="PATCH", body={"body": new_body}
    )


def gated_marker_present(body: str | None) -> bool:
    """Return True if the body carries the ``gated-model`` marker.

    Args:
        body:
            The markdown body of the issue, or None.

    Returns:
        True if the marker is present, False otherwise.
    """
    return bool(body) and bool(GATED_MARKER_RE.search(body))


def set_gated_marker(number: int, body: str | None) -> None:
    """Append the ``gated-model`` marker to the issue body.

    Also strips any ``errored-on`` marker so the two states stay mutually
    exclusive.

    Args:
        number:
            The issue number to update.
        body:
            The current issue body, or None.
    """
    cleaned = GATED_MARKER_RE.sub("", body or "").rstrip()
    cleaned = ERROR_MARKER_RE.sub("", cleaned).rstrip()
    new_body = f"{cleaned}\n\n<!-- gated-model -->\n"
    gh_request(
        path=f"/repos/{REPO}/issues/{number}", method="PATCH", body={"body": new_body}
    )


def set_gated_with_errored_block(number: int, body: str | None, version: str) -> None:
    """Set both the ``gated-model`` and ``errored-on`` markers in one PATCH.

    Used when euroeval reports a gated repository despite ``model_info``
    succeeding: the script and the subprocess disagree on the token's
    download permission. The gated marker drives the UI status (it wins over
    ``erroredOn`` in ``issueStatus``), while the errored marker prevents the
    script from re-running euroeval on every cron tick until the version
    bumps or access is reconfirmed.

    Args:
        number:
            The issue number to update.
        body:
            The current issue body, or None.
        version:
            The EuroEval version that observed the gated failure.
    """
    cleaned = GATED_MARKER_RE.sub("", body or "").rstrip()
    cleaned = ERROR_MARKER_RE.sub("", cleaned).rstrip()
    new_body = f"{cleaned}\n\n<!-- gated-model -->\n<!-- errored-on: v{version} -->\n"
    gh_request(
        path=f"/repos/{REPO}/issues/{number}", method="PATCH", body={"body": new_body}
    )


def clear_gated_marker(number: int, body: str | None) -> None:
    """Remove the ``gated-model`` marker from the issue body.

    Args:
        number:
            The issue number to update.
        body:
            The current issue body, or None.
    """
    cleaned = GATED_MARKER_RE.sub("", body or "").rstrip() + "\n"
    gh_request(
        path=f"/repos/{REPO}/issues/{number}", method="PATCH", body={"body": cleaned}
    )


def euroeval_version() -> str:
    """Return the locally installed EuroEval package version.

    Returns:
        The dotted version string, or "unknown" if the package metadata is
        not available.
    """
    try:
        return importlib.metadata.version("euroeval")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def version_tuple(v: str) -> tuple[int, ...]:
    """Return a comparable tuple for a dotted-numeric version like ``17.2.0``.

    Non-numeric or unknown components yield ``(-1,)`` so they sort before
    anything else and never block retries.

    Args:
        v:
            The version string to parse.

    Returns:
        A tuple of ints suitable for comparison with other parsed versions.
    """
    if v == "unknown":
        return (-1,)
    try:
        return tuple(int(p) for p in v.split(".") if p.isdigit())
    except ValueError:
        return (-1,)


def gh_request(
    path: str,
    *,
    method: str = "GET",
    body: dict | None = None,
    params: dict | None = None,
) -> dict | list | None:
    """Call the GitHub REST API and return the parsed JSON body.

    Args:
        path:
            The API path, including the leading slash (e.g. ``/repos/...``).
        method (optional):
            The HTTP method to use. Defaults to "GET".
        body (optional):
            The JSON body to send, or None for no body. Defaults to None.
        params (optional):
            Query-string parameters, or None for none. Defaults to None.

    Returns:
        The parsed JSON response, or None if the body was empty.
    """
    url = f"https://api.github.com{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {_token()}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "euroeval-queue-processor",
        },
    )
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else None


def _token() -> str:
    """Return the GitHub API token from the environment.

    Returns:
        The value of the ``GITHUB_TOKEN`` environment variable.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("GITHUB_TOKEN env var is required.")
        sys.exit(1)
    return token


if __name__ == "__main__":
    main()
