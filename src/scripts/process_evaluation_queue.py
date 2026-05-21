"""Pick up open model-evaluation-request issues and run EuroEval on them.

This script is meant to run on the compute server. For each open
``model evaluation request`` issue that is **not yet assigned** to anyone, it:

1. Verifies that the requested model exists on the Hugging Face Hub.
2. Assigns the issue to the GITHUB_TOKEN owner so the UI flips to "Evaluating".
3. Runs ``euroeval`` for each requested language group.
4. Posts the new ``euroeval_benchmark_results.jsonl`` lines as a comment on
   the issue, wrapped in a ``jsonl`` code fence so the local merge script
   can pick them up.

Required env vars
-----------------
GITHUB_TOKEN          A PAT with ``issues: write`` for the EuroEval repo.
                      Issues are assigned to the PAT owner while being
                      evaluated; the owner's login is resolved at startup
                      via the ``/user`` endpoint.
EUROEVAL_VM_ID        Optional identifier for this VM/host, written into a
                      hidden ``<!-- vm-id: ... -->`` marker on each issue
                      while it is being evaluated. Used to reclaim
                      orphaned issues after a crash without disturbing
                      work in progress on other VMs sharing the same
                      assignee. If unset, a stable id is read from (or
                      written to) a ``.env`` file in the working directory.
EUROEVAL_RESULTS_PATH Optional override for the path to
                      ``euroeval_benchmark_results.jsonl``. Defaults to
                      ``./euroeval_benchmark_results.jsonl``.
"""

from __future__ import annotations

import fcntl
import getpass
import importlib.metadata
import json
import logging
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from functools import cache
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from yaml import safe_load

from euroeval.data_models import BenchmarkResult
from leaderboards.evaluation_common import (
    GPU_FIT_OVERHEAD,
    LANGUAGE_GROUP_CODES,
    estimated_model_bytes,
    extract_language_groups,
    gpu_total_memory_bytes,
    missing_official_dataset_language_pairs,
    run_euroeval,
)
from leaderboards.paths import CORE_MODELS_CONFIG

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("process_evaluation_queue")

REPO = "EuroEval/EuroEval"
LABEL = "model evaluation request"
FAILED_LABEL = "evaluation-failed"
RESULTS_READY_LABEL = "results-ready"
TITLE_PREFIX = "[MODEL EVALUATION REQUEST]"
ASSIGNEE = ""
VM_ID = os.environ.get("EUROEVAL_VM_ID", "")
VM_ID_ENV_PATH = Path(os.environ.get("EUROEVAL_DOTENV_PATH", ".env"))
RESULTS_PATH = Path(
    os.environ.get("EUROEVAL_RESULTS_PATH", "euroeval_benchmark_results.jsonl")
)
LOCK_PATH = Path(os.environ.get("EUROEVAL_QUEUE_LOCK", "/tmp/euroeval_queue.lock"))
# Keep the lock file descriptor alive for the whole process lifetime.
_LOCK_FD: int | None = None

# Number of the issue currently being processed, used to release the
# assignment and VM marker if the script is interrupted (e.g. Ctrl-C) mid-run.
_current_issue_number: int | None = None

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

# Matches the "Model ID" section in an issue body (the form template renders
# the model id as the line immediately following a "### Model ID" heading).
MODEL_ID_BODY_RE = re.compile(r"(?:^|\n)#{1,6}\s*Model ID\s*\n+([^\n]+)")

# The frontend parses this marker on issue bodies to display "Error" /
# "Waiting for bug fix" statuses.
ERROR_MARKER_RE = re.compile(r"<!--\s*errored-on:\s*v([^\s>-]+)\s*-->")

# Per-VM ownership marker. Written before assignment, cleared on completion;
# the startup reclaim step uses it to undo claims that crashed mid-run without
# touching issues being actively processed by another VM.
VM_MARKER_RE = re.compile(r"<!--\s*vm-id:\s*([^\s>]+)\s*-->")

# The frontend parses this marker on issue bodies to display the
# "Gated model" status, which means the requested HF repo is gated and
# the configured assignee does not yet have read access.
GATED_MARKER_RE = re.compile(r"<!--\s*gated-model\s*-->")

# euroeval emits a summary line like "errored 3 benchmarks" when individual
# (dataset, language) combinations fail without crashing the whole run. We use
# this to detect partial failures even when the subprocess exited 0.
ERRORED_BENCHMARKS_RE = re.compile(r"errored\s+(\d+)\s+benchmarks?", re.IGNORECASE)

# Phrase euroeval prints when it cannot load a model because the repo is gated
# and the subprocess lacks the necessary HF token / accepted access terms.
# We treat this as "gated, please grant access" rather than as a code error.
GATED_OUTPUT_RE = re.compile(r"is a gated repository", re.IGNORECASE)


def load_dotenv_into_environ() -> None:
    """Populate ``os.environ`` from ``VM_ID_ENV_PATH`` for keys not already set.

    Existing env vars win, so an explicit ``GITHUB_TOKEN=... python ...`` override
    is honored. Malformed lines are ignored.
    """
    if not VM_ID_ENV_PATH.is_file():
        return
    try:
        for raw_line in VM_ID_ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and value and key not in os.environ:
                os.environ[key] = value
    except OSError as e:
        logger.warning(f"Could not read {VM_ID_ENV_PATH}: {e}")


def ensure_credentials() -> None:
    """Verify required env vars are set and the user is logged into HF, or exit."""
    load_dotenv_into_environ()
    if not os.environ.get("GITHUB_TOKEN"):
        token = prompt_and_persist_env_var(
            name="GITHUB_TOKEN",
            prompt_text=(
                f"GITHUB_TOKEN is required (a PAT with `issues: write` for {REPO}). "
                "Enter token"
            ),
            secret=True,
        )
        os.environ["GITHUB_TOKEN"] = token
    global ASSIGNEE
    ASSIGNEE = resolve_assignee_from_token()
    global VM_ID
    if not VM_ID:
        VM_ID = resolve_vm_id()
    logger.info(f"Using vm-id {VM_ID!r} (assignee {ASSIGNEE!r}).")
    try:
        HfApi().whoami()
    except Exception as e:  # noqa: BLE001
        logger.error(
            "Not logged in to Hugging Face. Run `huggingface-cli login` "
            f"(or set HF_TOKEN) and re-run. Underlying error: {e}"
        )
        sys.exit(1)


def resolve_assignee_from_token() -> str:
    """Return the GitHub login of the ``GITHUB_TOKEN`` owner, or exit on failure."""
    try:
        user = gh_request(path="/user")
    except urllib.error.HTTPError as e:
        logger.error(f"Could not resolve GITHUB_TOKEN owner via /user: {e}")
        sys.exit(1)
    if not isinstance(user, dict) or not user.get("login"):
        logger.error("GITHUB_TOKEN /user response did not include a login.")
        sys.exit(1)
    return str(user["login"])


def prompt_and_persist_env_var(
    name: str, prompt_text: str, secret: bool = False
) -> str:
    """Interactively prompt for an env var value and persist it to ``VM_ID_ENV_PATH``.

    Args:
        name:
            The env var name to set and persist.
        prompt_text:
            Text shown to the user before the input prompt.
        secret:
            When True, read the value via ``getpass`` so the input is not echoed.

    Returns:
        The non-empty value entered by the user.
    """
    if not sys.stdin.isatty():
        logger.error(
            f"{name} env var is required but stdin is not a TTY; cannot prompt. "
            "Set it in the environment or in the .env file and re-run."
        )
        sys.exit(1)
    if secret:
        reader = lambda: getpass.getpass(f"{prompt_text}: ")  # noqa: E731
    else:
        reader = lambda: input(f"{prompt_text}: ").strip()  # noqa: E731
    value = ""
    while not value:
        try:
            value = reader().strip()
        except (EOFError, KeyboardInterrupt):
            logger.error(f"Aborted while reading {name}.")
            sys.exit(1)
        if not value:
            print("Value cannot be empty; please try again.", file=sys.stderr)
    persist_env_var(name=name, value=value)
    return value


def persist_env_var(name: str, value: str) -> None:
    """Write ``NAME=VALUE`` to ``VM_ID_ENV_PATH``, replacing any prior entry.

    Args:
        name:
            The env var name to persist.
        value:
            The value to associate with ``name``.
    """
    try:
        existing = ""
        if VM_ID_ENV_PATH.is_file():
            existing = VM_ID_ENV_PATH.read_text(encoding="utf-8")
        lines = existing.splitlines()
        replaced = False
        for i, raw_line in enumerate(lines):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, _ = stripped.partition("=")
            if key.strip() == name:
                lines[i] = f"{name}={value}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{name}={value}")
        VM_ID_ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(f"Saved {name} to {VM_ID_ENV_PATH}.")
    except OSError as e:
        logger.warning(f"Could not persist {name} to {VM_ID_ENV_PATH}: {e}")


def resolve_vm_id() -> str:
    """Return a stable VM id, reading from or writing to a ``.env`` file.

    Resolution order:

    1. ``EUROEVAL_VM_ID`` env var (handled by the caller).
    2. ``EUROEVAL_VM_ID=...`` line in ``VM_ID_ENV_PATH``.
    3. Newly generated ``<hostname>-<8-char-uuid>``, appended to
       ``VM_ID_ENV_PATH`` so subsequent runs reuse the same id.

    Returns:
        The resolved VM id. Failures to persist a freshly generated id are
        logged and the in-memory id is still returned.
    """
    if VM_ID_ENV_PATH.is_file():
        try:
            for raw_line in VM_ID_ENV_PATH.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip() != "EUROEVAL_VM_ID":
                    continue
                value = value.strip().strip("'\"")
                if value:
                    return value
        except OSError as e:
            logger.warning(f"Could not read {VM_ID_ENV_PATH}: {e}")

    generated = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
    try:
        existing = ""
        if VM_ID_ENV_PATH.is_file():
            existing = VM_ID_ENV_PATH.read_text(encoding="utf-8")
        separator = "" if not existing or existing.endswith("\n") else "\n"
        with VM_ID_ENV_PATH.open("a", encoding="utf-8") as f:
            f.write(f"{separator}EUROEVAL_VM_ID={generated}\n")
        logger.info(f"Generated new vm-id {generated!r}, saved to {VM_ID_ENV_PATH}.")
    except OSError as e:
        logger.warning(
            f"Could not persist vm-id to {VM_ID_ENV_PATH}: {e}. "
            f"Using ephemeral vm-id {generated!r}."
        )
    return generated


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


QUEUE_PASS_SLEEP_SECONDS = 60 * 60


def main() -> None:
    """Process the queue forever, sleeping one hour between passes.

    Credentials and the single-instance lock are acquired once for the
    lifetime of the process, then :func:`process_queue_once` runs in a loop.
    """
    ensure_credentials()

    # Held for the lifetime of the process; released by the kernel on exit.
    acquire_single_instance_lock()

    # The flock guarantees no other queue processor on this host is mid-run,
    # so any issue still carrying this VM's marker is a crash-leftover.
    reclaim_orphaned_issues()

    try:
        while True:
            process_queue_once()
            logger.info(
                f"Queue pass complete; sleeping {QUEUE_PASS_SLEEP_SECONDS}s "
                "before next pass."
            )
            time.sleep(QUEUE_PASS_SLEEP_SECONDS)
    except KeyboardInterrupt:
        logger.info("Interrupted; releasing current issue and exiting.")
        release_current_issue()
        sys.exit(130)


def release_current_issue() -> None:
    """Clear the VM marker and unassign on the issue currently being processed.

    Used by the interrupt handler so Ctrl-C returns the in-flight issue to the
    queue instead of leaving it assigned to this VM until the next reclaim
    pass.
    """
    global _current_issue_number
    number = _current_issue_number
    if number is None:
        return
    _current_issue_number = None
    try:
        clear_vm_marker(number=number)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"#{number}: could not clear vm marker on interrupt: {e}")
    try:
        unassign_issue(number=number)
        logger.info(f"#{number}: released on interrupt.")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"#{number}: could not unassign on interrupt: {e}")


def process_queue_once() -> None:
    """Process every unassigned model-evaluation-request issue once.

    Issues are sorted by (status priority asc, parameter count asc,
    num-language-groups asc). Status priority is 0 for fresh issues, 1 for
    gated repos (cheap marker refresh), and 2 for retries of previously
    errored evaluations, so that quicker work is picked up first and gated
    items are surfaced ahead of errored ones.
    """
    try:
        issues = list_open_unassigned_issues()
    except urllib.error.HTTPError as e:
        logger.error(f"Failed to list issues: {e}")
        return

    existing_lines = read_jsonl_lines(path=RESULTS_PATH)
    candidates: list[tuple[int, int, int, int, dict, str, list[str]]] = []
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

        param_count = summary["param_count"]
        if summary.get("gated"):
            status_priority = 1
        elif errored_marker_present(body=body):
            status_priority = 2
        else:
            status_priority = 0
        # Among 'waiting' candidates, push models with partial results ahead so
        # we finish what we already started before claiming a new evaluation.
        languages: list[str] = sorted(
            {code for g in groups for code in LANGUAGE_GROUP_CODES[g]}
        )
        if status_priority == 0 and model_has_partial_results(
            lines=existing_lines, model_id=model_id, requested_languages=languages
        ):
            partial_rank = 0
        else:
            partial_rank = 1
        candidates.append(
            (
                status_priority,
                partial_rank,
                param_count,
                len(groups),
                issue,
                model_id,
                groups,
            )
        )

    candidates.sort(key=lambda c: (c[0], c[1], c[2], c[3]))
    logger.info(f"Found {len(candidates)} processable issue(s).")

    gpu_bytes = gpu_total_memory_bytes()
    if gpu_bytes is None:
        logger.info(
            "Could not determine local memory budget; skipping the fit pre-check."
        )
    else:
        logger.info(f"Local memory budget: {gpu_bytes / (1024**3):.1f} GiB.")

    for (
        status_priority,
        partial_rank,
        param_count,
        num_groups,
        issue,
        model_id,
        groups,
    ) in candidates:
        status = {0: "fresh", 1: "gated", 2: "retry of errored eval"}[status_priority]
        if status_priority == 0 and partial_rank == 0:
            status = "resuming partial"
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


def reclaim_orphaned_issues() -> None:
    """Unassign and clear the VM marker on issues this VM crashed mid-run on.

    Lists issues currently assigned to ``ASSIGNEE`` and, for each one whose
    body carries this VM's ``vm-id`` marker, clears the marker and removes
    the assignee so the issue returns to the queue. Issues owned by other
    VMs (different ``vm-id`` value) are left alone.
    """
    try:
        issues = gh_request(
            path=f"/repos/{REPO}/issues",
            params={
                "state": "open",
                "labels": LABEL,
                "per_page": "100",
                "assignee": ASSIGNEE,
            },
        )
    except urllib.error.HTTPError as e:
        logger.warning(f"Could not list assigned issues for reclaim: {e}")
        return
    if not isinstance(issues, list):
        return
    reclaimed = 0
    for issue in issues:
        if not isinstance(issue, dict) or "pull_request" in issue:
            continue
        body = issue.get("body") or ""
        m = VM_MARKER_RE.search(body)
        if not m or m.group(1) != VM_ID:
            continue
        number = issue["number"]
        try:
            clear_vm_marker(number=number)
            unassign_issue(number=number)
        except urllib.error.HTTPError as e:
            logger.warning(f"#{number}: failed to reclaim: {e}")
            continue
        reclaimed += 1
        logger.info(f"#{number}: reclaimed orphaned issue (vm-id {VM_ID}).")
    if reclaimed:
        logger.info(f"Reclaimed {reclaimed} orphaned issue(s) for vm-id {VM_ID}.")


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


def errored_marker_present(body: str | None) -> bool:
    """Return True if the body carries an ``errored-on`` marker.

    Args:
        body:
            The markdown body of the issue, or None.

    Returns:
        True if the marker is present, False otherwise.
    """
    return bool(body) and bool(ERROR_MARKER_RE.search(body))


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
        - ``gated`` (bool, True when the configured assignee does not have
          read access to a gated repo -- i.e. ``model_info`` raised
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
            logger.info(f"#{number}: marked gated -- {ASSIGNEE} lacks read access.")
        else:
            logger.info(f"#{number}: still gated -- leaving marker in place.")
        return
    if body_gated:
        clear_gated_marker(number=number, body=body)
        logger.info(f"#{number}: access granted, cleared gated marker.")

    logger.info(f"#{number}: claiming issue for {model_id!r}, languages={languages}")
    # Set the VM marker BEFORE assigning so a crash between the two leaves
    # the issue unassigned (harmless) rather than assigned-but-unowned.
    set_vm_marker(number=number)
    assign_issue(number=number)
    global _current_issue_number
    _current_issue_number = number
    try:
        _run_claimed_issue(issue=issue, model_id=model_id, languages=languages)
    finally:
        _current_issue_number = None


PROGRESS_COMMENT_MARKER = "<!-- queue-progress -->"

# Gist marker used to store and retrieve the gist ID across issue body updates.
GIST_MARKER_RE = re.compile(r"<!--\s*euroeval-results-gist:\s*([^\s>]+)\s*-->")

# Gist ID for uploading results as an attachment. Stored in the issue body
# so it persists across runs and can be cleaned up when the issue is done.
_gist_id: str | None = None


def _run_claimed_issue(issue: dict, model_id: str, languages: list[str]) -> None:
    """Run euroeval one language at a time and maintain a progress comment.

    A single comment carrying the ``queue-progress`` marker is created (or
    reused) and rewritten after every language so the issue always shows the
    completed/in-progress/remaining/failed split along with a link to a
    gist that holds the accumulated JSONL results.

    Args:
        issue:
            The GitHub issue object returned by the API.
        model_id:
            The Hugging Face model id to evaluate.
        languages:
            The flattened list of language codes for this evaluation.
    """
    number = issue["number"]
    is_core = model_id in load_core_model_ids()
    issue_body = issue.get("body")

    existing_lines = _result_lines_for_model(
        lines=read_jsonl_lines(path=RESULTS_PATH), model_id=model_id
    )
    accumulated = list(existing_lines)
    done = _completed_languages(lines=accumulated, requested_languages=languages)
    pending = [lang for lang in languages if lang not in done]
    failed: list[str] = []

    comment_id = _find_progress_comment(number=number)
    comment_id = _post_or_update_progress_comment(
        number=number,
        comment_id=comment_id,
        model_id=model_id,
        done=done,
        current=None,
        remaining=pending,
        failed=failed,
        lines=accumulated,
        issue_body=issue_body,
    )

    gated_detected = False
    failure_reason: str | None = None
    failure_output_tail = ""

    try:
        for i, lang in enumerate(pending):
            remaining_after = pending[i + 1 :]
            comment_id = _post_or_update_progress_comment(
                number=number,
                comment_id=comment_id,
                model_id=model_id,
                done=done,
                current=lang,
                remaining=remaining_after,
                failed=failed,
                lines=accumulated,
                issue_body=issue_body,
            )

            before = set(read_jsonl_lines(path=RESULTS_PATH))
            # Only ask euroeval to clear its model cache after the final language so
            # the model is downloaded once per issue, not once per language.
            is_last = i == len(pending) - 1
            returncode, output = run_euroeval(
                model_id=model_id,
                languages=[lang],
                evaluate_test_split=is_core,
                clear_model_cache=is_last,
            )
            after = read_jsonl_lines(path=RESULTS_PATH)
            new_lines = [line for line in after if line not in before]
            accumulated.extend(new_lines)

            if GATED_OUTPUT_RE.search(output):
                gated_detected = True
                failure_output_tail = output[-6000:].strip() or "(no output captured)"
                break

            num_errored = num_errored_benchmarks(output=output)
            missing = missing_official_dataset_language_pairs(
                lines=accumulated, requested_languages=[lang]
            )
            if returncode != 0:
                reason = f"euroeval exited with code {returncode} for language {lang!r}"
                failure_reason = failure_reason or reason
                failure_output_tail = output[-6000:].strip() or "(no output captured)"
                failed.append(lang)
            elif num_errored > 0:
                reason = (
                    f"euroeval reported {num_errored} errored benchmark(s) "
                    f"for language {lang!r}"
                )
                failure_reason = failure_reason or reason
                failure_output_tail = output[-6000:].strip() or "(no output captured)"
                failed.append(lang)
            elif missing:
                reason = (
                    f"missing official dataset-language pair(s) for {lang!r}: "
                    f"{format_dataset_language_pairs(dataset_language_pairs=missing)}"
                )
                failure_reason = failure_reason or reason
                failure_output_tail = output[-6000:].strip() or "(no output captured)"
                failed.append(lang)
            else:
                done.append(lang)

            comment_id = _post_or_update_progress_comment(
                number=number,
                comment_id=comment_id,
                model_id=model_id,
                done=done,
                current=None,
                remaining=remaining_after,
                failed=failed,
                lines=accumulated,
                issue_body=issue_body,
            )

        if gated_detected:
            version = euroeval_version()
            set_gated_with_errored_block(
                number=number, body=issue.get("body"), version=version
            )
            add_failed_label(number=number)
            clear_vm_marker(number=number)
            unassign_issue(number=number)
            logger.info(
                f"#{number}: euroeval reported a gated repo for {model_id!r}; "
                f"marked gated and errored-on v{version} to avoid retry loops."
            )
            return

        if failed:
            version = euroeval_version()
            reason = failure_reason or f"failed languages: {', '.join(failed)}"
            tail = failure_output_tail or "(no output captured)"
            if issue_has_matching_error_comment(number=number, reason=reason):
                clear_vm_marker(number=number)
                unassign_issue(number=number)
                logger.info(
                    f"#{number}: identical error already posted; "
                    "skipping comment and marker updates, returned to queue."
                )
                return
            error_comment = (
                f"Error encountered during evaluation ({reason}):\n\n"
                f"```bash\n{tail}\n```\n\n"
                f"EuroEval version: v{version}\n"
            )
            comment_on_issue(number=number, comment=error_comment)
            set_errored_marker(number=number, body=issue.get("body"), version=version)
            add_failed_label(number=number)
            clear_vm_marker(number=number)
            unassign_issue(number=number)
            logger.info(
                f"#{number}: marked errored on v{version} after {len(failed)} failed "
                f"language(s) ({', '.join(failed)}); returned to queue."
            )
            return

        remove_failed_label(number=number)
        add_results_ready_label(number=number)
        clear_vm_marker(number=number)
        logger.info(
            f"#{number}: completed all {len(done)} language(s) for {model_id!r}; "
            "progress comment updated."
        )
    finally:
        _delete_gist()


def _result_lines_for_model(lines: list[str], model_id: str) -> list[str]:
    """Return the subset of ``lines`` whose parsed ``model`` matches ``model_id``."""
    out: list[str] = []
    for line in lines:
        try:
            parsed = BenchmarkResult.from_dict(config=json.loads(line))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if parsed.model == model_id:
            out.append(line)
    return out


def _completed_languages(lines: list[str], requested_languages: list[str]) -> list[str]:
    """Return requested languages whose official pair coverage is complete."""
    missing = missing_official_dataset_language_pairs(
        lines=lines, requested_languages=requested_languages
    )
    incomplete = {lang for _, lang in missing}
    return [lang for lang in requested_languages if lang not in incomplete]


def _ensure_gist_created() -> str | None:
    """Create or return the existing gist ID for uploading results.

    Returns:
        The gist ID if a gist was created or found, or None on failure.
    """
    global _gist_id
    if _gist_id:
        return _gist_id

    # Check issue body for an existing gist marker.
    # This is done lazily on first call so we don't need to fetch the body
    # upfront for every issue.
    return None


def _upload_results_gist(
    model_id: str, lines: list[str], issue_body: str | None = None
) -> str | None:
    """Upload the JSONL results as a GitHub Gist and return its ID.

    The gist ID is stored in the issue body via a marker so it persists
    across script restarts and can be cleaned up on issue completion.

    Args:
        model_id:
            The Hugging Face model id used to name the gist file.
        lines:
            The JSONL result lines to upload.
        issue_body:
            The current issue body (used to store the gist marker).

    Returns:
        The gist ID if the upload succeeded, or None on failure.
    """
    global _gist_id

    if _gist_id:
        return _gist_id

    filename = f"{model_id.replace('/', '_').replace('.', '_')}_results.jsonl"
    content = "\n".join(lines) + "\n" if lines else ""

    try:
        resp = gh_request(
            path="/gists",
            method="POST",
            body={
                "description": f"EuroEval results for {model_id}",
                "files": {filename: {"content": content}},
                "public": False,
            },
        )
        if isinstance(resp, dict):
            gist_id = resp.get("id")
            if isinstance(gist_id, str) and gist_id:
                _gist_id = gist_id
                # Store the gist ID in the issue body so it persists across runs.
                if issue_body:
                    cleaned = GIST_MARKER_RE.sub("", issue_body).rstrip()
                    new_body = (
                        f"{cleaned}\n\n<!-- euroeval-results-gist: {gist_id} -->\n"
                    )
                    try:
                        gh_request(
                            path=f"/repos/{REPO}/issues/{_current_issue_number}",
                            method="PATCH",
                            body={"body": new_body},
                        )
                        logger.info(f"Stored gist marker {gist_id} in issue body.")
                    except urllib.error.HTTPError as e:
                        logger.warning(
                            f"Could not store gist marker in issue body: {e}"
                        )
                logger.info(f"Created results gist {gist_id} for {model_id!r}.")
                return gist_id
            return None
        return None
    except urllib.error.HTTPError as e:
        logger.warning(f"Could not create results gist for {model_id!r}: {e}")
        return None


def _delete_gist() -> None:
    """Delete the results gist if one was created.

    Called when the issue is done (success or failure) so the gist doesn't
    linger after the evaluation is finished.
    """
    global _gist_id
    if not _gist_id:
        return
    try:
        gh_request(path=f"/gists/{_gist_id}", method="DELETE")
        logger.info(f"Deleted results gist {_gist_id}.")
    except urllib.error.HTTPError as e:
        logger.warning(f"Could not delete results gist {_gist_id}: {e}")
    _gist_id = None


def _find_progress_comment(number: int) -> int | None:
    """Return the id of the existing progress comment on ``number``, or None."""
    try:
        comments = gh_request(
            path=f"/repos/{REPO}/issues/{number}/comments", params={"per_page": "100"}
        )
    except urllib.error.HTTPError as e:
        logger.warning(f"#{number}: could not list comments: {e}")
        return None
    if not isinstance(comments, list):
        return None
    for c in comments:
        if not isinstance(c, dict):
            continue
        if PROGRESS_COMMENT_MARKER in (c.get("body") or ""):
            cid = c.get("id")
            if isinstance(cid, int):
                return cid
    return None


def _render_progress_comment(
    model_id: str,
    done: list[str],
    current: str | None,
    remaining: list[str],
    failed: list[str],
    lines: list[str],
    gist_url: str | None = None,
) -> str:
    """Render the progress comment body for the given evaluation state.

    Args:
        model_id:
            The Hugging Face model id.
        done:
            List of completed language codes.
        current:
            The currently processing language code, or None.
        remaining:
            List of remaining language codes.
        failed:
            List of failed language codes.
        lines:
            The accumulated JSONL result lines (used to create the gist).
        gist_url:
            URL to the results gist, or None if no gist has been created yet.

    Returns:
        The markdown body of the progress comment, including the hidden
        ``queue-progress`` marker and a link to the results gist.
    """

    def fmt(langs: list[str]) -> str:
        return ", ".join(langs) if langs else "(none)"

    body = (
        f"{PROGRESS_COMMENT_MARKER}\n"
        f"**Evaluation progress for `{model_id}`**\n\n"
        f"- Completed: {fmt(done)}\n"
        f"- In progress: {current if current else '(none)'}\n"
        f"- Remaining: {fmt(remaining)}\n"
        f"- Failed: {fmt(failed)}\n"
    )
    if gist_url:
        body += f"\n\n[Benchmark results gist]({gist_url})\n"

    return body


def _post_or_update_progress_comment(
    number: int,
    comment_id: int | None,
    model_id: str,
    done: list[str],
    current: str | None,
    remaining: list[str],
    failed: list[str],
    lines: list[str],
    issue_body: str | None = None,
) -> int | None:
    """Create or PATCH the progress comment for the given issue.

    Returns:
        The id of the progress comment if it could be created or updated,
        otherwise None (e.g. when the API call failed and the existing
        ``comment_id`` was not known).
    """
    gist_url = None
    if lines:
        gist_id = _upload_results_gist(
            model_id=model_id, lines=lines, issue_body=issue_body
        )
        if gist_id:
            gist_url = f"https://gist.github.com/{gist_id}"

    body = _render_progress_comment(
        model_id=model_id,
        done=done,
        current=current,
        remaining=remaining,
        failed=failed,
        lines=lines,
        gist_url=gist_url,
    )
    try:
        if comment_id is None:
            resp = gh_request(
                path=f"/repos/{REPO}/issues/{number}/comments",
                method="POST",
                body={"body": body},
            )
            if isinstance(resp, dict):
                cid = resp.get("id")
                if isinstance(cid, int):
                    return cid
            return None
        gh_request(
            path=f"/repos/{REPO}/issues/comments/{comment_id}",
            method="PATCH",
            body={"body": body},
        )
        return comment_id
    except urllib.error.HTTPError as e:
        logger.warning(f"#{number}: could not update progress comment: {e}")
        return comment_id


def model_has_partial_results(
    lines: list[str], model_id: str, requested_languages: list[str]
) -> bool:
    """Return True if ``model_id`` has some but not all expected result lines.

    "Expected" here is the official dataset/language pairs for the requested
    languages, matching the completeness check used after an evaluation run.

    Args:
        lines:
            The current contents of ``euroeval_benchmark_results.jsonl``.
        model_id:
            The Hugging Face model id whose existing results we inspect.
        requested_languages:
            The flattened language codes selected on the issue.

    Returns:
        True when at least one matching result line exists and at least one
        official pair is still missing; False otherwise.
    """
    matching: list[str] = []
    for line in lines:
        try:
            parsed = BenchmarkResult.from_dict(config=json.loads(line))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if parsed.model == model_id:
            matching.append(line)
    if not matching:
        return False
    missing = missing_official_dataset_language_pairs(
        lines=matching, requested_languages=requested_languages
    )
    return bool(missing)


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


def add_results_ready_label(number: int) -> None:
    """Attach the ``results-ready`` label to an issue.

    Args:
        number:
            The issue number to label.
    """
    try:
        gh_request(
            path=f"/repos/{REPO}/issues/{number}/labels",
            method="POST",
            body={"labels": [RESULTS_READY_LABEL]},
        )
    except urllib.error.HTTPError as e:
        logger.warning(f"#{number}: could not add {RESULTS_READY_LABEL!r} label: {e}")


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


def issue_has_matching_error_comment(number: int, reason: str) -> bool:
    """Return True if an error comment with the same ``reason`` already exists.

    The tail of subprocess output varies run-to-run (timestamps, ANSI),
    so we match on the stable error-reason phrase rendered in the comment
    header instead of doing an exact-body comparison.

    Args:
        number:
            The issue number to inspect.
        reason:
            The reason string that would be used in a new error comment.

    Returns:
        True if any existing comment on the issue contains the same
        ``Error encountered during evaluation (<reason>):`` header.
    """
    try:
        comments = gh_request(
            path=f"/repos/{REPO}/issues/{number}/comments", params={"per_page": "100"}
        )
    except urllib.error.HTTPError as e:
        logger.warning(f"#{number}: could not list comments: {e}")
        return False
    if not isinstance(comments, list):
        return False
    marker = f"Error encountered during evaluation ({reason}):"
    return any(
        isinstance(c, dict) and marker in (c.get("body") or "") for c in comments
    )


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


def set_vm_marker(number: int) -> None:
    """Stamp the issue body with this VM's ``vm-id`` marker.

    The issue body is re-fetched so concurrent updates earlier in the same
    ``process_issue`` call (e.g. clearing a gated marker) are not clobbered.

    Args:
        number:
            The issue number to mark.
    """
    body = _fetch_issue_body(number=number)
    cleaned = VM_MARKER_RE.sub("", body).rstrip()
    new_body = f"{cleaned}\n\n<!-- vm-id: {VM_ID} -->\n"
    gh_request(
        path=f"/repos/{REPO}/issues/{number}", method="PATCH", body={"body": new_body}
    )


def clear_vm_marker(number: int) -> None:
    """Remove this VM's ``vm-id`` marker from the issue body.

    Re-fetches the body before patching so other markers set during the run
    are preserved. Markers belonging to other VMs are left untouched.

    Args:
        number:
            The issue number to clean.
    """
    body = _fetch_issue_body(number=number)
    m = VM_MARKER_RE.search(body)
    if not m or m.group(1) != VM_ID:
        return
    cleaned = VM_MARKER_RE.sub("", body, count=1).rstrip() + "\n"
    gh_request(
        path=f"/repos/{REPO}/issues/{number}", method="PATCH", body={"body": cleaned}
    )


def _fetch_issue_body(number: int) -> str:
    """Return the current body of an issue, or empty string on failure."""
    try:
        current = gh_request(path=f"/repos/{REPO}/issues/{number}")
    except urllib.error.HTTPError as e:
        logger.warning(f"#{number}: could not fetch issue body: {e}")
        return ""
    if isinstance(current, dict):
        return current.get("body") or ""
    return ""


@cache
def load_core_model_ids() -> frozenset[str]:
    """Return the set of model ids listed as core models in ``core_models.yaml``.

    Core models are evaluated on the test split; every other model is run on
    the validation split.
    """
    try:
        with CORE_MODELS_CONFIG.open("r", encoding="utf-8") as f:
            config = safe_load(f)
    except OSError as e:
        logger.warning(f"Could not read {CORE_MODELS_CONFIG}: {e}")
        return frozenset()
    if not isinstance(config, dict):
        return frozenset()
    models = config.get("models") or []
    ids: set[str] = set()
    for entry in models:
        if isinstance(entry, dict):
            model_id = entry.get("id")
            if isinstance(model_id, str) and model_id:
                ids.add(model_id)
    return frozenset(ids)


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
