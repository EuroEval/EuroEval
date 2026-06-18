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
HF_TOKEN              A Hugging Face token with read access to any gated
                      repos that are expected to be evaluated. Used both
                      for Hub metadata lookups and for downloads inside
                      the ``euroeval`` subprocess. ``HUGGINGFACE_API_KEY``
                      is accepted as an alias.
EUROEVAL_VM_ID        Optional identifier for this VM/host, written into a
                      hidden ``<!-- vm-id: ... -->`` marker on each issue
                      while it is being evaluated. Used to reclaim
                      orphaned issues after a crash without disturbing
                      work in progress on other VMs sharing the same
                      assignee. If unset, a stable id is read from (or
                      written to) a ``.env`` file in the working directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
import time
import urllib.error
from functools import cache
from pathlib import Path

from huggingface_hub import BucketFile, HfApi
from huggingface_hub.errors import HfHubHTTPError
from yaml import safe_load

from euroeval import __version__
from leaderboards.constants import (
    CORE_MODELS_CONFIG,
    FAILED_LABEL,
    GATED_LABEL,
    GATED_OUTPUT_RE,
    GPU_FIT_OVERHEAD,
    LANGUAGE_GROUP_CODES,
    MODEL_REQUEST_LABEL,
    REPO,
    RESULTS_DIR,
    RESULTS_READY_LABEL,
    VM_MARKER_RE,
)
from leaderboards.evaluation_common import (
    estimated_model_bytes,
    extract_language_groups,
    gpu_total_memory_bytes,
    missing_official_dataset_language_pairs,
    run_euroeval,
)
from leaderboards.github_api import (
    add_failed_label,
    add_gated_label,
    add_results_ready_label,
    assign_issue,
    comment_on_issue,
    gh_request,
    remove_failed_label,
    remove_gated_label,
    unassign_issue,
)
from leaderboards.queue_env import (
    acquire_single_instance_lock,
    load_dotenv_into_environ,
    prompt_and_persist_env_var,
    resolve_assignee_from_token,
    resolve_vm_id,
)
from leaderboards.queue_hf_cache import cached_model_summary
from leaderboards.queue_markers import (
    clear_vm_marker,
    release_issue_if_owned,
    set_vm_marker,
    vm_marker_matches,
)
from leaderboards.queue_parsing import (
    completed_languages,
    extract_model_id,
    format_dataset_language_pairs,
    num_errored_benchmarks,
    num_skipped_benchmarks,
    read_jsonl_lines,
    result_lines_for_model,
)
from leaderboards.queue_runtime import (
    ThermalConfig,
    cool_down_between_issues,
    lower_process_priority,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("process_evaluation_queue")


ASSIGNEE = ""
VM_ID = os.environ.get("EUROEVAL_VM_ID", "")
VM_ID_ENV_PATH = Path(os.environ.get("EUROEVAL_DOTENV_PATH", ".env"))
RESULTS_PATH = Path("euroeval_benchmark_results.jsonl")
RESULTS_CACHE_DIR = RESULTS_DIR
LOCK_PATH = Path(os.environ.get("EUROEVAL_QUEUE_LOCK", "/tmp/euroeval_queue.lock"))

# Canonical HF bucket for storing results (public read access).
HF_RESULTS_BUCKET = "EuroEval/results"

# Held for the lifetime of the process so the kernel keeps the queue lock
# alive; released automatically when the process exits.
_LOCK_FD: int | None = None

# Issue currently being processed, used to release the assignment and VM
# marker if the script is interrupted (e.g. Ctrl-C) mid-run.
_current_issue_number: int | None = None

QUEUE_PASS_SLEEP_SECONDS = 60 * 60

# Runtime overrides set from CLI args in main(). None means "use the
# euroeval CLI default" for the memory utilization knob.
GPU_MEMORY_UTILIZATION: float | None = None
THERMAL_CONFIG: ThermalConfig = ThermalConfig()


# Param bucket thresholds matching leaderboards (src/leaderboards/core_models.py)
_BUCKET_THRESHOLDS = [
    (2_000_000_000, 0),  # tiny
    (10_000_000_000, 1),  # small
    (40_000_000_000, 2),  # medium
    (80_000_000_000, 3),  # large
    (float("inf"), 4),  # xlarge
]


def main() -> None:
    """Process the queue forever, sleeping one hour between passes.

    Credentials and the single-instance lock are acquired once for the
    lifetime of the process, then :func:`process_queue_once` runs in a loop.
    """
    parse_args()
    lower_process_priority()
    ensure_credentials()

    # Download the canonical results file from HF before starting.
    download_results_from_hf()

    global _LOCK_FD
    _LOCK_FD = acquire_single_instance_lock(lock_path=LOCK_PATH)

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


def _param_bucket(param_count: int) -> int:
    """Map parameter count to bucket index for queue sorting.

    Args:
        param_count:
            Number of parameters in the model.

    Returns:
        Bucket index (0=tiny, 1=small, 2=medium, 3=large, 4=xlarge).
    """
    for threshold, bucket in _BUCKET_THRESHOLDS:
        if param_count < threshold:
            return bucket
    return 4


def _model_id_to_filename(model_id: str) -> str:
    """Convert a model ID to a safe filename.

    Args:
        model_id:
            The model identifier (e.g., "meta-llama/Llama-3-8B").

    Returns:
        A safe filename with slashes and dots replaced by underscores.
    """
    return model_id.replace("/", "_") + ".jsonl"


def download_results_from_hf() -> int:
    """Download all results from the Hugging Face bucket.

    Only downloads files, never deletes local files. This is a one-way sync
    from bucket to local cache.

    Returns:
        The number of lines loaded.
    """
    RESULTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    bucket_id = HF_RESULTS_BUCKET
    try:
        # List all files in the bucket
        bucket_files = list(api.list_bucket_tree(bucket_id=bucket_id, recursive=True))
        # Build download list for .jsonl files not already present
        files_to_download: list[tuple[str | BucketFile, str | Path]] = []
        for file_info in bucket_files:
            if not file_info.path.endswith(".jsonl"):
                continue
            local_file = RESULTS_CACHE_DIR / file_info.path
            # Only download if not already present (additive only)
            if not local_file.exists():
                files_to_download.append((file_info.path, str(local_file)))

        if files_to_download:
            api.download_bucket_files(bucket_id=bucket_id, files=files_to_download)
    except HfHubHTTPError as e:
        logger.warning(f"Could not download results from HF bucket: {e}")
        return 0

    all_lines: list[str] = []
    for model_file in RESULTS_CACHE_DIR.glob("*.jsonl"):
        lines = model_file.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.strip():
                all_lines.append(line)

    if all_lines:
        RESULTS_PATH.write_text("\n".join(all_lines) + "\n", encoding="utf-8")

    num_models = len(list(RESULTS_CACHE_DIR.glob("*.jsonl")))
    logger.info(
        f"Downloaded {len(all_lines):,} result lines from {num_models} model(s) "
        f"in bucket {HF_RESULTS_BUCKET!r}."
    )
    return len(all_lines)


def parse_args() -> None:
    """Parse CLI arguments and populate the module-level runtime overrides."""
    global GPU_MEMORY_UTILIZATION
    global THERMAL_CONFIG
    parser = argparse.ArgumentParser(
        description="Pick up and evaluate open model-evaluation-request issues."
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help=(
            "vLLM GPU memory utilization fraction (0.0-1.0). When omitted, "
            "the euroeval CLI's own default is used."
        ),
    )
    parser.add_argument(
        "--inter-issue-sleep",
        type=float,
        default=THERMAL_CONFIG.inter_issue_sleep_seconds,
        help="Seconds to wait between issues regardless of thermal state.",
    )
    parser.add_argument(
        "--thermal-pause-temp",
        type=float,
        default=THERMAL_CONFIG.pause_temp_c,
        help=(
            "GPU temperature (deg C) at or above which to pause before the next issue."
        ),
    )
    parser.add_argument(
        "--thermal-resume-temp",
        type=float,
        default=THERMAL_CONFIG.resume_temp_c,
        help="GPU temperature (deg C) the GPU must cool to before resuming.",
    )
    args = parser.parse_args()
    GPU_MEMORY_UTILIZATION = args.gpu_memory_utilization
    THERMAL_CONFIG = ThermalConfig(
        inter_issue_sleep_seconds=args.inter_issue_sleep,
        pause_temp_c=args.thermal_pause_temp,
        resume_temp_c=args.thermal_resume_temp,
    )


def ensure_credentials() -> None:
    """Verify required env vars are set and the user is logged into HF, or exit."""
    load_dotenv_into_environ(env_path=VM_ID_ENV_PATH)
    if not os.environ.get("GITHUB_TOKEN"):
        token = prompt_and_persist_env_var(
            env_path=VM_ID_ENV_PATH,
            name="GITHUB_TOKEN",
            prompt_text=(
                f"GITHUB_TOKEN is required (a PAT with `issues: write` for {REPO}). "
                "Enter token"
            ),
            secret=True,
        )
        os.environ["GITHUB_TOKEN"] = token
    # Accept HF_TOKEN or HUGGINGFACE_API_KEY; normalize to HF_TOKEN for huggingface_hub.
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if not hf_token:
        token = prompt_and_persist_env_var(
            env_path=VM_ID_ENV_PATH,
            name="HF_TOKEN",
            prompt_text=(
                "HF_TOKEN is required (a Hugging Face token with read access to gated "
                "repos you intend to evaluate; HUGGINGFACE_API_KEY is also accepted). "
                "Enter token"
            ),
            secret=True,
        )
        os.environ["HF_TOKEN"] = token
    elif not os.environ.get("HF_TOKEN"):
        # HUGGINGFACE_API_KEY was set, but not HF_TOKEN; copy for huggingface_hub.
        os.environ["HF_TOKEN"] = hf_token
    global ASSIGNEE
    ASSIGNEE = resolve_assignee_from_token()
    global VM_ID
    if not VM_ID:
        VM_ID = resolve_vm_id(env_path=VM_ID_ENV_PATH)
    logger.info(f"Using vm-id {VM_ID!r} (assignee {ASSIGNEE!r}).")
    try:
        HfApi().whoami()
    except Exception as e:  # noqa: BLE001
        # Any auth/network failure here means we cannot proceed; report it and
        # exit cleanly rather than crash with a traceback.
        logger.error(
            "Not logged in to Hugging Face. Run `huggingface-cli login` "
            f"(or set HF_TOKEN) and re-run. Underlying error: {e}"
        )
        sys.exit(1)


def release_current_issue() -> None:
    """Clear the VM marker and unassign on the issue currently being processed.

    Used by the interrupt handler so Ctrl-C returns the in-flight issue to
    the queue instead of leaving it assigned to this VM until the next
    reclaim pass.
    """
    global _current_issue_number
    number = _current_issue_number
    if number is None:
        return
    _current_issue_number = None
    try:
        if release_issue_if_owned(number=number, vm_id=VM_ID, assignee=ASSIGNEE):
            logger.info(f"#{number}: released on interrupt.")
    except Exception as e:  # noqa: BLE001
        # Best-effort cleanup from an interrupt handler: never let a release
        # failure mask the original Ctrl-C or crash the shutdown path.
        logger.warning(f"#{number}: could not release on interrupt: {e}")


def age_sort_value(issue: dict) -> float:
    """Return a sort value that orders the oldest issues first.

    Used as the final queue tiebreaker so that, all else being equal,
    the longest-waiting request is picked up first and stale models are
    drained from the queue rather than left to accumulate.

    Args:
        issue:
            The GitHub issue object returned by the API.

    Returns:
        The ``created_at`` epoch (seconds), or ``float("inf")`` when the
        timestamp is missing or unparseable, so such issues sort after
        those with a known creation time under the ascending
        ``candidates.sort`` ordering.
    """
    created_at = issue.get("created_at")
    if not isinstance(created_at, str):
        return float("inf")
    try:
        parsed = dt.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return float("inf")
    return parsed.timestamp()


def process_queue_once() -> None:
    """Process every unassigned model-evaluation-request issue once.

    Issues are sorted by (slow priority asc, status priority asc,
    parameter bucket asc, age asc). Parameter buckets match the
    leaderboards: tiny (<2B), small (2-10B), medium (10-40B),
    large (40-80B), xlarge (>=80B).
    Slow priority is 0 for normal issues and 1 for issues with the 'slow'
    label, pushing slow evaluations to the end of the queue.
    Status priority is 0 for gated repos (cheap marker refresh),
    1 for retries of previously errored evaluations (issues with the
    ``evaluation-failed`` label), and 2 for fresh issues, so that gated repos
    are surfaced first, then retries, then new work.
    Age is a final tiebreaker so that, when everything else is equal,
    the oldest (longest-waiting) issue is picked up first and stale requests
    don't linger in the queue.
    """
    try:
        issues = list_open_unassigned_issues()
    except urllib.error.HTTPError as e:
        logger.error(f"Failed to list issues: {e}")
        return

    candidates: list[tuple[int, int, int, float, dict, str, list[str]]] = []
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

        if summary.get("gguf"):
            logger.info(
                f"#{number}: skipping -- {model_id!r} is a GGUF model, which the "
                "evaluation queue cannot run."
            )
            continue

        param_count = summary["param_count"]
        if summary.get("gated"):
            status_priority = 0
        elif issue_has_failed_label(issue=issue):
            status_priority = 1
        else:
            status_priority = 2
        slow_priority = (
            1
            if any(label["name"] == "slow" for label in issue.get("labels", []))
            else 0
        )

        param_bucket_idx = _param_bucket(param_count)
        candidates.append(
            (
                slow_priority,
                status_priority,
                param_bucket_idx,
                age_sort_value(issue=issue),
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
        slow_priority,
        status_priority,
        param_count,
        _age,
        issue,
        model_id,
        groups,
    ) in candidates:
        status = {0: "gated", 1: "retry of errored eval", 2: "fresh"}[status_priority]
        slow_tag = ", slow" if slow_priority else ""
        logger.info(
            f"#{issue['number']}: queueing {model_id!r} ({param_count} params, "
            f"{status}{slow_tag})."
        )
        if gpu_bytes is not None:
            needed = estimated_model_bytes(model_id=model_id)
            if needed is not None and int(needed * GPU_FIT_OVERHEAD) > gpu_bytes:
                logger.info(
                    f"#{issue['number']}: skipping -- model {model_id!r} needs "
                    f"~{needed / (1024**3):.1f} GiB of weights "
                    f"(x {GPU_FIT_OVERHEAD} overhead), which exceeds the local "
                    f"GPU memory of {gpu_bytes / (1024**3):.1f} GiB. Leaving the "
                    "issue unassigned so a larger machine can pick it up."
                )
                continue
        try:
            process_issue(issue=issue, model_id=model_id, groups=groups)
        except Exception as e:  # noqa: BLE001
            # Top-level per-issue guard: one failing issue must not abort the
            # whole queue loop, so log it and move on to the next issue.
            logger.exception(f"Error while processing issue #{issue['number']}: {e}")
        cool_down_between_issues(config=THERMAL_CONFIG)


def reclaim_orphaned_issues() -> None:
    """Unassign and clear the VM marker on issues this VM crashed mid-run on.

    Lists issues currently assigned to ``ASSIGNEE`` and, for each one
    whose body carries this VM's ``vm-id`` marker, clears the marker and
    removes the assignee so the issue returns to the queue. Issues owned
    by other VMs (different ``vm-id`` value) are left alone.
    """
    try:
        issues = gh_request(
            path=f"/repos/{REPO}/issues",
            params={
                "state": "open",
                "labels": MODEL_REQUEST_LABEL,
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
        labels = issue.get("labels") or []
        label_names = {label.get("name") for label in labels if isinstance(label, dict)}
        if RESULTS_READY_LABEL in label_names:
            continue
        body = issue.get("body") or ""
        m = VM_MARKER_RE.search(body)
        if not m or m.group(1) != VM_ID:
            continue
        number = issue["number"]
        try:
            clear_vm_marker(number=number, vm_id=VM_ID)
            unassign_issue(number=number, assignee=ASSIGNEE)
        except urllib.error.HTTPError as e:
            logger.warning(f"#{number}: failed to reclaim: {e}")
            continue
        reclaimed += 1
        logger.info(f"#{number}: reclaimed orphaned issue (vm-id {VM_ID}).")


def issue_has_failed_label(issue: dict) -> bool:
    """Return True if the issue has the ``evaluation-failed`` label.

    Args:
        issue:
            The issue dict from the GitHub API.

    Returns:
        True if the failed label is present, False otherwise.
    """
    return any(label.get("name") == FAILED_LABEL for label in issue.get("labels", []))


def issue_has_gated_label(issue: dict) -> bool:
    """Return True if the issue has the ``Gated`` label.

    Args:
        issue:
            The issue dict from the GitHub API.

    Returns:
        True if the gated label is present, False otherwise.
    """
    return any(label.get("name") == GATED_LABEL for label in issue.get("labels", []))


def list_open_unassigned_issues() -> list[dict]:
    """Return open model-evaluation-request issues with no assignee.

    Returns:
        The list of open unassigned issues, with pull requests filtered out.
    """
    issues = gh_request(
        path=f"/repos/{REPO}/issues",
        params={
            "state": "open",
            "labels": MODEL_REQUEST_LABEL,
            "per_page": "100",
            "assignee": "none",
        },
    )
    assert isinstance(issues, list)
    return [i for i in issues if "pull_request" not in i]


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
    # access when the label says gated but HF now says otherwise.
    live_summary = cached_model_summary(model_id=model_id)
    is_gated = live_summary is not None and live_summary.get("gated")
    has_gated_label = issue_has_gated_label(issue=issue)
    if is_gated:
        if not has_gated_label:
            add_gated_label(number=number)
            logger.info(f"#{number}: marked gated -- {ASSIGNEE} lacks read access.")
        else:
            logger.info(f"#{number}: still gated -- leaving label in place.")
        return
    if has_gated_label:
        remove_gated_label(number=number)
        logger.info(f"#{number}: access granted, removed gated label.")

    logger.info(f"#{number}: claiming issue for {model_id!r}, languages={languages}")
    # Set the VM marker BEFORE assigning so a crash between the two leaves
    # the issue unassigned (harmless) rather than assigned-but-unowned.
    if not set_vm_marker(number=number, vm_id=VM_ID):
        logger.info(f"#{number}: another VM already owns this issue; aborting.")
        return
    assign_issue(number=number, assignee=ASSIGNEE)
    # Two VMs sharing a PAT cannot be told apart by the assignee, so another
    # VM that raced through the same set_vm_marker + assign_issue window will
    # have overwritten our marker. Verify ownership before proceeding so the
    # losing VM doesn't both duplicate work and later strip the assignment
    # out from under the winning VM's still-running evaluation.
    if not vm_marker_matches(number=number, vm_id=VM_ID):
        logger.info(
            f"#{number}: another VM won the claim race; aborting without "
            "touching the assignee."
        )
        return
    global _current_issue_number
    _current_issue_number = number
    try:
        _run_claimed_issue(issue=issue, model_id=model_id, languages=languages)
    except BaseException:
        release_current_issue()
        raise
    _current_issue_number = None


def upload_results_to_hf_bucket(lines: list[str], model_id: str) -> bool:
    """Upload result lines to the HF results bucket.

    Appends new lines to the model-specific file and uploads only that file.
    Never deletes existing files or lines from the bucket (additive only).

    Args:
        lines:
            The JSONL result lines to upload.
        model_id:
            The HuggingFace model ID.

    Returns:
        True if upload succeeded, False otherwise.
    """
    # Ensure cache dir exists
    RESULTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = _model_id_to_filename(model_id)
    model_file = RESULTS_CACHE_DIR / filename

    # Load existing lines for dedup within this batch
    existing_lines: set[str] = set()
    if model_file.exists():
        existing_lines = {
            line
            for line in model_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    # Append new unique lines to cache file
    new_lines = [line for line in lines if line and line not in existing_lines]
    if new_lines:
        with model_file.open("a", encoding="utf-8") as f:
            for line in new_lines:
                f.write(line + "\n")

    # Upload only the model-specific file (additive only, never deletes)
    try:
        logger.info(f"Uploading results to {HF_RESULTS_BUCKET}...")
        api = HfApi()
        api.sync_bucket(
            source=str(RESULTS_CACHE_DIR), dest=f"hf://buckets/{HF_RESULTS_BUCKET}/"
        )
        logger.info(
            f"Uploaded {len(new_lines)} new result lines for {model_id!r} to HF bucket."
        )
        return True
    except HfHubHTTPError as e:
        logger.error(f"Failed to upload to HF bucket: {e}")
        return False


def _run_claimed_issue(issue: dict, model_id: str, languages: list[str]) -> None:
    """Run euroeval for all languages, uploading to HF bucket after each.

    Results are uploaded incrementally to the Hugging Face results bucket
    after each language completes, enabling crash recovery with minimal loss.

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

    # Load existing results from cache for resume support
    existing_lines = result_lines_for_model(
        lines=read_jsonl_lines(path=RESULTS_PATH), model_id=model_id
    )
    accumulated = list(existing_lines)
    done = completed_languages(lines=accumulated, requested_languages=languages)
    pending = [lang for lang in languages if lang not in done]
    failed: list[str] = []

    gated_detected = False
    failure_reason: str | None = None
    failure_output_tail = ""
    last_output = ""
    total_skipped = 0

    # Run languages one at a time and upload after each completes.
    # This restores crash resilience lost when removing gist-based uploads.
    for i, lang in enumerate(pending):
        logger.info(
            f"#{number}: running {model_id!r} on {lang} ({i + 1}/{len(pending)})."
        )
        before = set(read_jsonl_lines(path=RESULTS_PATH))
        returncode, output = run_euroeval(
            model_id=model_id,
            languages=[lang],
            evaluate_test_split=is_core,
            clear_model_cache=True,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        )
        last_output = output

        # Check for gating / errors BEFORE uploading results.
        # This prevents partial/broken results from being synced to the bucket.
        gated_in_lang = GATED_OUTPUT_RE.search(output)
        num_errored = num_errored_benchmarks(output=output)
        num_skipped_lang = num_skipped_benchmarks(output=output)
        total_skipped += num_skipped_lang
        has_error = returncode != 0 or num_errored > 0

        if not gated_in_lang and not has_error:
            # Only upload if the language run succeeded.
            after = read_jsonl_lines(path=RESULTS_PATH)
            new_lines = [line for line in after if line not in before]
            accumulated.extend(new_lines)

            if new_lines:
                upload_ok = upload_results_to_hf_bucket(
                    lines=new_lines, model_id=model_id
                )
                if not upload_ok:
                    logger.error(
                        f"#{number}: bucket upload failed after {lang}; "
                        "continuing with remaining languages."
                    )
                    failed.append(f"{lang} (upload-failed)")
                    continue
                done.append(lang)
                logger.info(f"#{number}: uploaded results for {lang}.")

        # Handle gating.
        if gated_in_lang:
            gated_detected = True
            failure_output_tail = output[-6000:].strip() or "(no output captured)"
            failed.append(lang)
            break

        # Handle errors.
        if returncode != 0:
            failure_reason = f"euroeval exited with code {returncode}"
            failure_output_tail = output[-6000:].strip() or "(no output captured)"
            failed.append(lang)
            break
        elif num_errored > 0:
            failure_reason = f"euroeval reported {num_errored} errored benchmark(s)"
            failure_output_tail = output[-6000:].strip() or "(no output captured)"
            failed.append(lang)
            break

    # Handle skips for missing official pairs (only if no hard failures).
    if not failed and not pending:
        # Every requested language was already fully covered by results
        # downloaded from the bucket, so no evaluation needed to run. This is a
        # completed evaluation, not an error.
        logger.info(
            f"#{number}: all requested language(s) already present in the bucket "
            f"for {model_id!r}; nothing to evaluate."
        )
    elif not failed:
        missing = missing_official_dataset_language_pairs(
            lines=accumulated, requested_languages=pending
        )
        # Pairs missing beyond what euroeval intentionally skipped (e.g. a
        # dataset whose type the model can't run) are genuine failures. A run
        # that produces no new lines is not itself a failure -- on a resume
        # where the bucket already held the results, euroeval re-reports the
        # same skips and ``total_skipped`` accounts for the missing pairs.
        if len(missing) > total_skipped:
            failure_reason = (
                f"missing official dataset-language pair(s): "
                f"{format_dataset_language_pairs(dataset_language_pairs=missing)}"
            )
            failure_output_tail = last_output[-6000:].strip() or "(no output captured)"
            failed.extend([lang for lang in pending if lang not in done])
        elif missing:
            logger.info(
                f"#{number}: euroeval skipped {total_skipped} benchmark(s); "
                f"treating missing pair(s) as intentional skips: "
                f"{format_dataset_language_pairs(dataset_language_pairs=missing)}"
            )
            done.extend([lang for lang in pending if lang not in done])
        else:
            done.extend([lang for lang in pending if lang not in done])

    # Log completion status.
    if failed:
        logger.info(
            f"#{number}: evaluation failed for {model_id!r} "
            f"after {len(done)} completed language(s)."
        )
    else:
        logger.info(
            f"#{number}: completed all {len(done)} language(s) for {model_id!r}."
        )

    if gated_detected:
        add_gated_label(number=number)
        add_failed_label(number=number)
        release_issue_if_owned(number=number, vm_id=VM_ID, assignee=ASSIGNEE)
        logger.info(
            f"#{number}: euroeval reported a gated repo for {model_id!r}; "
            f"added Gated and evaluation-failed labels to avoid retry loops."
        )
        return

    if failed:
        version = __version__
        reason = failure_reason or f"failed languages: {', '.join(failed)}"
        tail = failure_output_tail or "(no output captured)"
        if issue_has_matching_error_comment(number=number, reason=reason):
            release_issue_if_owned(number=number, vm_id=VM_ID, assignee=ASSIGNEE)
            logger.info(
                f"#{number}: identical error already posted; returned to queue."
            )
            return
        error_comment = (
            f"Error encountered during evaluation ({reason}):\n\n"
            f"```bash\n{tail}\n```\n\n"
            f"EuroEval version: v{version}\n"
        )
        comment_on_issue(number=number, body=error_comment)
        add_failed_label(number=number)
        release_issue_if_owned(number=number, vm_id=VM_ID, assignee=ASSIGNEE)
        logger.info(
            f"#{number}: marked errored on v{version} after {len(failed)} failed "
            f"language(s) ({', '.join(failed)}); returned to queue."
        )
        return

    remove_failed_label(number=number)
    add_results_ready_label(number=number)
    clear_vm_marker(number=number, vm_id=VM_ID)


def issue_is_still_claimable(number: int) -> bool:
    """Return True if the issue is still open with no assignees.

    Re-fetches the issue at claim time so that issues which were closed
    or assigned between the initial snapshot and now are not
    double-processed.

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


@cache
def load_core_model_ids() -> frozenset[str]:
    """Return the set of model ids listed as core models.

    Core models are evaluated on the test split; every other model is
    run on the validation split.

    Returns:
        The set of core model ids defined in ``core_models.yaml``.
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


if __name__ == "__main__":
    main()
