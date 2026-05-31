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
HUGGINGFACE_API_KEY   A Hugging Face token with read access to any gated
                      repos that are expected to be evaluated. Used both
                      for Hub metadata lookups and for downloads inside
                      the ``euroeval`` subprocess.
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
import logging
import os
import sys
import time
import urllib.error
from functools import cache
from pathlib import Path

from huggingface_hub import HfApi
from yaml import safe_load

from leaderboards.evaluation_common import (
    GPU_FIT_OVERHEAD,
    LANGUAGE_GROUP_CODES,
    estimated_model_bytes,
    extract_language_groups,
    gpu_total_memory_bytes,
    missing_official_dataset_language_pairs,
    run_euroeval,
)
from leaderboards.github_api import (
    LABEL,
    REPO,
    RESULTS_READY_LABEL,
    add_failed_label,
    add_results_ready_label,
    assign_issue,
    comment_on_issue,
    gh_request,
    remove_failed_label,
    unassign_issue,
)
from leaderboards.paths import CORE_MODELS_CONFIG
from leaderboards.queue_env import (
    acquire_single_instance_lock,
    load_dotenv_into_environ,
    prompt_and_persist_env_var,
    resolve_assignee_from_token,
    resolve_vm_id,
)
from leaderboards.queue_hf_cache import cached_model_summary
from leaderboards.queue_markers import (
    VM_MARKER_RE,
    clear_gated_marker,
    clear_vm_marker,
    errored_marker_present,
    gated_marker_present,
    release_issue_if_owned,
    set_errored_marker,
    set_gated_marker,
    set_gated_with_errored_block,
    set_vm_marker,
    vm_marker_matches,
)
from leaderboards.queue_parsing import (
    GATED_OUTPUT_RE,
    completed_languages,
    euroeval_version,
    extract_model_id,
    format_dataset_language_pairs,
    model_has_partial_results,
    num_errored_benchmarks,
    num_skipped_benchmarks,
    read_jsonl_lines,
    result_lines_for_model,
)
from leaderboards.queue_progress import (
    ProgressState,
    find_partial_results_for_issue,
    find_progress_comment,
    post_or_update_progress_comment,
    upload_results_gist,
)
from leaderboards.queue_runtime import (
    ThermalConfig,
    cool_down_between_issues,
    lower_process_priority,
    wait_for_gpu_to_cool,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("process_evaluation_queue")

ASSIGNEE = ""
VM_ID = os.environ.get("EUROEVAL_VM_ID", "")
VM_ID_ENV_PATH = Path(os.environ.get("EUROEVAL_DOTENV_PATH", ".env"))
RESULTS_PATH = Path("euroeval_benchmark_results.jsonl")
LOCK_PATH = Path(os.environ.get("EUROEVAL_QUEUE_LOCK", "/tmp/euroeval_queue.lock"))

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


def main() -> None:
    """Process the queue forever, sleeping one hour between passes.

    Credentials and the single-instance lock are acquired once for the
    lifetime of the process, then :func:`process_queue_once` runs in a loop.
    """
    parse_args()
    lower_process_priority()
    ensure_credentials()

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
        help="GPU temperature (°C) at or above which to pause before the next issue.",
    )
    parser.add_argument(
        "--thermal-resume-temp",
        type=float,
        default=THERMAL_CONFIG.resume_temp_c,
        help="GPU temperature (°C) the GPU must cool to before resuming.",
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
    if not os.environ.get("HUGGINGFACE_API_KEY"):
        token = prompt_and_persist_env_var(
            env_path=VM_ID_ENV_PATH,
            name="HUGGINGFACE_API_KEY",
            prompt_text=(
                "HUGGINGFACE_API_KEY is required (a Hugging Face token with read "
                "access to gated repos you intend to evaluate). Enter token"
            ),
            secret=True,
        )
        os.environ["HUGGINGFACE_API_KEY"] = token
    global ASSIGNEE
    ASSIGNEE = resolve_assignee_from_token()
    global VM_ID
    if not VM_ID:
        VM_ID = resolve_vm_id(env_path=VM_ID_ENV_PATH)
    logger.info(f"Using vm-id {VM_ID!r} (assignee {ASSIGNEE!r}).")
    try:
        HfApi().whoami()
    except Exception as e:  # noqa: BLE001
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
        logger.warning(f"#{number}: could not release on interrupt: {e}")


def process_queue_once() -> None:
    """Process every unassigned model-evaluation-request issue once.

    Issues are sorted by (status priority asc, parameter count asc,
    num-language-groups asc). Status priority is 0 for fresh issues, 1
    for gated repos (cheap marker refresh), and 2 for retries of
    previously errored evaluations, so that quicker work is picked up
    first and gated items are surfaced ahead of errored ones.
    """
    try:
        issues = list_open_unassigned_issues()
    except urllib.error.HTTPError as e:
        logger.error(f"Failed to list issues: {e}")
        return

    existing_lines = read_jsonl_lines(path=RESULTS_PATH)
    candidates: list[tuple[int, int, int, int, dict, str, list[str], dict | None]] = []
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
        # A previous VM may have crashed mid-run on this now-unassigned
        # issue; if its progress comment links to a still-reachable gist,
        # that gist holds the canonical partial results even when our
        # local results file is empty.
        partial_state = find_partial_results_for_issue(number=number)
        combined_lines = existing_lines + (
            partial_state["lines"] if partial_state else []
        )
        if status_priority == 0 and model_has_partial_results(
            lines=combined_lines, model_id=model_id, requested_languages=languages
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
                partial_state,
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
        partial_state,
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
            process_issue(
                issue=issue,
                model_id=model_id,
                groups=groups,
                partial_state=partial_state,
            )
        except Exception as e:  # noqa: BLE001
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


def process_issue(
    issue: dict, model_id: str, groups: list[str], partial_state: dict | None = None
) -> None:
    """Claim, evaluate, and report back on a single queue issue.

    Args:
        issue:
            The GitHub issue object returned by the API.
        model_id:
            The Hugging Face model id to evaluate.
        groups:
            The selected language-group labels for this issue.
        partial_state (optional):
            Optional pre-fetched partial-results state from a prior run
            on this issue (``comment_id``, ``gist_id``, ``lines``). When
            set, the existing gist is reused and its lines seed the
            accumulated results so an evaluation orphaned by another VM
            can be continued. Defaults to None.
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
    set_vm_marker(number=number, vm_id=VM_ID)
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
        _run_claimed_issue(
            issue=issue,
            model_id=model_id,
            languages=languages,
            partial_state=partial_state,
        )
    except BaseException:
        release_current_issue()
        raise
    _current_issue_number = None


def _run_claimed_issue(
    issue: dict, model_id: str, languages: list[str], partial_state: dict | None = None
) -> None:
    """Run euroeval one language at a time and maintain a progress comment.

    A single comment carrying the ``queue-progress`` marker is created
    (or reused) and rewritten after every language so the issue always
    shows the completed/in-progress/remaining/failed split along with a
    link to a gist that holds the accumulated JSONL results.

    Args:
        issue:
            The GitHub issue object returned by the API.
        model_id:
            The Hugging Face model id to evaluate.
        languages:
            The flattened list of language codes for this evaluation.
        partial_state (optional):
            Optional pre-fetched partial-results state from a prior run
            on this issue. When set, the existing gist is reused (so the
            progress comment keeps pointing at a single, growing gist)
            and its lines seed the accumulated results. Defaults to None.
    """
    number = issue["number"]
    is_core = model_id in load_core_model_ids()
    issue_body = issue.get("body")

    progress = ProgressState(issue_number=number)

    existing_lines = result_lines_for_model(
        lines=read_jsonl_lines(path=RESULTS_PATH), model_id=model_id
    )
    if partial_state:
        progress.gist_id = partial_state["gist_id"]
        gist_lines = result_lines_for_model(
            lines=partial_state["lines"], model_id=model_id
        )
        seen = set(existing_lines)
        for line in gist_lines:
            if line not in seen:
                existing_lines.append(line)
                seen.add(line)
    accumulated = list(existing_lines)
    done = completed_languages(lines=accumulated, requested_languages=languages)
    pending = [lang for lang in languages if lang not in done]
    failed: list[str] = []

    comment_id = (
        partial_state["comment_id"]
        if partial_state
        else find_progress_comment(number=number)
    )
    comment_id = post_or_update_progress_comment(
        state=progress,
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

    for i, lang in enumerate(pending):
        remaining_after = pending[i + 1 :]
        comment_id = post_or_update_progress_comment(
            state=progress,
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
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        )
        after = read_jsonl_lines(path=RESULTS_PATH)
        new_lines = [line for line in after if line not in before]
        accumulated.extend(new_lines)

        if GATED_OUTPUT_RE.search(output):
            gated_detected = True
            failure_output_tail = output[-6000:].strip() or "(no output captured)"
            break

        num_errored = num_errored_benchmarks(output=output)
        num_skipped = num_skipped_benchmarks(output=output)
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
        elif missing and (not new_lines or len(missing) > num_skipped):
            reason = (
                f"missing official dataset-language pair(s) for {lang!r}: "
                f"{format_dataset_language_pairs(dataset_language_pairs=missing)}"
            )
            failure_reason = failure_reason or reason
            failure_output_tail = output[-6000:].strip() or "(no output captured)"
            failed.append(lang)
        elif missing:
            logger.info(
                f"#{number}: euroeval skipped {num_skipped} benchmark(s) for "
                f"{lang!r}; treating missing pair(s) as intentional skips: "
                f"{format_dataset_language_pairs(dataset_language_pairs=missing)}"
            )
            done.append(lang)
        else:
            done.append(lang)

        comment_id = post_or_update_progress_comment(
            state=progress,
            comment_id=comment_id,
            model_id=model_id,
            done=done,
            current=None,
            remaining=remaining_after,
            failed=failed,
            lines=accumulated,
            issue_body=issue_body,
        )

        if not is_last:
            wait_for_gpu_to_cool(config=THERMAL_CONFIG)

    # Upload the fully accumulated results to the gist one final time.
    # This ensures the gist contains all language results after evaluation
    # completes (the per-language calls above may have been interrupted or
    # the gist may only contain partial results).
    try:
        upload_results_gist(
            state=progress, model_id=model_id, lines=accumulated, issue_body=issue_body
        )
    except Exception:
        logger.warning(
            f"#{number}: failed to upload final results gist for {model_id!r}"
        )

    if gated_detected:
        version = euroeval_version()
        set_gated_with_errored_block(
            number=number, body=issue.get("body"), version=version
        )
        add_failed_label(number=number)
        release_issue_if_owned(number=number, vm_id=VM_ID, assignee=ASSIGNEE)
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
            release_issue_if_owned(number=number, vm_id=VM_ID, assignee=ASSIGNEE)
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
        comment_on_issue(number=number, body=error_comment)
        set_errored_marker(number=number, body=issue.get("body"), version=version)
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
    logger.info(
        f"#{number}: completed all {len(done)} language(s) for {model_id!r}; "
        "progress comment updated."
    )


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
