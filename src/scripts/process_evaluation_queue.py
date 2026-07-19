"""Pick up open model-evaluation-request issues and run EuroEval on them.

The queue processor claims unassigned request issues, runs EuroEval for the
requested language groups, uploads successful results to the Hugging Face
bucket, and marks finished issues as results-ready.

Required env vars
-----------------
GITHUB_TOKEN          A PAT with ``issues: write`` for the EuroEval repo.
HF_TOKEN              A Hugging Face token with read access to gated repos.
"""

import datetime as dt
import logging
import os
import sys
import time
import urllib.error
from pathlib import Path

import click
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from euroeval import __version__
from leaderboards.bucket_sync import merge_results, sync_bucket
from leaderboards.constants import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
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
    summarise_evaluation_error,
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


# Canonical HF bucket for storing results (public read access).
HF_RESULTS_BUCKET = "EuroEval/results"


# Param bucket thresholds matching leaderboards (src/leaderboards/core_models.py)
_BUCKET_THRESHOLDS = [
    (2_000_000_000, 0),  # tiny
    (10_000_000_000, 1),  # small
    (40_000_000_000, 2),  # medium
    (80_000_000_000, 3),  # large
    (float("inf"), 4),  # xlarge
]


@click.command()
@click.option(
    "--vm-id",
    required=True,
    help="Identifier for this VM/host while it is evaluating an issue.",
)
@click.option(
    "--lock-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default="/tmp/euroeval_queue.lock",
    show_default=True,
    help="Single-instance lock file for this queue processor.",
)
@click.option(
    "--gpu-memory-utilization",
    type=click.FloatRange(min=0.0, max=1.0),
    default=None,
    help=(
        "vLLM GPU memory utilization fraction. When omitted, the euroeval "
        "CLI's own default is used."
    ),
)
@click.option(
    "--inter-issue-sleep",
    type=float,
    default=30.0,
    show_default=True,
    help="Seconds to wait between issues regardless of thermal state.",
)
@click.option(
    "--thermal-pause-temp",
    type=float,
    default=80.0,
    show_default=True,
    help=("GPU temperature in deg C at or above which to pause before the next issue."),
)
@click.option(
    "--thermal-resume-temp",
    type=float,
    default=70.0,
    show_default=True,
    help="GPU temperature in deg C the GPU must cool to before resuming.",
)
def main(
    vm_id: str,
    lock_path: Path,
    gpu_memory_utilization: float | None,
    inter_issue_sleep: float,
    thermal_pause_temp: float,
    thermal_resume_temp: float,
) -> None:
    """Process the queue forever, sleeping one hour between passes."""
    sleep_seconds = 60 * 60
    thermal_config = ThermalConfig(
        inter_issue_sleep_seconds=inter_issue_sleep,
        pause_temp_c=thermal_pause_temp,
        resume_temp_c=thermal_resume_temp,
    )

    lower_process_priority()
    load_dotenv_into_environ(env_path=Path(".env"))
    if not os.environ.get("GITHUB_TOKEN"):
        os.environ["GITHUB_TOKEN"] = prompt_and_persist_env_var(
            env_path=Path(".env"),
            name="GITHUB_TOKEN",
            prompt_text=(
                f"GITHUB_TOKEN is required (a PAT with `issues: write` for {REPO}). "
                "Enter token"
            ),
            secret=True,
        )

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if not hf_token:
        os.environ["HF_TOKEN"] = prompt_and_persist_env_var(
            env_path=Path(".env"),
            name="HF_TOKEN",
            prompt_text=(
                "HF_TOKEN is required (a Hugging Face token with read access to gated "
                "repos you intend to evaluate). Enter token"
            ),
            secret=True,
        )
    elif not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = hf_token

    assignee = resolve_assignee_from_token()
    logger.info(f"Using vm-id {vm_id!r} (assignee {assignee!r}).")
    try:
        HfApi().whoami()
    except Exception as e:  # noqa: BLE001
        logger.error(
            "Not logged in to Hugging Face. Run `huggingface-cli login` "
            f"(or set HF_TOKEN) and re-run. Underlying error: {e}"
        )
        sys.exit(1)

    lock_fd = acquire_single_instance_lock(lock_path=lock_path)
    try:
        # The flock guarantees no other queue processor on this host is mid-run,
        # so any issue still carrying this VM's marker is a crash-leftover.
        reclaim_orphaned_issues(assignee=assignee, vm_id=vm_id)

        while True:
            try:
                sync_bucket()
                merge_results(results_file=Path("euroeval_benchmark_results.jsonl"))
            except RuntimeError as e:
                logger.warning(f"Could not download results from HF bucket: {e}")

            process_queue_once(
                assignee=assignee,
                vm_id=vm_id,
                gpu_memory_utilization=gpu_memory_utilization,
                thermal_config=thermal_config,
            )
            logger.info(
                f"Queue pass complete; sleeping {sleep_seconds}s before next pass."
            )
            time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        logger.info("Interrupted; exiting.")
        sys.exit(130)
    finally:
        os.close(lock_fd)


def process_queue_once(
    assignee: str,
    vm_id: str,
    gpu_memory_utilization: float | None,
    thermal_config: ThermalConfig,
) -> None:
    """Process every unassigned model-evaluation-request issue once."""
    candidates = _queue_candidates()
    gpu_bytes = gpu_total_memory_bytes()

    # vLLM can only allocate `gpu_memory_utilization * total GPU memory`, so the
    # fit pre-check must budget against that fraction rather than the whole card;
    # otherwise a model whose weights fit but whose KV cache does not still slips
    # through and OOMs at runtime.
    usable_bytes: int | None = None
    if gpu_bytes is None:
        logger.info(
            "Could not determine local memory budget; skipping the fit pre-check."
        )
    else:
        utilization = (
            gpu_memory_utilization
            if gpu_memory_utilization is not None
            else DEFAULT_GPU_MEMORY_UTILIZATION
        )
        usable_bytes = int(gpu_bytes * utilization)
        logger.info(
            f"Local memory budget: {gpu_bytes / (1024**3):.1f} GiB total, "
            f"{usable_bytes / (1024**3):.1f} GiB usable at "
            f"gpu_memory_utilization={utilization}."
        )

    for (
        slow_priority,
        type_priority,
        status_priority,
        param_count,
        _age,
        issue,
        model_id,
        groups,
    ) in candidates:
        status = {0: "gated", 1: "retry of errored eval", 2: "fresh"}[status_priority]
        model_type = "generative" if type_priority == 0 else "encoder"
        slow_tag = ", slow" if slow_priority else ""
        logger.info(
            f"#{issue['number']}: queueing {model_id!r} ({param_count} params, "
            f"{model_type}, {status}{slow_tag})."
        )
        if usable_bytes is not None:
            needed = estimated_model_bytes(model_id=model_id)
            if needed is not None and int(needed * GPU_FIT_OVERHEAD) > usable_bytes:
                logger.info(
                    f"#{issue['number']}: skipping -- model {model_id!r} needs "
                    f"~{needed / (1024**3):.1f} GiB of weights "
                    f"(x {GPU_FIT_OVERHEAD} overhead), which exceeds the usable "
                    f"vLLM budget of {usable_bytes / (1024**3):.1f} GiB. Leaving "
                    "the issue unassigned so a larger or multi-GPU machine can "
                    "pick it up."
                )
                continue
        try:
            process_issue(
                issue=issue,
                model_id=model_id,
                groups=groups,
                assignee=assignee,
                vm_id=vm_id,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        except Exception as e:  # noqa: BLE001
            # Top-level per-issue guard: one failing issue must not abort the
            # whole queue loop, so log it and move on to the next issue.
            logger.exception(f"Error while processing issue #{issue['number']}: {e}")
        cool_down_between_issues(config=thermal_config)


def _queue_candidates() -> list[tuple[int, int, int, int, float, dict, str, list[str]]]:
    """Return processable issues sorted by priority.

    Returns:
        Queue candidates as sortable tuples followed by issue, model id and
        language groups.
    """
    try:
        issues = gh_request(
            path=f"/repos/{REPO}/issues",
            params={
                "state": "open",
                "labels": MODEL_REQUEST_LABEL,
                "per_page": "100",
                "assignee": "none",
            },
        )
    except urllib.error.HTTPError as e:
        logger.error(f"Failed to list issues: {e}")
        return []

    if not isinstance(issues, list):
        logger.error("Failed to list issues: GitHub returned a non-list response.")
        return []

    candidates: list[tuple[int, int, int, int, float, dict, str, list[str]]] = []
    for issue in (issue for issue in issues if "pull_request" not in issue):
        number = issue["number"]
        body = issue.get("body") or ""
        model_id = extract_model_id(title=issue.get("title", ""), body=body)
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

        label_names = {
            label.get("name")
            for label in issue.get("labels", [])
            if isinstance(label, dict)
        }
        if summary.get("gated"):
            status_priority = 0
        elif FAILED_LABEL in label_names:
            status_priority = 1
        else:
            status_priority = 2

        created_at = issue.get("created_at")
        if isinstance(created_at, str):
            try:
                age = dt.datetime.fromisoformat(
                    created_at.replace("Z", "+00:00")
                ).timestamp()
            except ValueError:
                age = float("inf")
        else:
            age = float("inf")

        candidates.append(
            (
                1 if "slow" in label_names else 0,
                0 if summary.get("generative", True) else 1,
                status_priority,
                next(
                    bucket
                    for threshold, bucket in _BUCKET_THRESHOLDS
                    if summary["param_count"] < threshold
                ),
                age,
                issue,
                model_id,
                groups,
            )
        )

    candidates.sort(key=lambda c: (c[0], c[1], c[2], c[3], c[4]))
    logger.info(f"Found {len(candidates)} processable issue(s).")
    return candidates


def reclaim_orphaned_issues(assignee: str, vm_id: str) -> None:
    """Return this VM's orphaned issues to the queue.

    Args:
        assignee:
            GitHub user assigned to issues owned by this runner.
        vm_id:
            VM marker used to distinguish this runner from other VMs.
    """
    try:
        issues = gh_request(
            path=f"/repos/{REPO}/issues",
            params={
                "state": "open",
                "labels": MODEL_REQUEST_LABEL,
                "per_page": "100",
                "assignee": assignee,
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
        if not m or m.group(1) != vm_id:
            continue
        number = issue["number"]
        try:
            clear_vm_marker(number=number, vm_id=vm_id)
            unassign_issue(number=number, assignee=assignee)
        except urllib.error.HTTPError as e:
            logger.warning(f"#{number}: failed to reclaim: {e}")
            continue
        reclaimed += 1
        logger.info(f"#{number}: reclaimed orphaned issue (vm-id {vm_id}).")


def process_issue(
    issue: dict,
    model_id: str,
    groups: list[str],
    assignee: str,
    vm_id: str,
    gpu_memory_utilization: float | None,
) -> None:
    """Claim, evaluate, and report back on a single queue issue.

    Args:
        issue:
            The GitHub issue object returned by the API.
        model_id:
            The Hugging Face model id to evaluate.
        groups:
            The selected language-group labels for this issue.
        assignee:
            The GitHub user to assign while evaluating.
        vm_id:
            The VM marker written to the issue body.
        gpu_memory_utilization:
            Optional vLLM memory fraction.
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
    label_names = {
        label.get("name")
        for label in issue.get("labels", [])
        if isinstance(label, dict)
    }
    has_gated_label = GATED_LABEL in label_names
    if is_gated:
        if not has_gated_label:
            add_gated_label(number=number)
            logger.info(f"#{number}: marked gated -- {assignee} lacks read access.")
        else:
            logger.info(f"#{number}: still gated -- leaving label in place.")
        return
    if has_gated_label:
        remove_gated_label(number=number)
        logger.info(f"#{number}: access granted, removed gated label.")

    logger.info(f"#{number}: claiming issue for {model_id!r}, languages={languages}")

    # Set the VM marker BEFORE assigning so a crash between the two leaves
    # the issue unassigned (harmless) rather than assigned-but-unowned.
    if not set_vm_marker(number=number, vm_id=vm_id):
        logger.info(f"#{number}: another VM already owns this issue; aborting.")
        return
    assign_issue(number=number, assignee=assignee)

    # Two VMs sharing a PAT cannot be told apart by the assignee, so another
    # VM that raced through the same set_vm_marker + assign_issue window will
    # have overwritten our marker. Verify ownership before proceeding so the
    # losing VM doesn't both duplicate work and later strip the assignment
    # out from under the winning VM's still-running evaluation.
    if not vm_marker_matches(number=number, vm_id=vm_id):
        logger.info(
            f"#{number}: another VM won the claim race; aborting without "
            "touching the assignee."
        )
        return
    try:
        _run_claimed_issue(
            issue=issue,
            model_id=model_id,
            languages=languages,
            assignee=assignee,
            vm_id=vm_id,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    except BaseException:
        release_issue_if_owned(number=number, vm_id=vm_id, assignee=assignee)
        raise


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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_file = RESULTS_DIR / f"{model_id.replace('/', '_')}.jsonl"

    existing_lines: set[str] = set()
    if model_file.exists():
        existing_lines = {
            line
            for line in model_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    new_lines = [line for line in lines if line and line not in existing_lines]
    if new_lines:
        with model_file.open("a", encoding="utf-8") as f:
            for line in new_lines:
                f.write(line + "\n")

    try:
        logger.info(f"Uploading results to {HF_RESULTS_BUCKET}...")
        api = HfApi()
        api.sync_bucket(
            source=str(RESULTS_DIR), dest=f"hf://buckets/{HF_RESULTS_BUCKET}/"
        )
        logger.info(
            f"Uploaded {len(new_lines)} new result lines for {model_id!r} to HF bucket."
        )
        return True
    except HfHubHTTPError as e:
        logger.error(f"Failed to upload to HF bucket: {e}")
        return False


def _run_claimed_issue(
    issue: dict,
    model_id: str,
    languages: list[str],
    assignee: str,
    vm_id: str,
    gpu_memory_utilization: float | None,
) -> None:
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
        assignee:
            The GitHub user assigned while evaluating.
        vm_id:
            The VM marker written to the issue body.
        gpu_memory_utilization:
            Optional vLLM memory fraction.
    """
    number = issue["number"]

    results_path = Path("euroeval_benchmark_results.jsonl")
    model_results_path = RESULTS_DIR / f"{model_id.replace('/', '_')}.jsonl"
    existing_lines = read_jsonl_lines(path=model_results_path)
    accumulated = result_lines_for_model(lines=existing_lines, model_id=model_id)
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
        before = set(read_jsonl_lines(path=results_path))
        # The queue always evaluates on the validation split; the test split
        # is reserved for the dedicated core-model run in
        # run_core_model_evaluations.py.
        returncode, output = run_euroeval(
            model_id=model_id,
            languages=[lang],
            evaluate_test_split=False,
            clear_model_cache=True,
            gpu_memory_utilization=gpu_memory_utilization,
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
            after = read_jsonl_lines(path=results_path)
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
            failure_output_tail = summarise_evaluation_error(output=output)
            failed.append(lang)
            break

        # Handle errors.
        if returncode != 0:
            failure_reason = f"euroeval exited with code {returncode}"
            failure_output_tail = summarise_evaluation_error(output=output)
            failed.append(lang)
            break
        elif num_errored > 0:
            failure_reason = f"euroeval reported {num_errored} errored benchmark(s)"
            failure_output_tail = summarise_evaluation_error(output=output)
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
            failure_output_tail = summarise_evaluation_error(output=last_output)
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
        release_issue_if_owned(number=number, vm_id=vm_id, assignee=assignee)
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
            release_issue_if_owned(number=number, vm_id=vm_id, assignee=assignee)
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
        release_issue_if_owned(number=number, vm_id=vm_id, assignee=assignee)
        logger.info(
            f"#{number}: marked errored on v{version} after {len(failed)} failed "
            f"language(s) ({', '.join(failed)}); returned to queue."
        )
        return

    remove_failed_label(number=number)
    add_results_ready_label(number=number)
    clear_vm_marker(number=number, vm_id=vm_id)


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


if __name__ == "__main__":
    main()
