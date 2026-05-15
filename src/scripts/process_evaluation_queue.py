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
from typing import Any

from huggingface_hub import HfApi

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s ⋅ %(levelname)s ⋅ %(message)s"
)
logger = logging.getLogger("process_evaluation_queue")

REPO = "EuroEval/EuroEval"
LABEL = "model evaluation request"
TITLE_PREFIX = "[MODEL EVALUATION REQUEST]"
ASSIGNEE = "saattrupdan"
RESULTS_PATH = os.environ.get(
    "EUROEVAL_RESULTS_PATH", "euroeval_benchmark_results.jsonl"
)

# Mapping from issue-checkbox label -> list of EuroEval --language codes.
_BALTIC = "Baltic languages (Latvian, Lithuanian)"
_FINNIC = "Finnic languages (Estonian, Finnish)"
_ROMANCE = "Romance languages (Catalan, French, Italian, Portuguese, Romanian, Spanish)"
_SCANDI = "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)"
_SLAVIC = (
    "Slavic languages (Belarusian, Bulgarian, Bosnian, Croatian, Czech, Polish,"
    " Serbian, Slovak, Slovenian, Ukrainian)"
)
_WGERMANIC = "West Germanic languages (Dutch, English, German)"

# Marker the script appends to an issue body when an evaluation fails. The
# frontend parses this to display "Error" / "Waiting for bug fix" statuses.
ERROR_MARKER_RE = re.compile(r"<!--\s*errored-on:\s*v([^\s>-]+)\s*-->")


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


def _token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("GITHUB_TOKEN env var is required.")
        sys.exit(1)
    return token


def gh_request(
    path: str,
    *,
    method: str = "GET",
    body: dict | None = None,
    params: dict | None = None,
) -> Any:  # noqa: ANN401
    """Call the GitHub REST API and return the parsed JSON body.

    Returns:
        The parsed JSON response, or ``None`` if the body was empty.
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


def list_open_unassigned_issues() -> list[dict]:
    """Return open model-evaluation-request issues with no assignee."""
    issues = gh_request(
        f"/repos/{REPO}/issues",
        params={
            "state": "open",
            "labels": LABEL,
            "per_page": "100",
            "assignee": "none",
        },
    )
    # Filter out pull requests, just in case.
    return [i for i in issues if "pull_request" not in i]


def extract_model_id(title: str) -> str | None:
    """Return the model id from an issue title, or ``None`` if not parseable."""
    prefix = f"{TITLE_PREFIX} "
    if not title.startswith(prefix):
        return None
    rest = title[len(prefix) :].strip()
    return rest if rest and rest != "<model-name>" else None


def extract_language_groups(body: str | None) -> list[str]:
    """Return the language-group labels that are ticked in an issue body."""
    if not body:
        return []
    selected: list[str] = []
    for group in LANGUAGE_GROUP_CODES:
        pattern = re.compile(rf"-\s*\[[xX]\]\s*{re.escape(group)}")
        if pattern.search(body):
            selected.append(group)
    return selected


def huggingface_model_info(model_id: str) -> object | None:
    """Return HF Hub metadata for ``model_id``, or ``None`` if unavailable."""
    try:
        return HfApi().model_info(repo_id=model_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("HF model lookup failed for %s: %s", model_id, e)
        return None


def huggingface_param_count(info: object) -> int:
    """Return the model's parameter count, or a large fallback if unknown.

    Falls back to ``sys.maxsize`` so that models with unknown size are
    deprioritised by the queue sorter.
    """
    safetensors = getattr(info, "safetensors", None)
    total = getattr(safetensors, "total", None) if safetensors else None
    if isinstance(total, int) and total > 0:
        return total
    return sys.maxsize


def assign_issue(number: int) -> None:
    """Assign the issue to the configured ``ASSIGNEE``."""
    gh_request(
        f"/repos/{REPO}/issues/{number}/assignees",
        method="POST",
        body={"assignees": [ASSIGNEE]},
    )


def unassign_issue(number: int) -> None:
    """Remove ``ASSIGNEE`` so the issue returns to the unassigned pool."""
    gh_request(
        f"/repos/{REPO}/issues/{number}/assignees",
        method="DELETE",
        body={"assignees": [ASSIGNEE]},
    )


def euroeval_version() -> str:
    """Return the locally installed EuroEval package version."""
    try:
        return importlib.metadata.version("euroeval")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def version_tuple(v: str) -> tuple[int, ...]:
    """Return a comparable tuple for a dotted-numeric version like ``17.2.0``.

    Non-numeric or unknown components yield ``(-1,)`` so they sort before
    anything else and never block retries.
    """
    if v == "unknown":
        return (-1,)
    try:
        return tuple(int(p) for p in v.split(".") if p.isdigit())
    except ValueError:
        return (-1,)


def errored_on_version(body: str | None) -> str | None:
    """Return the EuroEval version recorded in the issue body, if any."""
    if not body:
        return None
    m = ERROR_MARKER_RE.search(body)
    return m.group(1) if m else None


def set_errored_marker(number: int, body: str | None, version: str) -> None:
    """Append/replace the ``errored-on`` marker in the issue body."""
    cleaned = ERROR_MARKER_RE.sub("", body or "").rstrip()
    new_body = f"{cleaned}\n\n<!-- errored-on: v{version} -->\n"
    gh_request(
        f"/repos/{REPO}/issues/{number}", method="PATCH", body={"body": new_body}
    )


def comment_on_issue(number: int, comment: str) -> None:
    """Post ``comment`` to the issue with the given number."""
    gh_request(
        f"/repos/{REPO}/issues/{number}/comments", method="POST", body={"body": comment}
    )


def read_jsonl_lines(path: str) -> list[str]:
    """Return the non-empty lines of a JSONL file, or an empty list if absent."""
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def run_euroeval(model_id: str, languages: list[str]) -> tuple[bool, str]:
    """Run the euroeval CLI for the given model and language list.

    Returns:
        ``(success, captured_stderr)``. ``captured_stderr`` is populated
        with the tail of the subprocess's combined output when the run
        fails, so it can be posted back to the issue.
    """
    cmd = ["euroeval", "--model", model_id]
    for lang in languages:
        cmd += ["--language", lang]
    logger.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        # Re-run with capture so we have a stack trace to report. The first
        # run already crashed, so this is at worst a second crash on the
        # same input — far cheaper than the eval itself.
        captured = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, check=False
        )
        output = (captured.stderr or "") + (captured.stdout or "")
        logger.error("euroeval failed (exit %s) for %s.", e.returncode, model_id)
        # Keep only the last ~6 KiB so the comment doesn't blow up.
        return False, output[-6000:].strip()
    except FileNotFoundError:
        logger.error("`euroeval` CLI not found on PATH. Is it installed?")
        return False, "`euroeval` CLI not found on PATH."


def process_issue(issue: dict, model_id: str, groups: list[str]) -> None:
    """Claim, evaluate, and report back on a single queue issue."""
    number = issue["number"]
    languages: list[str] = []
    for g in groups:
        languages.extend(LANGUAGE_GROUP_CODES[g])
    languages = sorted(set(languages))

    logger.info("#%s: claiming issue for %r, languages=%s", number, model_id, languages)
    assign_issue(number)

    before = set(read_jsonl_lines(RESULTS_PATH))
    ok, error_output = run_euroeval(model_id, languages)
    after = read_jsonl_lines(RESULTS_PATH)
    new_lines = [line for line in after if line not in before]

    if not ok:
        version = euroeval_version()
        comment = (
            f"Error encountered during evaluation:\n\n"
            f"```bash\n{error_output or '(no output captured)'}\n```\n\n"
            f"EuroEval version: v{version}\n"
        )
        comment_on_issue(number, comment)
        set_errored_marker(number, issue.get("body"), version)
        unassign_issue(number)
        logger.info("#%s: marked errored on v%s, returned to queue.", number, version)
        return

    if not new_lines:
        logger.warning(
            "#%s: no new lines produced in %s — leaving issue assigned for"
            " manual inspection.",
            number,
            RESULTS_PATH,
        )
        return

    payload = "\n".join(new_lines)
    comment = f"Results for `{model_id}`:\n\n```jsonl\n{payload}\n```\n"
    comment_on_issue(number, comment)
    logger.info("#%s: posted %d result line(s) as comment.", number, len(new_lines))


def main() -> None:
    """Process every unassigned model-evaluation-request issue once.

    Issues are sorted by (parameter count asc, num-language-groups asc) so
    that quicker evaluations are picked up first.
    """
    try:
        issues = list_open_unassigned_issues()
    except urllib.error.HTTPError as e:
        logger.error("Failed to list issues: %s", e)
        sys.exit(1)

    current_version = euroeval_version()
    current_v = version_tuple(current_version)

    # Resolve metadata once per issue, then sort by that metadata before
    # doing any heavy work. Tuple is (is_errored, param count, num groups,
    # issue, model id, groups) so errored issues sort after fresh ones.
    candidates: list[tuple[int, int, int, dict, str, list[str]]] = []
    for issue in issues:
        number = issue["number"]
        title = issue.get("title", "")
        body = issue.get("body") or ""

        model_id = extract_model_id(title)
        if not model_id:
            logger.info("#%s: skipping — could not parse model id from title.", number)
            continue

        groups = extract_language_groups(body)
        if not groups:
            logger.info("#%s: skipping — no language groups selected.", number)
            continue

        errored_v = errored_on_version(body)
        if errored_v is not None and version_tuple(errored_v) >= current_v:
            logger.info(
                "#%s: skipping — errored on v%s and current version is v%s.",
                number,
                errored_v,
                current_version,
            )
            continue

        info = huggingface_model_info(model_id)
        if info is None:
            logger.info("#%s: skipping — model %r not on HF Hub.", number, model_id)
            continue

        param_count = huggingface_param_count(info)
        is_errored = 1 if errored_v is not None else 0
        candidates.append(
            (is_errored, param_count, len(groups), issue, model_id, groups)
        )

    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    logger.info("Found %d processable issue(s).", len(candidates))

    for is_errored, param_count, num_groups, issue, model_id, groups in candidates:
        logger.info(
            "#%s: queueing %r (%s params, %d group(s), %s).",
            issue["number"],
            model_id,
            param_count,
            num_groups,
            "retry of errored eval" if is_errored else "fresh",
        )
        try:
            process_issue(issue, model_id, groups)
        except Exception as e:  # noqa: BLE001
            logger.exception("Error while processing issue #%s: %s", issue["number"], e)
        time.sleep(1)


if __name__ == "__main__":
    main()
