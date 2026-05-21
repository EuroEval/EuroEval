"""Recompute the core-model list and update GitHub issue #1186.

The 'core model' list defines which models we re-run when datasets
change. See `leaderboards/core_models.py` for the methodology. This
script is the user-facing entry point: it reads the already-processed
results, computes the list, updates `core_models.yaml` (including
`last_updated`), and pushes both the new issue body and a diff comment
to GitHub. It does NOT re-process the results archive — run
`make leaderboards` first if new evaluations have landed.

Run manually with `make update-core-models`, or via the staleness prompt
that `generate_leaderboards.py` shows when the list is older than 30
days.

Required env vars
-----------------
GITHUB_TOKEN
    A PAT with ``issues: write`` for the EuroEval repo. Without it the
    script writes the YAML and prints the would-be issue body, but
    skips the GitHub update (so dry runs are cheap).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import click
from dotenv import load_dotenv
from yaml import safe_load

from leaderboards.core_models import CoreModel, build_core_model_list
from leaderboards.paths import CORE_MODELS_CONFIG

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s ⋅ %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("update_core_models")

load_dotenv()


REPO = "EuroEval/EuroEval"

# Section ordering and headers used in the GitHub issue body. The size
# buckets here mirror those in `core_models.py::_size_bucket`.
_BUCKET_HEADER = {
    "encoder": "## Encoder models",
    "tiny": "## Tiny scale (<2B parameters)",
    "small": "## Small scale (2B-10B parameters)",
    "medium": "## Medium scale (10B-40B parameters)",
    "large": "## Large scale (40B-80B parameters)",
    "xlarge": "## Extra-large scale (>80B parameters)",
    "api": "## SOTA API models",
}
_BUCKET_ORDER = ["encoder", "tiny", "small", "medium", "large", "xlarge", "api"]

_ISSUE_INTRO = """## What is a "core model"?

To prevent an ever-growing list of models to evaluate every time we change a \
dataset or add a new language, we maintain this list of _core models_: the \
minimum set of models to re-run when such a change is made.

The list is generated automatically by \
`uv run python src/scripts/update_core_models.py` from three sources:

- ⭐ Pareto frontier within the model's type (encoder, base decoder, \
instruction-tuned decoder, reasoning decoder) — strictly better than every \
smaller-or-equal-sized model of the same type, in at least one language. \
See [src/leaderboards/core_models.yaml](src/leaderboards/core_models.yaml) \
for the exact list of languages per model.
- 🇪🇺 Trained in the EU (matched against the regex list in \
[src/leaderboards/core_models.yaml](src/leaderboards/core_models.yaml)).
- 💜 Top-10 'truly open' models from [osai-index.eu][osai] (filtered to \
text models with open base weights, training code, and data sources).

[osai]: https://osai-index.eu/database/?type=text&weights_basemodel=1&trainingcode=1&datasources_basemodel=1

To add or remove a model, edit \
[src/leaderboards/core_models.yaml](src/leaderboards/core_models.yaml) (the \
EU regex list and OSAI overrides), or open a PR to adjust the source-selection \
logic. Manual comments below are welcome but won't survive the next \
regeneration."""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_line(model: CoreModel) -> str:
    """Render a single model line for the issue body.

    Args:
        model:
            The core model.

    Returns:
        A markdown list line (without trailing newline).
    """
    flags = []
    if model.pareto_languages:
        flags.append("⭐")
    if model.osai_rank is not None:
        flags.append("💜")
    if model.eu:
        flags.append("🇪🇺")
    suffix = "".join(flags)
    line = f"- {model.model_id}"
    if suffix:
        line += f" {suffix}"
    return line


def render_issue_body(models: list[CoreModel]) -> str:
    """Render the full issue body for #1186.

    Args:
        models:
            The core model list, already sorted.

    Returns:
        Markdown text suitable for `PATCH /repos/.../issues/1186`.
    """
    sections = []
    by_bucket: dict[str, list[CoreModel]] = {b: [] for b in _BUCKET_ORDER}
    for model in models:
        by_bucket[model.size_bucket].append(model)
    for bucket in _BUCKET_ORDER:
        bucket_models = by_bucket[bucket]
        if not bucket_models:
            continue
        body = _BUCKET_HEADER[bucket] + "\n\n"
        body += "\n".join(_render_line(m) for m in bucket_models)
        sections.append(body)
    return _ISSUE_INTRO + "\n\n" + "\n\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# YAML I/O — we rewrite the file with a hand-rolled formatter to keep the
# top section human-edited and the bottom section regenerated. PyYAML's
# default dump would reorder keys and strip comments.
# ---------------------------------------------------------------------------


_MODELS_MARKER = "models:"


def _format_models_yaml(models: list[CoreModel]) -> str:
    """Format the models list as a YAML fragment.

    Args:
        models:
            The core model list.

    Returns:
        Text starting with ``models:`` and ending with a newline.
    """
    if not models:
        return "models: []\n"
    lines = ["models:"]
    for m in models:
        lines.append(f"  - id: {m.model_id}")
        lines.append(f"    type: {m.model_type}")
        lines.append(f"    bucket: {m.size_bucket}")
        if m.parameters == m.parameters:  # not NaN
            lines.append(f"    parameters: {int(m.parameters)}")
        else:
            lines.append("    parameters: null")
        if m.pareto_languages:
            langs = ", ".join(m.pareto_languages)
            lines.append(f"    pareto_languages: [{langs}]")
        else:
            lines.append("    pareto_languages: []")
        lines.append(f"    eu: {str(m.eu).lower()}")
        if m.osai_rank is None:
            lines.append("    osai_rank: null")
        else:
            lines.append(f"    osai_rank: {m.osai_rank}")
    return "\n".join(lines) + "\n"


def _write_yaml(
    config_path: Path, last_updated: dt.date, models: list[CoreModel]
) -> None:
    """Update `last_updated` and rewrite the `models:` section in place.

    The hand-edited top section (intro comments, issue number, EU regex
    list, OSAI overrides) is preserved verbatim — we only replace the
    `last_updated:` line and everything from the `models:` marker
    onwards.

    Args:
        config_path:
            Path to `core_models.yaml`.
        last_updated:
            ISO date to record.
        models:
            The new core model list.
    """
    text = config_path.read_text(encoding="utf-8")
    text = re.sub(
        r"^last_updated:.*$",
        f"last_updated: {last_updated.isoformat()}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    idx = text.find("\n" + _MODELS_MARKER)
    if idx == -1:
        # Append if missing.
        text = text.rstrip() + "\n\n" + _format_models_yaml(models)
    else:
        text = text[: idx + 1] + _format_models_yaml(models)
    config_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class IssueDiff:
    """Summary of changes between two core-model lists."""

    added: list[str]
    removed: list[str]
    flag_changes: list[tuple[str, str, str]]  # (model_id, old_flags, new_flags)

    @property
    def is_empty(self) -> bool:
        """Whether the diff has no entries."""
        return not (self.added or self.removed or self.flag_changes)


def _parse_issue_models(body: str) -> dict[str, str]:
    """Parse the existing issue body into {model_id: flag_string}.

    Args:
        body:
            Issue body markdown.

    Returns:
        Map of model_id to the trailing flag/emoji string for that line.
    """
    result: dict[str, str] = {}
    in_model_section = False
    bucket_headers = set(_BUCKET_HEADER.values())
    for line in body.splitlines():
        stripped = line.strip()
        # Bullets under the intro legend (⭐/🇪🇺/💜 explainers) look like
        # list items too — skip everything until we hit a known bucket
        # header. Any other `## …` heading (e.g. the intro) doesn't count.
        if stripped in bucket_headers:
            in_model_section = True
            continue
        if not in_model_section:
            continue
        # The first token after "- " is the model id, which may contain
        # `@revision`, `:tag`, or `#parameter` segments. The flag emojis
        # (and the legacy `_(Pareto: …)_` annotation) are separated from
        # the id by whitespace, so we split on the first space.
        m = re.match(r"^- (\S+)(.*)$", stripped)
        if not m:
            continue
        flags = re.sub(r"_\(Pareto:.*?\)_", "", m.group(2)).strip()
        result[m.group(1)] = flags
    return result


def diff_issue(old_body: str, new_models: list[CoreModel]) -> IssueDiff:
    """Compute the change set between the published issue and the new list.

    Args:
        old_body:
            Current issue body markdown.
        new_models:
            The newly-generated core list.

    Returns:
        An `IssueDiff` with added, removed, and flag-changed entries.
    """
    old = _parse_issue_models(old_body)

    def flags_for(m: CoreModel) -> str:
        rendered = _render_line(m).removeprefix(f"- {m.model_id}")
        return re.sub(r"_\(Pareto:.*?\)_", "", rendered).strip()

    new_flags = {m.model_id: flags_for(m) for m in new_models}
    added = sorted(set(new_flags) - set(old))
    removed = sorted(set(old) - set(new_flags))
    flag_changes = sorted(
        (mid, old[mid], new_flags[mid])
        for mid in set(old) & set(new_flags)
        if old[mid] != new_flags[mid]
    )
    return IssueDiff(added=added, removed=removed, flag_changes=flag_changes)


def render_diff_comment(diff: IssueDiff) -> str:
    """Render the diff as a GitHub issue comment.

    Args:
        diff:
            The change set.

    Returns:
        Markdown text for the comment body.
    """
    parts = ["Here are some updates to the core model list:\n"]
    if diff.is_empty:
        parts.append("No changes since the previous run.")
        return "\n".join(parts)
    if diff.added:
        parts.append("**Added** (" + str(len(diff.added)) + "):\n")
        parts.extend(f"- {m}" for m in diff.added)
    if diff.removed:
        parts.append("\n**Removed** (" + str(len(diff.removed)) + "):\n")
        parts.extend(f"- {m}" for m in diff.removed)
    if diff.flag_changes:
        parts.append("\n**Flag changes** (" + str(len(diff.flag_changes)) + "):\n")
        for mid, old, new in diff.flag_changes:
            parts.append(f"- {mid}: `{old or '∅'}` → `{new or '∅'}`")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# GitHub API helpers (mirror the style of `collect_evaluation_results.py`)
# ---------------------------------------------------------------------------


def _gh_request(
    path: str, *, method: str = "GET", payload: dict | None = None, token: str
) -> dict | list:
    """Call the GitHub REST API.

    Args:
        path:
            Path component, e.g. ``/repos/owner/repo/issues/1186``.
        method:
            HTTP method.
        payload:
            JSON-serializable body for write methods.
        token:
            GitHub token with `issues: write`.

    Returns:
        Parsed JSON response body.
    """
    url = f"https://api.github.com{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url=url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "EuroEval-CoreModels/1",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_issue_body(issue_number: int, token: str) -> str:
    """Fetch the current body of an issue.

    Args:
        issue_number:
            The issue number.
        token:
            GitHub PAT.

    Returns:
        The issue body string (empty if the issue has none).
    """
    issue = _gh_request(path=f"/repos/{REPO}/issues/{issue_number}", token=token)
    return issue.get("body") or ""


def _update_issue_body(issue_number: int, body: str, token: str) -> None:
    """PATCH an issue with a new body.

    Args:
        issue_number:
            The issue number.
        body:
            The new body markdown.
        token:
            GitHub PAT.
    """
    _gh_request(
        path=f"/repos/{REPO}/issues/{issue_number}",
        method="PATCH",
        payload=dict(body=body),
        token=token,
    )


def _post_issue_comment(issue_number: int, body: str, token: str) -> str:
    """POST a comment to an issue.

    Args:
        issue_number:
            The issue number.
        body:
            The comment markdown.
        token:
            GitHub PAT.

    Returns:
        The `html_url` of the newly-created comment.
    """
    response = _gh_request(
        path=f"/repos/{REPO}/issues/{issue_number}/comments",
        method="POST",
        payload=dict(body=body),
        token=token,
    )
    assert isinstance(response, dict)
    return response["html_url"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the new issue body and diff but don't touch GitHub or the YAML.",
)
def main(dry_run: bool) -> None:
    """Refresh the core-model list and update issue #1186.

    The result archive is expected to be already processed (i.e. the
    `processed.jsonl` cache is up-to-date). Run `make leaderboards` first
    if you've ingested new results.

    Args:
        dry_run:
            Print outputs instead of writing them.
    """
    with CORE_MODELS_CONFIG.open("r") as f:
        config = safe_load(f)
    issue_number = int(config["issue_number"])
    eu_patterns = list(config.get("eu_model_patterns") or [])
    osai_overrides = list(config.get("osai_overrides") or [])
    logger.info(f"Issue: https://github.com/{REPO}/issues/{issue_number}")

    logger.info("Building core model list…")
    models = build_core_model_list(
        eu_patterns=eu_patterns, osai_overrides=osai_overrides
    )
    logger.info(f"Generated {len(models)} core models.")

    new_body = render_issue_body(models)
    today = dt.date.today()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.warning("GITHUB_TOKEN not set — skipping GitHub update.")
        if dry_run:
            click.echo(new_body)
        if not dry_run:
            _write_yaml(CORE_MODELS_CONFIG, today, models)
            logger.info(f"Wrote {CORE_MODELS_CONFIG}.")
        return

    try:
        old_body = _get_issue_body(issue_number=issue_number, token=token)
    except urllib.error.HTTPError as e:
        logger.error(f"Failed to fetch issue #{issue_number}: {e}")
        sys.exit(1)

    diff = diff_issue(old_body=old_body, new_models=models)
    comment = render_diff_comment(diff=diff)

    if dry_run:
        click.echo("=== NEW ISSUE BODY ===")
        click.echo(new_body)
        click.echo("=== DIFF COMMENT ===")
        click.echo(comment)
        return

    if new_body.strip() != old_body.strip():
        _update_issue_body(issue_number=issue_number, body=new_body, token=token)
        logger.info(f"Updated issue #{issue_number} body.")
    else:
        logger.info(f"Issue #{issue_number} body unchanged; skipping PATCH.")

    if not diff.is_empty:
        comment_url = _post_issue_comment(
            issue_number=issue_number, body=comment, token=token
        )
        logger.info(f"Posted diff comment: {comment_url}")
    else:
        logger.info("No diff to comment.")

    _write_yaml(CORE_MODELS_CONFIG, today, models)
    logger.info(f"Wrote {CORE_MODELS_CONFIG}.")


if __name__ == "__main__":
    main()
