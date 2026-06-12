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
import math
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
from leaderboards.task_metadata import languages_with_official_datasets

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s ⋅ %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("update_core_models")

load_dotenv()


REPO = "EuroEval/EuroEval"

# Catalonia has no national-flag emoji; the regional-indicator-tag form
# below is the canonical "subdivision flag" sequence (ES-CT). Renderers
# that don't support it degrade to the black-flag fallback, which is
# still recognisable next to the country flags around it.
_LANGUAGE_FLAG: dict[str, str] = {
    "albanian": "🇦🇱",
    "bosnian": "🇧🇦",
    "bulgarian": "🇧🇬",
    "catalan": "🏴󠁥󠁳󠁣󠁴󠁿",
    "croatian": "🇭🇷",
    "czech": "🇨🇿",
    "danish": "🇩🇰",
    "dutch": "🇳🇱",
    "english": "🇬🇧",
    "estonian": "🇪🇪",
    "faroese": "🇫🇴",
    "finnish": "🇫🇮",
    "french": "🇫🇷",
    "german": "🇩🇪",
    "greek": "🇬🇷",
    "hungarian": "🇭🇺",
    "icelandic": "🇮🇸",
    "italian": "🇮🇹",
    "latvian": "🇱🇻",
    "lithuanian": "🇱🇹",
    "norwegian": "🇳🇴",
    "polish": "🇵🇱",
    "portuguese": "🇵🇹",
    "romanian": "🇷🇴",
    "serbian": "🇷🇸",
    "slovak": "🇸🇰",
    "slovene": "🇸🇮",
    "spanish": "🇪🇸",
    "swedish": "🇸🇪",
    "ukrainian": "🇺🇦",
}

_ISSUE_INTRO = """## What is a "core model"?

To prevent an ever-growing list of models to evaluate every time we change a \
dataset or add a new language, we maintain this list of _core models_: the \
minimum set of models to re-run when such a change is made.

The list is generated automatically by running [this script][script] from \
three sources:

- ⭐ Pareto frontier within the model's type (encoder, base decoder, \
instruction-tuned decoder, reasoning decoder) — strictly better than every \
smaller-or-equal-sized model of the same type, in at least one language. \
Check out [this list][config] for the exact languages each model is \
evaluated on.
- 🇪🇺 Trained in the EU.
- 💜 Top-10 'truly open' models from [osai-index.eu][osai] (filtered to \
text models with open base weights, training code, and data sources).
- 👾 SOTA API model. One small + one large from each major lab; the list \
is hardcoded in [`api_models`][config] and refreshed manually as labs \
ship new flagship models.

[script]: https://github.com/EuroEval/EuroEval/blob/main/src/scripts/update_core_models.py
[config]: https://github.com/EuroEval/EuroEval/blob/main/src/leaderboards/core_models.yaml
[osai]: https://osai-index.eu/database/?type=text&weights_basemodel=1&trainingcode=1&datasources_basemodel=1
"""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _reasoning_flags(model: CoreModel) -> str:
    """Return the trio of emoji flags explaining why this model was picked.

    Args:
        model:
            The core model.

    Returns:
        Concatenated emoji string (⭐💜🇪🇺), possibly empty.
    """
    flags = []
    if model.pareto_languages:
        flags.append("⭐")
    if model.osai_rank is not None:
        flags.append("💜")
    if model.eu:
        flags.append("🇪🇺")
    if model.api:
        flags.append("👾")
    return "".join(flags)


def _format_parameters(parameters: float) -> str:
    """Render a parameter count in compact ``Xm`` / ``X.YB`` notation.

    Args:
        parameters:
            Number of parameters; NaN renders as a dash.

    Returns:
        Short display string for the issue table.
    """
    if parameters != parameters:  # NaN
        return "—"
    if parameters >= 1_000_000_000:
        value = parameters / 1_000_000_000
        return (f"{value:.1f}" if value < 10 else f"{value:.0f}") + "B"
    if parameters >= 1_000_000:
        return f"{parameters / 1_000_000:.0f}M"
    return f"{parameters:.0f}"


def _format_languages(model: CoreModel, all_languages: tuple[str, ...]) -> str:
    """Render the Languages column for the issue table.

    EU- or OSAI-listed models always get evaluated everywhere — those
    inclusions aren't language-scoped. Pareto-only inclusions get the
    flags of the languages they're on the frontier in, unless that
    happens to be every supported language, in which case we collapse
    to "All languages" so we don't splat 30 flags.

    Args:
        model:
            The core model record.
        all_languages:
            All languages with an official leaderboard.

    Returns:
        Space-separated flag emojis, or the string ``All languages``.
    """
    if model.eu or model.osai_rank is not None or model.api:
        return "All languages"
    if not model.pareto_languages or set(model.pareto_languages) >= set(all_languages):
        return "All languages"
    return " ".join(_LANGUAGE_FLAG.get(lang, lang) for lang in model.pareto_languages)


def render_issue_body(models: list[CoreModel], all_languages: tuple[str, ...]) -> str:
    """Render the full issue body for #1186.

    Args:
        models:
            The core model list, already sorted.
        all_languages:
            Every language that has an official leaderboard, used to
            collapse "all flags" down to "All languages".

    Returns:
        Markdown text suitable for `PATCH /repos/.../issues/1186`.
    """
    header = "| Model | Parameters | Labels | Languages |"
    separator = "| --- | --- | --- | --- |"
    rows = [header, separator]
    # Smallest first; models with unknown parameter counts sink to the
    # bottom so they don't pollute the top of the table.
    sorted_models = sorted(
        models,
        key=lambda m: (
            m.parameters if m.parameters == m.parameters else math.inf,
            m.model_id.lower(),
        ),
    )
    for model in sorted_models:
        rows.append(
            f"| {model.model_id} "
            f"| {_format_parameters(model.parameters)} "
            f"| {_reasoning_flags(model)} "
            f"| {_format_languages(model, all_languages)} |"
        )
    table = "\n".join(rows)
    return _ISSUE_INTRO + "\n\n" + table + "\n"


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
        lines.append(f"    api: {str(m.api).lower()}")
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
    for line in body.splitlines():
        stripped = line.strip()
        # The issue body lays models out in a Markdown table with columns
        # Model | Parameters | Labels | Languages. Skip the header row,
        # separator row, and any non-table content (intro, blank lines).
        if not stripped.startswith("|") or stripped.startswith("| ---"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 4:
            continue
        if cells[0].lower() == "model":
            continue
        result[cells[0]] = cells[2]
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

    new_flags = {m.model_id: _reasoning_flags(m) for m in new_models}
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
            parts.append(f"- {mid}: `{old or '(none)'}` -> `{new or '(none)'}`")
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
    refresh_core_models(dry_run=dry_run)


def refresh_core_models(dry_run: bool = False) -> list[CoreModel]:
    """Rebuild the core-model list, update the YAML, and sync issue #1186.

    Equivalent to invoking this script as a CLI, but callable in-process
    so other tools (e.g. ``run_core_model_evaluations.py``) can ensure
    the YAML and the GitHub issue are fresh before running downstream
    work.

    Args:
        dry_run (optional):
            When True, print the would-be issue body / diff comment and
            skip the YAML and GitHub writes. Defaults to False.

    Returns:
        The freshly-built core-model list, regardless of ``dry_run``.
    """
    with CORE_MODELS_CONFIG.open("r") as f:
        config = safe_load(f)
    issue_number = int(config["issue_number"])
    eu_patterns = list(config.get("eu_model_patterns") or [])
    osai_overrides = list(config.get("osai_overrides") or [])
    api_model_ids = list(config.get("api_models") or [])
    logger.info(f"Issue: https://github.com/{REPO}/issues/{issue_number}")

    logger.info("Building core model list...")
    models = build_core_model_list(
        eu_patterns=eu_patterns,
        api_model_ids=api_model_ids,
        osai_overrides=osai_overrides,
    )
    logger.info(f"Generated {len(models)} core models.")

    all_languages = tuple(languages_with_official_datasets())
    new_body = render_issue_body(models, all_languages=all_languages)
    today = dt.date.today()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.warning("GITHUB_TOKEN not set -- skipping GitHub update.")
        if dry_run:
            click.echo(new_body)
        else:
            _write_yaml(CORE_MODELS_CONFIG, today, models)
            logger.info(f"Wrote {CORE_MODELS_CONFIG}.")
        return models

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
        return models

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
    return models


if __name__ == "__main__":
    main()
