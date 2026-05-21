"""Generate all leaderboards."""

import datetime as dt
import logging
import re
import subprocess
import sys
import typing as t
import warnings
from pathlib import Path

import click
from dotenv import load_dotenv
from yaml import safe_load

from leaderboards.backup import backup_results, restore_from_backup_if_missing
from leaderboards.leaderboard_generation import generate_leaderboard
from leaderboards.paths import CORE_MODELS_CONFIG, LEADERBOARD_CONFIGS_DIR
from leaderboards.result_processing import process_results
from leaderboards.task_metadata import languages_with_official_datasets

try:
    # Normal invocation: `python -m scripts.generate_leaderboards`, with
    # `src/` on `sys.path` so `scripts` resolves as a package.
    from scripts.generate_task_metrics import main as generate_task_metrics
except ImportError:
    # File-path invocation (e.g. `uv run src/scripts/generate_leaderboards.py`)
    # — `scripts` isn't a package on the path, so import the sibling module
    # directly.
    from generate_task_metrics import (
        main as generate_task_metrics,  # type: ignore[no-redef]
    )

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s ⋅ %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

warnings.simplefilter(action="ignore", category=RuntimeWarning)

load_dotenv()


# Constants for leaderboard generation
MINIMUM_VERSION: str = "15.0.0"
MINIMUM_NUMBER_OF_MODEL_RECORDS: int = 7
BANNED_VERSIONS: list[str] = ["9.3.0", "10.0.0"]
BANNED_MODEL_PATTERNS: list[re.Pattern] = [
    re.compile("^meta-llama/Llama-3.1-405B-Instruct$"),  # Temporary ban
    re.compile("^utter-project/EuroVLM-9B-Preview$"),  # Temporary ban
]
API_MODEL_PATTERNS: list[re.Pattern] = [
    re.compile(r"gemini/.*"),
    re.compile(r"(openai/)?gpt-[456789].*"),
    re.compile(r"(anthropic/)?claude.*"),
    re.compile(r"(xai/)?grok.*"),
]
OPEN_SOURCE_MODEL_PATTERNS: list[re.Pattern] = []
TRAINED_FROM_SCRATCH_PATTERNS: list[re.Pattern] = [
    re.compile(r"Qwen/.*"),
    re.compile(r"google/.*"),
    re.compile(r"mistralai/.*"),
    re.compile(r"meta-llama/.*"),
    re.compile(r"facebook/.*"),
    re.compile(r"FacebookAI/.*"),
    re.compile(r"zai-org/.*"),
    re.compile(r"deepseek-ai/.*"),
    re.compile(r"PleIAs/.*"),
    re.compile(r"openai/.*"),
    re.compile(r"nvidia/.*"),
    re.compile(r"allenai/.*"),
    re.compile(r"utter-project/.*"),
    re.compile(r"CohereLabs/.*"),
    re.compile(r"speakleash/.*"),
    re.compile(r"yulan-team/.*"),
    re.compile(r"BSC-LT/.*"),
    re.compile(r"tencent/.*"),
    re.compile(r"LiquidAI/.*"),
    re.compile(r"HuggingFaceTB/.*"),
    re.compile(r"tiiuae/.*"),
    re.compile(r"AIDC-AI/.*"),
    re.compile(r"inclusionAI/.*"),
    re.compile(r"jhu-clsp/.*"),
    re.compile(r"vesteinn/(Dansk|Fo|Scandi)BERT.*"),
    re.compile(r"EuropeanParliament/EUBERT"),
    re.compile(r"microsoft/.*"),
    re.compile(r"EuroBERT/.*"),
    re.compile(r"fresh-.*"),
    re.compile(r"answerdotai/.*"),
    re.compile(r".*-scratch"),
]


@click.command()
@click.option(
    "--categories",
    "-c",
    default=["generative", "all_models"],
    multiple=True,
    help=(
        "Categories to generate leaderboards for. Defaults to 'generative' and "
        "'all_models'."
    ),
)
@click.option(
    "--force/--no-force",
    "-f",
    default=False,
    show_default=True,
    help="Force the generation of the leaderboard, even if no updates are found.",
)
@click.option(
    "--skip-core-models-check",
    is_flag=True,
    default=False,
    help=(
        "Skip the staleness prompt for the core-model list. Useful in non-"
        "interactive runs (CI, batch jobs)."
    ),
)
def main(
    categories: tuple[t.Literal["generative", "all_models"]],
    force: bool,
    skip_core_models_check: bool,
) -> None:
    """Generate all leaderboards.

    Args:
        categories (optional):
            Categories to generate leaderboards for. Defaults to 'generative' and
            'all_models'.
        force (optional):
            Whether to force the generation of the leaderboard, even if no updates
            are found. Defaults to False.
        skip_core_models_check (optional):
            If True, skip prompting to refresh the core-model list when stale.
    """
    # If results.tar.gz isn't here, pull the newest backup into place.
    restore_from_backup_if_missing()

    process_results(
        min_version=MINIMUM_VERSION,
        min_number_of_model_records=MINIMUM_NUMBER_OF_MODEL_RECORDS,
        banned_versions=BANNED_VERSIONS,
        banned_model_patterns=BANNED_MODEL_PATTERNS,
        api_model_patterns=API_MODEL_PATTERNS,
        trained_from_scratch_patterns=TRAINED_FROM_SCRATCH_PATTERNS,
    )

    # Offer to refresh the core-model list if it hasn't been touched in
    # over a month. Doing this inside `generate_leaderboards` keeps it on
    # the maintainer's radar without forcing a slow re-process each time.
    if not skip_core_models_check:
        _maybe_refresh_core_models()
    # Monolingual leaderboards are derived directly from the lib: one per
    # language that has at least one official leaderboard dataset.
    for language in languages_with_official_datasets():
        generate_leaderboard(
            leaderboard_name=language,
            language_names=[language],
            categories=list(categories),
            force=force,
        )
    # Multilingual leaderboards stay yaml-configured since they encode a
    # curated grouping (e.g. Mainland Scandinavian, Slavic).
    for config_path in sorted(LEADERBOARD_CONFIGS_DIR.glob("*.yaml")):
        with config_path.open(mode="r") as f:
            config = safe_load(stream=f)
        generate_leaderboard(
            leaderboard_name=config_path.stem,
            language_names=list(config["languages"]),
            categories=list(categories),
            force=force,
        )

    # Keep the frontend's task -> metric-names map in sync with euroeval.
    generate_task_metrics()

    # Snapshot the (possibly updated) results to the backup directory,
    # rotating out oldest backups to keep total size under the cap.
    try:
        backup_results()
    except OSError as exc:  # pCloud unavailable / disk full / etc.
        logging.warning(f"Results backup failed: {exc}")


CORE_MODELS_STALE_DAYS = 30


def _maybe_refresh_core_models() -> None:
    """Prompt the user to refresh the core-model list if it's stale.

    Reads `last_updated` from `core_models.yaml`. If it's missing or older
    than `CORE_MODELS_STALE_DAYS`, asks the user whether to invoke the
    updater now. Skipped silently when stdin isn't a TTY (CI, piped
    invocations).
    """
    if not sys.stdin.isatty():
        return
    try:
        with CORE_MODELS_CONFIG.open("r") as f:
            config = safe_load(f) or {}
    except OSError as exc:
        logging.warning(f"Core models config unreadable: {exc}")
        return

    last_updated_raw = config.get("last_updated")
    if last_updated_raw is None:
        prompt = "Core model list has never been generated. Refresh now?"
    else:
        if isinstance(last_updated_raw, dt.date):
            last = last_updated_raw
        else:
            try:
                last = dt.date.fromisoformat(str(last_updated_raw))
            except ValueError:
                logging.warning(
                    f"Cannot parse last_updated={last_updated_raw!r}; "
                    "treating as stale."
                )
                last = None  # type: ignore[assignment]
        if last is not None:
            age_days = (dt.date.today() - last).days
            if age_days < CORE_MODELS_STALE_DAYS:
                return
            prompt = f"Core model list is {age_days} days old. Refresh now?"
        else:
            prompt = "Core model list timestamp is unparseable. Refresh now?"

    if not click.confirm(prompt, default=False):
        return

    # Spawn as a script. `process_results` already ran above, and the
    # updater reuses the same processed cache.
    script_path = Path(__file__).resolve().parent / "update_core_models.py"
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
    except subprocess.CalledProcessError as exc:
        logging.warning(f"update_core_models failed (exit {exc.returncode}).")


if __name__ == "__main__":
    main()
