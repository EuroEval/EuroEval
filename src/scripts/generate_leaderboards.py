"""Generate all leaderboards."""

import datetime as dt
import json
import logging
import subprocess
import sys
import typing as t
import warnings
from pathlib import Path

import click
from dotenv import load_dotenv
from yaml import safe_load

from euroeval.string_utils import split_model_id
from leaderboards.backup import backup_results, restore_from_backup_if_missing
from leaderboards.constants import (
    API_MODEL_PATTERNS,
    BANNED_MODEL_PATTERNS,
    BANNED_VERSIONS,
    CORE_MODELS_CONFIG,
    CORE_MODELS_STALE_DAYS,
    LEADERBOARD_CONFIGS_DIR,
    LEADERBOARD_TASKS,
    MINIMUM_NUMBER_OF_MODEL_RECORDS,
    MINIMUM_VERSION,
    MODELS_PY_PATH,
    REPO_ROOT,
    TRAINED_FROM_SCRATCH_PATTERNS,
)
from leaderboards.leaderboard_generation import generate_leaderboard
from leaderboards.records import get_dataset, plain_model_id
from leaderboards.result_loading import load_raw_results
from leaderboards.result_processing import process_results
from leaderboards.score_extraction import extract_model_metadata
from leaderboards.task_metadata import (
    languages_with_official_datasets,
    official_datasets_for_language,
    task_metric_pretty_names,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


logger = logging.getLogger(__name__)

warnings.simplefilter(action="ignore", category=RuntimeWarning)

load_dotenv()


# Constants for leaderboard generation
@click.command()
@click.option(
    "--categories",
    "-c",
    default=("generative", "all_models"),
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
@click.option(
    "--skip-results-processing",
    is_flag=True,
    default=False,
    help=(
        "Skip processing evaluation results from JSONL. Assumes the results "
        "directory already contains processed results. Useful for repeated "
        "leaderboard generation when results haven't changed."
    ),
)
def main(
    categories: tuple[t.Literal["generative", "all_models"], ...],
    force: bool,
    skip_core_models_check: bool,
    skip_results_processing: bool,
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
        skip_results_processing (optional):
            If True, skip processing evaluation results from JSONL. Assumes the
            results directory already contains processed results.
    """
    # If the results directory isn't populated, restore the newest backup.
    restore_from_backup_if_missing()

    if not skip_results_processing:
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
    # curated grouping (e.g. Scandinavian, Slavic).
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

    # Keep the HF Space's model list in sync so it shows up on model cards.
    generate_model_list()

    # Snapshot the (possibly updated) results to the backup directory,
    # rotating out oldest backups to keep total size under the cap.
    try:
        backup_results()
    except OSError as exc:  # pCloud unavailable / disk full / etc.
        logger.warning(f"Results backup failed: {exc}")


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
        logger.warning(f"Core models config unreadable: {exc}")
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
                logger.warning(
                    f"Cannot parse last_updated={last_updated_raw!r}; "
                    "treating as stale."
                )
                last = None
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
        logger.warning(f"update_core_models failed (exit {exc.returncode}).")


def generate_model_list() -> None:
    """Generate the models.py file for upload to the leaderboard HF Space."""
    all_datasets = {
        dataset
        for language in languages_with_official_datasets()
        for task_datasets in official_datasets_for_language(language).values()
        for dataset in task_datasets
    }
    results = [
        r
        for r in load_raw_results()
        if get_dataset(r) in all_datasets and not r.get("use_bits_per_character", False)
    ]
    model_metadata_dict = extract_model_metadata(results=results)
    model_ids = sorted(
        {
            split_model_id(model_id=plain_model_id(model_id)).model_id
            for model_id, model_metadata in model_metadata_dict.items()
            if model_metadata.get("open", False)
        }
    )
    MODELS_PY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODELS_PY_PATH.open(mode="w") as f:
        f.write('"""Auto-generated list of models in the EuroEval leaderboards."""\n\n')
        f.write("MODEL_NAMES = [\n")
        f.writelines(f'    "{model_id}",\n' for model_id in model_ids)
        f.write("]\n")
    logger.info(f"Wrote {MODELS_PY_PATH.relative_to(REPO_ROOT)}")


def generate_task_metrics() -> None:
    """Generate the task-metrics JSON file."""
    output_path: Path = (
        REPO_ROOT / "src" / "frontend" / "generated" / "task-metrics.json"
    )
    payload: dict[str, list[str]] = {}
    for task in LEADERBOARD_TASKS:
        primary, secondary = task_metric_pretty_names(task)
        payload[task] = [primary] + ([secondary] if secondary is not None else [])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode="w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info(f"Wrote {output_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
