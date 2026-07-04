r"""Replace an official leaderboard dataset with a new one, end to end.

When a new dataset should take over a task slot on a language's leaderboard
from the current official one, three things have to happen:

1. **Evaluate.** Every model that currently holds a full rank score on the
   affected leaderboard(s) must be evaluated on the new (still unofficial)
   dataset -- mirroring exactly how each model appears on the leaderboard
   (validation vs test split, zero-shot vs few-shot). This runs *before* the
   official flags are flipped, so the live leaderboard stays intact while the
   data is gathered.
2. **Swap the configs.** The outgoing dataset is demoted to unofficial and the
   incoming one promoted to official, in both the ``euroeval`` dataset configs
   and the frontend dataset documentation, keeping the official-first grouping
   each file uses.
3. **Open a PR** (optional) with the config/doc changes.

Everything happens on a dedicated branch -- ``--branch`` is required and may
not be the default branch.

Invoke as::

    uv run src/scripts/swap_leaderboard_dataset.py \\
        --old-dataset scala-da --new-dataset dala --branch swap-scala-dala [--pr]

Required env vars (open-weight models)
--------------------------------------
HF_TOKEN          Resolved via :func:`evaluation_common.resolve_hf_token`.

Required env vars (API models, only with --include-api)
-------------------------------------------------------
OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY / XAI_API_KEY
"""

from __future__ import annotations

import collections.abc as c
import json
import logging
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import click

from euroeval.constants import ORTHOGONAL_TASKS
from euroeval.data_models import DatasetConfig
from euroeval.dataset_configs import get_all_dataset_configs
from euroeval.languages import get_all_languages
from leaderboards.constants import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    LEADERBOARD_CATEGORIES,
    NEW_RESULTS_PATH,
    RESULTS_DIR,
)
from leaderboards.evaluation_common import (
    gpu_total_memory_bytes,
    model_fits_locally,
    run_euroeval,
)
from leaderboards.records import get_bool_field, get_dataset, plain_model_id
from leaderboards.task_metadata import (
    category_includes_task,
    official_datasets_for_language,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("swap_leaderboard_dataset")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIG_DIR = REPO_ROOT / "src" / "euroeval" / "dataset_configs"
DATASET_DOC_DIR = REPO_ROOT / "src" / "frontend" / "md" / "datasets"
UNOFFICIAL_MARKER = "# Unofficial datasets ###"
UNOFFICIAL_LINE_RE = re.compile(r"^\s*unofficial\s*=\s*True\s*,\s*$")
DOC_UNOFFICIAL_PREFIX = "Unofficial: "
DEFAULT_REVIEWER = "saattrupdan"


@click.command()
@click.option(
    "--old-dataset",
    "old_dataset",
    required=True,
    help="The official dataset being replaced (demoted to unofficial).",
)
@click.option(
    "--new-dataset",
    "new_dataset",
    required=True,
    help="The unofficial candidate dataset being promoted to official.",
)
@click.option(
    "--branch",
    required=True,
    help="Branch to do the work on. May not be the default branch (e.g. main).",
)
@click.option(
    "--language",
    "languages",
    multiple=True,
    help="Restrict re-evaluation to these language codes. When omitted, defaults "
    "to the languages both datasets cover.",
)
@click.option(
    "--include-api",
    is_flag=True,
    default=False,
    help="Opt in to evaluating API models. Without it they are skipped, so a "
    "plain run never spends money.",
)
@click.option(
    "--api-providers",
    default=None,
    help="Comma-separated provider names to run (openai, anthropic, google, xai). "
    "Requires --include-api.",
)
@click.option(
    "--gpu-memory-utilization",
    "gpu_memory_utilization",
    type=click.FloatRange(min=0.0, max=1.0, min_open=True),
    default=None,
    help="vLLM GPU memory utilization fraction (0.0-1.0) the fit pre-check budgets "
    f"against. When omitted, defaults to {DEFAULT_GPU_MEMORY_UTILIZATION}.",
)
@click.option(
    "--skip-eval",
    is_flag=True,
    default=False,
    help="Skip the evaluation phase and only perform the config/doc swap (useful "
    "when the evaluations already ran).",
)
@click.option(
    "--pr",
    is_flag=True,
    default=False,
    help="After swapping, commit and push the branch and open a pull request.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-run even (model, language) pairs that already have a new-dataset "
    "result line.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the planned evaluations and file edits without running or "
    "modifying anything.",
)
def main(
    old_dataset: str,
    new_dataset: str,
    branch: str,
    languages: tuple[str, ...],
    include_api: bool,
    api_providers: str | None,
    gpu_memory_utilization: float | None,
    skip_eval: bool,
    pr: bool,
    force: bool,
    dry_run: bool,
) -> None:
    """Replace an official leaderboard dataset with a new one.

    Args:
        old_dataset:
            The official dataset being replaced.
        new_dataset:
            The unofficial candidate being promoted.
        branch:
            The branch to do the work on; may not be the default branch.
        languages:
            Language codes to restrict evaluation to; empty means the codes
            both datasets cover.
        include_api:
            Whether to evaluate API models.
        api_providers:
            Optional comma-separated provider filter.
        gpu_memory_utilization:
            vLLM GPU memory utilization fraction, or None for the default.
        skip_eval:
            When True, only perform the config/doc swap.
        pr:
            When True, commit, push, and open a pull request after swapping.
        force:
            When True, re-run pairs that already have a new-dataset result.
        dry_run:
            When True, print the plan and make no changes.

    Raises:
        click.ClickException:
            When the dataset pair or branch is invalid, or a git/gh step fails.
    """
    if api_providers and not include_api:
        raise click.ClickException(
            "--api-providers requires --include-api; pass both or neither."
        )
    old_config, new_config = validate_datasets(
        old_dataset=old_dataset, new_dataset=new_dataset
    )
    validate_branch(branch=branch)

    target_codes = resolve_languages(
        old_config=old_config, new_config=new_config, requested=languages
    )
    logger.info(
        f"Swap {old_dataset!r} -> {new_dataset!r} (task {old_config.task.name!r}) "
        f"on language(s): {', '.join(sorted(target_codes))}."
    )

    if not dry_run:
        checkout_branch(branch=branch)

    if skip_eval:
        logger.info("--skip-eval set; skipping the evaluation phase.")
    else:
        run_evaluations(
            old_dataset=old_dataset,
            new_dataset=new_dataset,
            swapped_task=old_config.task.name,
            target_codes=target_codes,
            include_api=include_api,
            api_providers_arg=api_providers,
            gpu_memory_utilization=gpu_memory_utilization,
            force=force,
            dry_run=dry_run,
        )

    changed = apply_swap(
        old_dataset=old_dataset, new_dataset=new_dataset, dry_run=dry_run
    )

    if dry_run:
        logger.info("Dry run complete; no evaluations ran and no files changed.")
        return

    if pr:
        open_pull_request(
            old_dataset=old_dataset,
            new_dataset=new_dataset,
            branch=branch,
            changed_paths=changed,
        )
    else:
        logger.info(
            "Swap complete on branch %r. Re-run with --pr to open a pull request, "
            "or commit the changes manually.",
            branch,
        )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_datasets(
    old_dataset: str, new_dataset: str
) -> tuple[DatasetConfig, DatasetConfig]:
    """Validate the dataset pair and return their configs.

    The swap runs before the flags are flipped, so the outgoing dataset must
    still be official and the incoming candidate still unofficial, both must be
    configured, and both must belong to the same task.

    Args:
        old_dataset:
            The outgoing dataset id (expected official).
        new_dataset:
            The incoming candidate id (expected unofficial).

    Returns:
        The ``(old_config, new_config)`` pair.

    Raises:
        click.ClickException:
            When a dataset is unknown, mis-flagged, or the tasks differ.
    """
    old_config = dataset_config(dataset_id=old_dataset)
    new_config = dataset_config(dataset_id=new_dataset)
    if old_config is None:
        raise click.ClickException(f"--old-dataset {old_dataset!r} has no config.")
    if new_config is None:
        raise click.ClickException(f"--new-dataset {new_dataset!r} has no config.")
    if old_config.unofficial:
        raise click.ClickException(
            f"--old-dataset {old_dataset!r} must be official; it is unofficial."
        )
    if not new_config.unofficial:
        raise click.ClickException(
            f"--new-dataset {new_dataset!r} must be unofficial; it is official."
        )
    if old_config.task.name != new_config.task.name:
        raise click.ClickException(
            f"Both datasets must share a task; {old_dataset!r} is "
            f"{old_config.task.name!r} but {new_dataset!r} is {new_config.task.name!r}."
        )
    return old_config, new_config


def validate_branch(branch: str) -> None:
    """Reject an empty branch name or the default branch.

    Args:
        branch:
            The requested branch name.

    Raises:
        click.ClickException:
            When the branch is empty or is the repo's default branch.
    """
    if not branch.strip():
        raise click.ClickException("--branch must be a non-empty branch name.")
    default = default_branch()
    if branch in {"main", "master", default}:
        raise click.ClickException(
            f"--branch may not be the default branch ({default!r}); pick a new "
            "branch name for the swap."
        )


def resolve_languages(
    old_config: DatasetConfig, new_config: DatasetConfig, requested: tuple[str, ...]
) -> set[str]:
    """Resolve the language codes whose leaderboards are affected.

    Args:
        old_config:
            The outgoing dataset config.
        new_config:
            The incoming dataset config.
        requested:
            Language codes passed via ``--language``; empty means "both".

    Returns:
        The language codes to operate on.

    Raises:
        click.ClickException:
            When no languages remain after intersecting the datasets.
    """
    old_codes = {language.code for language in old_config.languages}
    new_codes = {language.code for language in new_config.languages}
    if requested:
        target = set(requested)
    else:
        target = old_codes & new_codes
    if not target:
        raise click.ClickException(
            "The two datasets share no languages; pass --language explicitly."
        )
    return target


# --------------------------------------------------------------------------- #
# Evaluation phase
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Job:
    """A single evaluation of one model on the new dataset in one language."""

    model_id: str
    languages: tuple[str, ...]
    is_api: bool
    evaluate_test_split: bool
    zero_shot: bool


@dataclass(frozen=True)
class _ObsConfig:
    """How a model was evaluated on a (dataset, language), for mirroring."""

    validation_split: bool
    few_shot: bool
    generative: bool


@dataclass(frozen=True)
class _Corpus:
    """Parsed results indexed for ranked-model selection and mirroring."""

    datasets_by_language: dict[str, dict[str, set[str]]]
    api_model_ids: set[str]
    observations: set[tuple[str, str, str]]
    eval_configs: dict[tuple[str, str, str], _ObsConfig]


def run_evaluations(
    old_dataset: str,
    new_dataset: str,
    swapped_task: str,
    target_codes: set[str],
    include_api: bool,
    api_providers_arg: str | None,
    gpu_memory_utilization: float | None,
    force: bool,
    dry_run: bool,
) -> None:
    """Evaluate every ranked model on the new dataset, mirroring their setup.

    Args:
        old_dataset:
            The outgoing dataset (defines the ranked-model set).
        new_dataset:
            The dataset to evaluate on.
        swapped_task:
            The task both datasets belong to.
        target_codes:
            The affected language codes.
        include_api:
            Whether to evaluate API models.
        api_providers_arg:
            Optional comma-separated provider filter.
        gpu_memory_utilization:
            vLLM GPU memory utilization fraction, or None for the default.
        force:
            When True, re-run pairs already holding a new-dataset result.
        dry_run:
            When True, print the plan without running.
    """
    corpus = load_corpus()
    ranked = ranked_model_language_pairs(
        old_dataset=old_dataset,
        new_dataset=new_dataset,
        swapped_task=swapped_task,
        language_codes=target_codes,
        datasets_by_language=corpus.datasets_by_language,
    )
    logger.info(f"Found {len(ranked)} ranked (model, language) pair(s).")

    ranked_api = sorted({m for m, _ in ranked if m in corpus.api_model_ids})
    present_providers = {
        provider.name
        for model_id in ranked_api
        if (provider := provider_for_model_id(model_id=model_id)) is not None
    }
    selected_providers = resolve_api_providers(
        include_api=include_api,
        api_providers_arg=api_providers_arg,
        present_providers=present_providers,
    )
    if ranked_api and not include_api:
        logger.info(
            f"Excluding {len(ranked_api)} ranked API model(s); pass --include-api "
            "to evaluate them."
        )

    jobs = build_eval_jobs(
        ranked=ranked,
        old_dataset=old_dataset,
        new_dataset=new_dataset,
        corpus=corpus,
        include_api=include_api,
        selected_providers=selected_providers,
        force=force,
    )
    logger.info(f"Planned {len(jobs)} evaluation(s) before the size check.")
    jobs = apply_size_filter(jobs=jobs, gpu_memory_utilization=gpu_memory_utilization)
    logger.info(f"{len(jobs)} evaluation(s) survive the size check.")

    if dry_run:
        for job in jobs:
            tag = "api" if job.is_api else "open"
            split = "test" if job.evaluate_test_split else "val"
            shots = "zero-shot" if job.zero_shot else "few-shot"
            click.echo(
                f"[{tag}] {job.model_id} :: {new_dataset} :: "
                f"{', '.join(job.languages)} :: {split}, {shots}"
            )
        return

    execute_jobs(
        jobs=jobs, dataset=new_dataset, gpu_memory_utilization=gpu_memory_utilization
    )


def load_corpus() -> _Corpus:
    """Load the recorded results, indexed for selection and mirroring.

    Reads the per-model JSONL files in ``RESULTS_DIR`` plus the optional
    ``new_results.jsonl`` without the destructive unlink the leaderboard
    pipeline performs. A model counts as an API model when its record was
    produced by the ``litellm`` engine or is flagged as not open-weight. Each
    ``(model, dataset, language)`` triple records its leaderboard variant
    (split + prompting), preferring the test-split record when both exist
    (that is the row the leaderboard shows).

    Returns:
        The parsed corpus.

    Raises:
        click.ClickException:
            When no results can be loaded.
    """
    lines: list[str] = []
    for model_file in sorted(RESULTS_DIR.glob("*.jsonl")):
        lines.extend(model_file.read_text(encoding="utf-8").splitlines())
    if NEW_RESULTS_PATH.exists():
        lines.extend(NEW_RESULTS_PATH.read_text(encoding="utf-8").splitlines())
    if not lines:
        raise click.ClickException(
            f"No results found under {RESULTS_DIR}; cannot find ranked models."
        )

    datasets_by_language: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    api_model_ids: set[str] = set()
    observations: set[tuple[str, str, str]] = set()
    eval_configs: dict[tuple[str, str, str], _ObsConfig] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        for record_text in re.split(pattern=r"(?<=})(?={)", string=line):
            if not record_text.strip():
                continue
            try:
                record = json.loads(record_text)
            except json.JSONDecodeError:
                continue
            model = plain_model_id(str(record.get("model_info", {}).get("name", "")))
            dataset = get_dataset(record=record)
            if not model or not dataset:
                continue
            model_info = record.get("model_info", {})
            if record_is_api(model_info=model_info):
                api_model_ids.add(model)
            config = _ObsConfig(
                validation_split=get_bool_field(record, "validation_split", False),
                few_shot=get_bool_field(record, "few_shot", True),
                generative=str(
                    model_info.get("additional_details", {}).get("generative")
                ).lower()
                == "true",
            )
            for language in _record_languages(record=record):
                datasets_by_language[language][model].add(str(dataset))
                key = (model, str(dataset), language)
                observations.add(key)
                existing = eval_configs.get(key)
                # Prefer the test-split record: when a model has both, the
                # leaderboard row shows the test-split variant.
                if existing is None or (
                    not config.validation_split and existing.validation_split
                ):
                    eval_configs[key] = config
    logger.info(
        f"Loaded results for {len(datasets_by_language):,} language(s) "
        f"({len(api_model_ids):,} API model(s))."
    )
    return _Corpus(
        datasets_by_language=datasets_by_language,
        api_model_ids=api_model_ids,
        observations=observations,
        eval_configs=eval_configs,
    )


def ranked_model_language_pairs(
    old_dataset: str,
    new_dataset: str,
    swapped_task: str,
    language_codes: set[str],
    datasets_by_language: dict[str, dict[str, set[str]]],
) -> set[tuple[str, str]]:
    """Return ``(model_id, language)`` pairs ranked on the affected leaderboards.

    A model is ranked in a language when it holds a result for every required
    (non-orthogonal) dataset of that language's single-language leaderboard, in
    any leaderboard category the swapped task belongs to. The ``generative``
    category scores all tasks; ``all_models`` scores only NLU tasks so encoder
    models can be compared. A model ranked in *either* category is selected, so
    encoder models are included whenever the swapped task is one they can run.

    Args:
        old_dataset:
            The outgoing dataset (kept in the required set).
        new_dataset:
            The incoming candidate (kept out of the required set).
        swapped_task:
            The task both datasets belong to.
        language_codes:
            The affected language codes.
        datasets_by_language:
            ``{language: {model: {dataset, ...}}}`` from the corpus.

    Returns:
        The ranked ``(model_id, language)`` pairs.
    """
    languages = get_all_languages()
    affected = [
        category
        for category in LEADERBOARD_CATEGORIES
        if category_includes_task(category=category, task=swapped_task)
    ]
    if not affected:
        logger.warning(f"Task {swapped_task!r} is in no leaderboard category.")
        return set()

    ranked: set[tuple[str, str]] = set()
    for code in sorted(language_codes):
        language = languages.get(code)
        if language is None:
            logger.warning(f"Unknown language code {code!r}; skipping.")
            continue
        name = language.name.lower()
        if " " in name:
            logger.warning(f"{code!r} ({name!r}) has no standalone leaderboard.")
            continue
        try:
            by_task = official_datasets_for_language(name)
        except ValueError:
            logger.warning(f"No leaderboard datasets for {name!r}; skipping.")
            continue

        models_in_language = datasets_by_language.get(code, {})
        for category in affected:
            required = {
                dataset
                for task, task_datasets in by_task.items()
                if task not in ORTHOGONAL_TASKS
                and category_includes_task(category=category, task=task)
                for dataset in task_datasets
            }
            required.discard(new_dataset)
            required.add(old_dataset)
            if not required:
                continue
            for model_id, datasets in models_in_language.items():
                if required <= datasets:
                    ranked.add((model_id, code))
    return ranked


def build_eval_jobs(
    ranked: set[tuple[str, str]],
    old_dataset: str,
    new_dataset: str,
    corpus: _Corpus,
    include_api: bool,
    selected_providers: set[str],
    force: bool,
) -> list[Job]:
    """Turn ranked pairs into evaluation jobs, mirroring each model's setup.

    Args:
        ranked:
            The ranked ``(model_id, language)`` pairs.
        old_dataset:
            The outgoing dataset (whose recorded setup is mirrored).
        new_dataset:
            The dataset to evaluate on.
        corpus:
            The parsed corpus.
        include_api:
            Whether API models are evaluated.
        selected_providers:
            Provider names whose env vars are set; a known-provider API model
            from another provider is skipped.
        force:
            When True, keep pairs already holding a new-dataset result.

    Returns:
        One :class:`Job` per model, its languages grouped, so identical
        split/prompting settings run together.
    """
    # Group languages per model when the mirrored settings match, so one
    # euroeval call covers them.
    by_model: dict[tuple[str, bool, bool, bool], list[str]] = defaultdict(list)
    for model_id, code in sorted(ranked):
        is_api = model_id in corpus.api_model_ids
        if is_api:
            if not include_api:
                continue
            provider = provider_for_model_id(model_id=model_id)
            if provider is not None and provider.name not in selected_providers:
                continue
        if not force and (model_id, new_dataset, code) in corpus.observations:
            continue
        config = corpus.eval_configs.get((model_id, old_dataset, code))
        evaluate_test_split, zero_shot = mirror_eval_config(
            config=config, is_api=is_api
        )
        by_model[(model_id, is_api, evaluate_test_split, zero_shot)].append(code)

    jobs: list[Job] = []
    for (model_id, is_api, evaluate_test_split, zero_shot), codes in by_model.items():
        jobs.append(
            Job(
                model_id=model_id,
                languages=tuple(sorted(codes)),
                is_api=is_api,
                evaluate_test_split=evaluate_test_split,
                zero_shot=zero_shot,
            )
        )
    return jobs


def mirror_eval_config(config: _ObsConfig | None, is_api: bool) -> tuple[bool, bool]:
    """Return ``(evaluate_test_split, zero_shot)`` matching the leaderboard row.

    Mirrors how the model appears on the leaderboard for the outgoing dataset:
    the ``(val)`` variant means the validation split, otherwise the test split;
    the ``(zero-shot)`` variant means zero-shot, and only generative models are
    ever run zero-shot. When no record is available, fall back to the model-type
    default (API: validation split, zero-shot; open-weight: test, few-shot).

    Args:
        config:
            The recorded setup for the outgoing dataset, or None.
        is_api:
            Whether the model is an API model (fallback only).

    Returns:
        The ``(evaluate_test_split, zero_shot)`` flags.
    """
    if config is None:
        return (not is_api), is_api
    evaluate_test_split = not config.validation_split
    zero_shot = config.generative and not config.few_shot
    return evaluate_test_split, zero_shot


def apply_size_filter(
    jobs: list[Job], gpu_memory_utilization: float | None
) -> list[Job]:
    """Drop open-weight jobs whose model can't fit the local GPU budget.

    Budgets against ``gpu_memory_utilization * total GPU memory`` (matching
    ``process_evaluation_queue.py``) rather than the whole card, so a model
    whose weights fit but whose KV cache does not is dropped up front. API jobs
    and un-measurable models pass through.

    Args:
        jobs:
            The planned jobs.
        gpu_memory_utilization:
            The utilization fraction, or None for the default.

    Returns:
        The jobs that should still run.
    """
    gpu_bytes = gpu_total_memory_bytes()
    if gpu_bytes is None:
        logger.info("Local memory budget unknown; skipping the size pre-check.")
        return jobs
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
    sized: dict[str, bool] = {}
    kept: list[Job] = []
    for job in jobs:
        if job.is_api:
            kept.append(job)
            continue
        if job.model_id not in sized:
            fits, needed = model_fits_locally(
                model_id=job.model_id, gpu_bytes=usable_bytes
            )
            sized[job.model_id] = fits
            if not fits and needed is not None:
                logger.info(
                    f"{job.model_id}: skipping -- needs "
                    f"~{needed / (1024**3):.1f} GiB, exceeds the usable "
                    f"{usable_bytes / (1024**3):.1f} GiB budget."
                )
        if sized[job.model_id]:
            kept.append(job)
    return kept


def execute_jobs(
    jobs: list[Job], dataset: str, gpu_memory_utilization: float | None
) -> None:
    """Run each evaluation in sequence via the shared euroeval runner.

    Args:
        jobs:
            The jobs to run.
        dataset:
            The new dataset id to evaluate on.
        gpu_memory_utilization:
            The utilization fraction to pass to euroeval, or None.
    """
    for index, job in enumerate(jobs, start=1):
        split = "test" if job.evaluate_test_split else "val"
        shots = "zero-shot" if job.zero_shot else "few-shot"
        logger.info(
            f"[{index}/{len(jobs)}] euroeval --model {job.model_id} "
            f"--dataset {dataset} --language {' --language '.join(job.languages)} "
            f"({split}, {shots})"
        )
        returncode, _ = run_euroeval(
            model_id=job.model_id,
            languages=job.languages,
            datasets=[dataset],
            evaluate_test_split=job.evaluate_test_split,
            zero_shot=job.zero_shot,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if returncode != 0:
            logger.warning(
                f"{job.model_id} / {dataset}: euroeval exited with code "
                f"{returncode}; continuing."
            )


@dataclass(frozen=True)
class _Provider:
    """An API provider: its short name, env var, and id predicate."""

    name: str
    env_var: str
    matches: c.Callable[[str], bool]


PROVIDERS: list[_Provider] = [
    _Provider("openai", "OPENAI_API_KEY", lambda m: m.startswith(("openai/", "gpt-"))),
    _Provider(
        "anthropic",
        "ANTHROPIC_API_KEY",
        lambda m: m.startswith(("claude-", "anthropic/")),
    ),
    _Provider(
        "google", "GEMINI_API_KEY", lambda m: m.startswith(("gemini/", "gemini-"))
    ),
    _Provider("xai", "XAI_API_KEY", lambda m: m.startswith(("xai/", "grok-"))),
]
PROVIDERS_BY_NAME: dict[str, _Provider] = {
    provider.name: provider for provider in PROVIDERS
}


def provider_for_model_id(model_id: str) -> _Provider | None:
    """Return the API provider that owns a model id, or None.

    Args:
        model_id:
            The model id to classify.

    Returns:
        The matching provider, or None when no provider claims the id.
    """
    for provider in PROVIDERS:
        if provider.matches(model_id):
            return provider
    return None


def resolve_api_providers(
    include_api: bool, api_providers_arg: str | None, present_providers: set[str]
) -> set[str]:
    """Resolve which API providers to run and verify their env vars.

    Args:
        include_api:
            Whether the user opted in to API evaluation.
        api_providers_arg:
            Comma-separated provider names, or None to run every provider
            present among the ranked API models.
        present_providers:
            Provider names actually present among the ranked API models.

    Returns:
        The provider names to run.

    Raises:
        click.ClickException:
            When an unknown provider is named or a selected provider's env var
            is missing.
    """
    if not include_api or not present_providers:
        return set()
    if api_providers_arg is None:
        selected = set(present_providers)
    else:
        names = {n.strip().lower() for n in api_providers_arg.split(",") if n.strip()}
        unknown = sorted(names - PROVIDERS_BY_NAME.keys())
        if unknown:
            raise click.ClickException(
                f"Unknown API provider(s): {', '.join(unknown)}. "
                f"Valid: {', '.join(PROVIDERS_BY_NAME)}."
            )
        selected = names & present_providers
    missing = [
        PROVIDERS_BY_NAME[name].env_var
        for name in sorted(selected)
        if not os.environ.get(PROVIDERS_BY_NAME[name].env_var)
    ]
    if missing:
        raise click.ClickException(
            f"Selected API provider(s) require: {', '.join(missing)} -- set the "
            "variable(s) and re-run."
        )
    if selected:
        logger.info(f"API providers enabled: {', '.join(sorted(selected))}.")
    return selected


def record_is_api(model_info: dict[str, object]) -> bool:
    """Return whether a record's model was evaluated via an API.

    Args:
        model_info:
            The ``model_info`` object of an EEE result record.

    Returns:
        True when produced by ``litellm`` or flagged as not open-weight.
    """
    engine = model_info.get("inference_engine", {})
    engine_name = engine.get("name", "") if isinstance(engine, dict) else ""
    if str(engine_name).lower() == "litellm":
        return True
    details = model_info.get("additional_details", {})
    open_flag = details.get("open") if isinstance(details, dict) else None
    return str(open_flag).lower() == "false"


def _record_languages(record: dict) -> list[str]:
    """Return the language codes a result record covers.

    Args:
        record:
            A result record in EEE format.

    Returns:
        The list of language codes.
    """
    raw = (
        record.get("eval_library", {})
        .get("additional_details", {})
        .get("languages", "[]")
    )
    if isinstance(raw, list):
        return [str(code) for code in raw]
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(code) for code in parsed] if isinstance(parsed, list) else []


# --------------------------------------------------------------------------- #
# Config + documentation swap
# --------------------------------------------------------------------------- #
def apply_swap(old_dataset: str, new_dataset: str, dry_run: bool) -> list[Path]:
    """Swap officiality in the dataset configs and the frontend docs.

    Args:
        old_dataset:
            The dataset to demote to unofficial.
        new_dataset:
            The dataset to promote to official.
        dry_run:
            When True, log the files that would change without editing them.

    Returns:
        The paths that were (or would be) modified.
    """
    changed: list[Path] = []
    for dataset_id, make_official in ((new_dataset, True), (old_dataset, False)):
        config_path = find_config_file(dataset_id=dataset_id)
        changed.append(config_path)
        if not dry_run:
            set_config_officiality(
                path=config_path, dataset_id=dataset_id, official=make_official
            )
        for doc_path in find_doc_files(dataset_id=dataset_id):
            changed.append(doc_path)
            if not dry_run:
                set_doc_officiality(
                    path=doc_path, dataset_id=dataset_id, official=make_official
                )
    unique = sorted({path for path in changed}, key=str)
    verb = "Would edit" if dry_run else "Edited"
    for path in unique:
        logger.info(f"{verb} {path.relative_to(REPO_ROOT)}.")
    return unique


def find_config_file(dataset_id: str) -> Path:
    """Return the dataset-config file that defines ``dataset_id``.

    Args:
        dataset_id:
            The dataset id to locate.

    Returns:
        The path to the ``dataset_configs/<lang>.py`` file.

    Raises:
        click.ClickException:
            When no config file references the dataset id.
    """
    needle = re.compile(rf'name\s*=\s*"{re.escape(dataset_id)}"')
    for path in sorted(DATASET_CONFIG_DIR.glob("*.py")):
        if needle.search(path.read_text(encoding="utf-8")):
            return path
    raise click.ClickException(
        f"Could not find a dataset config defining {dataset_id!r}."
    )


def find_doc_files(dataset_id: str) -> list[Path]:
    """Return the dataset-doc files that document ``dataset_id``.

    Each dataset's section ends with ``euroeval ... --dataset <id>``; that is a
    reliable anchor regardless of the section's human-readable heading.

    Args:
        dataset_id:
            The dataset id to locate.

    Returns:
        The matching doc paths (possibly several for multilingual datasets).

    Raises:
        click.ClickException:
            When no doc file references the dataset id.
    """
    needle = re.compile(rf"--dataset {re.escape(dataset_id)}\b")
    matches = [
        path
        for path in sorted(DATASET_DOC_DIR.glob("*.md"))
        if needle.search(path.read_text(encoding="utf-8"))
    ]
    if not matches:
        raise click.ClickException(
            f"Could not find dataset documentation for {dataset_id!r}."
        )
    return matches


def set_config_officiality(path: Path, dataset_id: str, official: bool) -> None:
    """Flip a dataset's officiality in a config file and reposition its block.

    Adds/removes the ``unofficial=True`` line and moves the ``DatasetConfig``
    block into the official section (just before the unofficial marker) or the
    unofficial section (just after it), keeping the file's grouping.

    Args:
        path:
            The config file to edit.
        dataset_id:
            The dataset id whose block to move.
        official:
            True to make it official, False to make it unofficial.
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    start, end = _config_block_span(lines=lines, dataset_id=dataset_id, path=path)
    block = lines[start : end + 1]
    if official:
        block = [line for line in block if not UNOFFICIAL_LINE_RE.match(line)]
    elif not any(UNOFFICIAL_LINE_RE.match(line) for line in block):
        block = block[:-1] + ["    unofficial=True,", block[-1]]

    # Drop the block and exactly one adjacent blank line to keep spacing tidy.
    del lines[start : end + 1]
    if start < len(lines) and lines[start].strip() == "":
        del lines[start]
    elif start > 0 and lines[start - 1].strip() == "":
        del lines[start - 1]

    marker = _marker_index(lines=lines, path=path)
    if official:
        insert_at = marker
    else:
        insert_at = marker + 1
        if insert_at < len(lines) and lines[insert_at].strip() == "":
            insert_at += 1
    lines[insert_at:insert_at] = block + [""]
    path.write_text("\n".join(lines), encoding="utf-8")


def set_doc_officiality(path: Path, dataset_id: str, official: bool) -> None:
    """Flip a dataset's ``Unofficial:`` heading prefix and reorder its section.

    Within the dataset's ``## Task`` section, official subsections (those
    without the ``Unofficial:`` prefix) are kept before unofficial ones; after
    flipping the heading the task section is stably re-partitioned so the
    promoted dataset moves up and the demoted one moves down.

    Args:
        path:
            The doc file to edit.
        dataset_id:
            The dataset id whose section to flip.
        official:
            True to drop the ``Unofficial:`` prefix, False to add it.
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    heading_idx = _doc_heading_index(lines=lines, dataset_id=dataset_id, path=path)
    lines[heading_idx] = _flip_heading(heading=lines[heading_idx], official=official)

    task_start, task_end = _doc_task_span(lines=lines, heading_idx=heading_idx)
    reordered = _partition_doc_subsections(section=lines[task_start:task_end])
    lines[task_start:task_end] = reordered
    path.write_text("\n".join(lines), encoding="utf-8")


def _config_block_span(
    lines: list[str], dataset_id: str, path: Path
) -> tuple[int, int]:
    """Return the inclusive ``(start, end)`` line span of a config block.

    Args:
        lines:
            The config file's lines.
        dataset_id:
            The dataset id whose block to find.
        path:
            The file (for error messages).

    Returns:
        The start (``NAME = DatasetConfig(``) and end (``)``) indices.

    Raises:
        click.ClickException:
            When the block can't be delimited.
    """
    name_re = re.compile(rf'name\s*=\s*"{re.escape(dataset_id)}"')
    name_idx = next((i for i, line in enumerate(lines) if name_re.search(line)), None)
    if name_idx is None:
        raise click.ClickException(f"{dataset_id!r} not found in {path}.")
    start = name_idx
    while start >= 0 and not lines[start].rstrip().endswith("DatasetConfig("):
        start -= 1
    end = name_idx
    while end < len(lines) and lines[end].strip() != ")":
        end += 1
    if start < 0 or end >= len(lines):
        raise click.ClickException(f"Could not delimit {dataset_id!r} block in {path}.")
    return start, end


def _marker_index(lines: list[str], path: Path) -> int:
    """Return the index of the unofficial-section marker comment.

    Args:
        lines:
            The config file's lines.
        path:
            The file (for error messages).

    Returns:
        The marker line index.

    Raises:
        click.ClickException:
            When the marker is absent.
    """
    for i, line in enumerate(lines):
        if line.strip() == UNOFFICIAL_MARKER:
            return i
    raise click.ClickException(f"No {UNOFFICIAL_MARKER!r} marker in {path}.")


def _doc_heading_index(lines: list[str], dataset_id: str, path: Path) -> int:
    """Return the ``### ...`` heading index for a dataset's doc section.

    The section is the one whose body contains ``--dataset <id>``.

    Args:
        lines:
            The doc file's lines.
        dataset_id:
            The dataset id to find.
        path:
            The file (for error messages).

    Returns:
        The heading line index.

    Raises:
        click.ClickException:
            When the section can't be found.
    """
    anchor = re.compile(rf"--dataset {re.escape(dataset_id)}\b")
    anchor_idx = next((i for i, line in enumerate(lines) if anchor.search(line)), None)
    if anchor_idx is None:
        raise click.ClickException(f"{dataset_id!r} not documented in {path}.")
    for i in range(anchor_idx, -1, -1):
        if lines[i].startswith("### "):
            return i
    raise click.ClickException(f"No '### ' heading above {dataset_id!r} in {path}.")


def _doc_task_span(lines: list[str], heading_idx: int) -> tuple[int, int]:
    """Return the ``[start, end)`` line span of the enclosing ``## Task`` section.

    The span starts at the first ``### `` subsection under the task heading and
    ends before the next ``## `` heading (or end of file), so the task's intro
    lines are left in place.

    Args:
        lines:
            The doc file's lines.
        heading_idx:
            The ``### `` heading index of the dataset's subsection.

    Returns:
        The ``(start, end)`` span covering the task's ``### `` subsections.
    """
    task_idx = heading_idx
    while task_idx >= 0 and not lines[task_idx].startswith("## "):
        task_idx -= 1
    start = task_idx + 1
    while start < len(lines) and not lines[start].startswith("### "):
        start += 1
    end = start
    while end < len(lines) and not lines[end].startswith("## "):
        end += 1
    return start, end


def _partition_doc_subsections(section: list[str]) -> list[str]:
    """Stably reorder a task section's ``### `` subsections official-first.

    Args:
        section:
            The lines of a task section, beginning at its first ``### ``
            subsection.

    Returns:
        The lines with official subsections (no ``Unofficial:`` prefix) kept in
        order first, then the unofficial ones.
    """
    subsections: list[list[str]] = []
    current: list[str] = []
    for line in section:
        if line.startswith("### ") and current:
            subsections.append(current)
            current = []
        current.append(line)
    if current:
        subsections.append(current)

    official = [
        sub
        for sub in subsections
        if not sub[0].startswith(f"### {DOC_UNOFFICIAL_PREFIX}")
    ]
    unofficial = [
        sub for sub in subsections if sub[0].startswith(f"### {DOC_UNOFFICIAL_PREFIX}")
    ]
    result: list[str] = []
    for sub in official + unofficial:
        result.extend(sub)
    return result


def _flip_heading(heading: str, official: bool) -> str:
    """Add or remove the ``Unofficial:`` prefix on a ``### `` heading.

    Args:
        heading:
            The heading line (``### Title`` or ``### Unofficial: Title``).
        official:
            True to drop the prefix, False to add it.

    Returns:
        The updated heading line.
    """
    title = heading[len("### ") :]
    has_prefix = title.startswith(DOC_UNOFFICIAL_PREFIX)
    if official and has_prefix:
        title = title[len(DOC_UNOFFICIAL_PREFIX) :]
    elif not official and not has_prefix:
        title = f"{DOC_UNOFFICIAL_PREFIX}{title}"
    return f"### {title}"


# --------------------------------------------------------------------------- #
# Git + pull request
# --------------------------------------------------------------------------- #
def default_branch() -> str:
    """Return the repository's default branch name.

    Returns:
        The default branch (``origin/HEAD`` target), or ``"main"`` if it can't
        be determined.
    """
    result = _git("symbolic-ref", "refs/remotes/origin/HEAD", check=False, capture=True)
    if result.returncode == 0:
        return result.stdout.strip().rsplit("/", 1)[-1]
    return "main"


def checkout_branch(branch: str) -> None:
    """Check out ``branch``, creating it if it doesn't exist.

    Args:
        branch:
            The branch to switch to.
    """
    existing = _git("rev-parse", "--verify", branch, check=False, capture=True)
    if existing.returncode == 0:
        _git("checkout", branch)
    else:
        _git("checkout", "-b", branch)
    logger.info(f"On branch {branch!r}.")


def open_pull_request(
    old_dataset: str, new_dataset: str, branch: str, changed_paths: list[Path]
) -> None:
    """Commit the swap, push the branch, and open a pull request.

    Assigns the logged-in GitHub user, requests ``saattrupdan`` as a reviewer,
    and best-effort requests a Copilot review.

    Args:
        old_dataset:
            The demoted dataset id.
        new_dataset:
            The promoted dataset id.
        branch:
            The branch to push.
        changed_paths:
            The files that were changed (staged explicitly).
    """
    for path in changed_paths:
        _git("add", str(path))
    title = f"feat: swap official dataset {old_dataset} -> {new_dataset}"
    body = _pr_body(old_dataset=old_dataset, new_dataset=new_dataset)
    _git("commit", "-m", title, "-m", body)
    _git("push", "--set-upstream", "origin", branch)

    _gh(
        "pr",
        "create",
        "--title",
        title,
        "--body",
        body,
        "--assignee",
        "@me",
        "--reviewer",
        DEFAULT_REVIEWER,
    )
    _request_copilot_review()
    logger.info("Opened pull request.")


def _pr_body(old_dataset: str, new_dataset: str) -> str:
    """Return the standard PR description for a dataset swap.

    Args:
        old_dataset:
            The demoted dataset id.
        new_dataset:
            The promoted dataset id.

    Returns:
        The markdown PR body.
    """
    return (
        f"Swaps the official dataset `{old_dataset}` for `{new_dataset}`.\n\n"
        "## What\n\n"
        f"- Every model with a rank score on the affected leaderboard(s) was "
        f"evaluated on `{new_dataset}`, mirroring how each appears on the "
        "leaderboard (validation/test split and zero-/few-shot).\n"
        f"- `{old_dataset}` is demoted to unofficial and `{new_dataset}` "
        "promoted to official in the dataset configs and the dataset "
        "documentation, keeping each file's official-first grouping.\n\n"
        "The leaderboards will pick up the change on the next regeneration.\n\n"
        "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
    )


def _request_copilot_review() -> None:
    """Best-effort request a Copilot review on the current branch's PR."""
    result = _gh(
        "pr",
        "edit",
        "--add-reviewer",
        "copilot-pull-request-reviewer[bot]",
        check=False,
        capture=True,
    )
    if result.returncode != 0:
        logger.info(
            "Could not explicitly request a Copilot review (it may still run "
            "automatically): %s",
            result.stderr.strip(),
        )


def _git(
    *args: str, check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run a git command in the repo root.

    Args:
        args:
            The git subcommand and arguments.
        check:
            Whether to raise on a non-zero exit.
        capture:
            Whether to capture stdout/stderr.

    Returns:
        The completed process.
    """
    return _run(["git", *args], check=check, capture=capture)


def _gh(
    *args: str, check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run a ``gh`` command in the repo root.

    Args:
        args:
            The gh subcommand and arguments.
        check:
            Whether to raise on a non-zero exit.
        capture:
            Whether to capture stdout/stderr.

    Returns:
        The completed process.
    """
    return _run(["gh", *args], check=check, capture=capture)


def _run(
    cmd: list[str], check: bool, capture: bool
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess in the repo root.

    Args:
        cmd:
            The command and arguments.
        check:
            Whether to raise on a non-zero exit.
        capture:
            Whether to capture stdout/stderr.

    Returns:
        The completed process.

    Raises:
        click.ClickException:
            When ``check`` is True and the command fails.
    """
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=capture, text=True, check=False
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() if capture and result.stderr else ""
        raise click.ClickException(
            f"Command failed ({' '.join(cmd)}): exit {result.returncode}. {detail}"
        )
    return result


def dataset_config(dataset_id: str) -> DatasetConfig | None:
    """Return the :class:`DatasetConfig` for a dataset id, or None if unknown.

    Args:
        dataset_id:
            The dataset id to look up.

    Returns:
        The matching config, or None when unknown.
    """
    configs = get_all_dataset_configs(
        custom_datasets_file=Path(""),
        dataset_ids=[dataset_id],
        api_key=None,
        cache_dir=Path(".cache"),
        trust_remote_code=False,
        run_with_cli=False,
    )
    return configs.get(dataset_id)


if __name__ == "__main__":
    main()
