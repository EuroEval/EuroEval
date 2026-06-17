"""Run EuroEval for the core model list.

Two related workflows live in this script:

1. **Dataset-replacement mode** (``--dataset <id>`` [repeatable]) -- re-run
   every core model whose language coverage overlaps a dataset's languages
   on that dataset.
2. **Backfill mode** (default) -- for every core model, run every official
   ``(dataset, language)`` pair that the model has no result line for yet.

Both modes share model selection, size-check, results lookup, and the API-
provider prompt.

Invoke as::

    uv run src/scripts/run_core_model_evaluations.py [...]

Required env vars (open-weight models)
--------------------------------------
HF_TOKEN          Resolved via :func:`evaluation_common.resolve_hf_token`,
                  injected into the euroeval subprocess so gated repos
                  with granted access can be downloaded.

Required env vars (API models)
------------------------------
Only the providers the user picks need a key set; the script aborts up
front if a selected provider's key is missing.

OPENAI_API_KEY     OpenAI models (``openai/...``).
ANTHROPIC_API_KEY  Anthropic models (``claude-...``).
GEMINI_API_KEY     Google models (``gemini/...``).
XAI_API_KEY        xAI models (``xai/...``).
"""

from __future__ import annotations

import collections.abc as c
import json
import logging
import os
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import click
from update_core_models import refresh_core_models

from euroeval.dataset_configs import get_all_dataset_configs
from euroeval.languages import get_all_languages
from leaderboards.constants import NEW_RESULTS_PATH, RESULTS_PATH
from leaderboards.core_models import CoreModel
from leaderboards.evaluation_common import (
    gpu_total_memory_bytes,
    model_fits_locally,
    official_dataset_language_pairs,
    run_euroeval,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("run_core_model_evaluations")


@click.command()
@click.option(
    "--dataset",
    "datasets",
    multiple=True,
    help="Dataset id to run in dataset-replacement mode. Repeatable. When omitted, "
    "the script runs in backfill mode against every official (dataset, language) "
    "pair.",
)
@click.option(
    "--include-api",
    is_flag=True,
    default=False,
    help="Opt in to running API models. Without this flag every api: true entry "
    "in core_models.yaml is silently skipped.",
)
@click.option(
    "--api-providers",
    default=None,
    help="Comma-separated provider names to run (openai, anthropic, google, xai). "
    "Skips the interactive prompt. Requires --include-api.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the planned jobs and exit without invoking euroeval.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-run even (model, dataset, language) triples that already have a "
    "result line.",
)
def main(
    datasets: tuple[str, ...],
    include_api: bool,
    api_providers: str | None,
    dry_run: bool,
    force: bool,
) -> None:
    """Run EuroEval for the core model list.

    Args:
        datasets:
            Dataset ids passed via ``--dataset``. When non-empty, the
            script runs in dataset-replacement mode against exactly those
            datasets; otherwise it runs in backfill mode.
        include_api:
            Whether to consider ``api: true`` entries at all.
        api_providers:
            Optional comma-separated provider filter; bypasses the prompt.
        dry_run:
            When True, print the planned jobs and return without running.
        force:
            When True, ignore previously recorded results.

    Raises:
        click.ClickException:
            When ``--api-providers`` is passed without ``--include-api``,
            when an unknown provider name is supplied, or when a selected
            provider's env var is missing.
    """
    if api_providers and not include_api:
        raise click.ClickException(
            "--api-providers requires --include-api; pass both or neither."
        )

    target_datasets = list(datasets) or None
    mode = "dataset-replacement" if target_datasets else "backfill"
    logger.info(f"Mode: {mode}.")
    if target_datasets:
        logger.info(f"Target dataset(s): {', '.join(target_datasets)}.")

    models = refresh_core_models()
    logger.info(f"Refreshed core-model list: {len(models)} entries.")

    selected_providers = select_api_providers(
        candidate_models=models,
        include_api=include_api,
        api_providers_arg=api_providers,
    )

    observations: set[tuple[str, str, str]] = set()
    if not force:
        try:
            observations = load_existing_observations()
        except FileNotFoundError as e:
            logger.warning(
                f"Could not load existing results ({e}); proceeding as if none exist."
            )

    jobs = build_jobs(
        models=models,
        target_datasets=target_datasets,
        selected_providers=selected_providers,
        observations=observations,
        force=force,
    )
    logger.info(f"Planned {len(jobs)} job(s) before size check.")

    jobs = apply_size_filter(jobs=jobs)
    logger.info(f"{len(jobs)} job(s) survive the size check.")

    if dry_run:
        for job in jobs:
            tag = "api" if job.is_api else "open"
            click.echo(
                f"[{tag}] {job.model_id} :: {job.dataset} :: {', '.join(job.languages)}"
            )
        return

    execute_jobs(jobs=jobs)


@dataclass(frozen=True)
class Job:
    """A single (model, dataset, languages) evaluation job."""

    model_id: str
    dataset: str
    languages: tuple[str, ...]
    is_api: bool


@dataclass(frozen=True)
class _Provider:
    """An API provider: its short name, required env var, and id predicate."""

    name: str
    env_var: str
    matches: c.Callable[[str], bool]


def select_api_providers(
    candidate_models: c.Iterable[CoreModel],
    include_api: bool,
    api_providers_arg: str | None,
) -> set[str]:
    """Resolve which API providers to run and verify their env vars.

    Args:
        candidate_models:
            The full set of models in scope, used to check whether any API
            models are present at all.
        include_api:
            Whether the user opted in to API evaluation. When False, no
            providers are selected regardless of ``api_providers_arg``.
        api_providers_arg:
            Comma-separated list of provider names, or None to prompt.

    Returns:
        The set of provider names that should be evaluated.

    Raises:
        click.ClickException:
            When a selected provider's env var is unset, or an unknown
            provider name is given.
    """
    if not include_api:
        return set()

    has_any_api = any(m.api for m in candidate_models)
    if not has_any_api:
        logger.info("No API models in the candidate set; ignoring --include-api.")
        return set()

    if api_providers_arg is None:
        selected = _prompt_api_providers()
    else:
        names = [n.strip().lower() for n in api_providers_arg.split(",") if n.strip()]
        unknown = sorted(set(names) - PROVIDERS_BY_NAME.keys())
        if unknown:
            raise click.ClickException(
                f"Unknown API provider(s): {', '.join(unknown)}. "
                f"Valid: {', '.join(PROVIDERS_BY_NAME)}."
            )
        selected = set(names)

    missing_env = [
        PROVIDERS_BY_NAME[name].env_var
        for name in sorted(selected)
        if not os.environ.get(PROVIDERS_BY_NAME[name].env_var)
    ]
    if missing_env:
        raise click.ClickException(
            f"Selected API provider(s) require: {', '.join(missing_env)} -- "
            "set the variable(s) and re-run."
        )
    if selected:
        logger.info(f"API providers enabled: {', '.join(sorted(selected))}.")
    return selected


def load_existing_observations() -> set[tuple[str, str, str]]:
    """Return the ``(model_id, dataset, language)`` triples already benchmarked.

    Reads the merged result corpus the same way the leaderboard pipeline
    does -- ``results.tar.gz`` plus the optional ``new_results.jsonl`` --
    but without the destructive unlink that
    :func:`leaderboards.result_loading.load_raw_results` performs. Supports
    both EEE format (with nested ``model_info`` and ``eval_library``) and
    old EuroEval format (flat keys).

    Returns:
        Every ``(model_id, dataset, language)`` triple in the merged
        result corpus, with model ids unwrapped from any leaderboard
        HTML anchor so the keys line up with the canonical ids in
        ``core_models.yaml``.
    """
    lines: list[str] = []
    if RESULTS_PATH.exists():
        with tarfile.open(RESULTS_PATH, "r:gz") as tar:
            member_file = tar.extractfile(member="results/results.jsonl")
            if member_file is not None:
                lines.extend(member_file.read().decode(encoding="utf-8").splitlines())
    if NEW_RESULTS_PATH.exists():
        lines.extend(NEW_RESULTS_PATH.read_text(encoding="utf-8").splitlines())

    observations: set[tuple[str, str, str]] = set()
    parse_failures = 0
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        # Some lines pack multiple JSON objects back-to-back; split on '}{'
        # the same way result_loading does.
        for record_text in re.split(pattern=r"(?<=})(?={)", string=line):
            if not record_text.strip():
                continue
            try:
                raw_record = json.loads(record_text)
            except json.JSONDecodeError as e:
                parse_failures += 1
                logger.warning(
                    f"Skipping malformed JSON on result line {line_idx:,}: {e}; "
                    f"snippet={record_text[:120]!r}."
                )
                continue

            # Extract fields from EEE or old format
            # EEE: model_info.name, eval_library.additional_details.dataset/languages
            # Old: model, dataset, languages/dataset_languages at top level
            is_eee = "schema_version" in raw_record
            if is_eee:
                model = raw_record.get("model_info", {}).get("name", "")
                eval_additional = raw_record.get("eval_library", {}).get(
                    "additional_details", {}
                )
                dataset = eval_additional.get("dataset")
                languages_raw = eval_additional.get("languages", "[]")
                try:
                    languages = (
                        json.loads(languages_raw)
                        if isinstance(languages_raw, str)
                        else languages_raw
                    )
                except json.JSONDecodeError:
                    languages = []
            else:
                model = raw_record.get("model", "")
                dataset = raw_record.get("dataset")
                languages = (
                    raw_record.get("languages")
                    or raw_record.get("dataset_languages")
                    or []
                )

            model = _strip_anchor(model_id=str(model))
            if not model or not dataset:
                continue
            for language in languages:
                observations.add((model, str(dataset), str(language)))
    logger.info(
        f"Loaded {len(observations):,} (model, dataset, language) observations."
    )
    if parse_failures:
        logger.warning(
            f"Skipped {parse_failures:,} unparseable result record(s); see "
            "earlier warnings for details."
        )
    return observations


def build_jobs(
    models: list[CoreModel],
    target_datasets: list[str] | None,
    selected_providers: set[str],
    observations: set[tuple[str, str, str]],
    force: bool,
) -> list[Job]:
    """Compute the (model, dataset, languages) jobs to schedule.

    Args:
        models:
            Parsed core-model entries.
        target_datasets:
            Dataset ids for dataset-replacement mode, or None for backfill.
        selected_providers:
            Provider names that survived the env-var check; API models
            belonging to other providers are silently skipped.
        observations:
            Already-recorded ``(model, dataset, language)`` triples; used
            to drop jobs that nothing new would learn (unless ``force``).
        force:
            When True, ignore ``observations`` and re-run even seen pairs.

    Returns:
        One :class:`Job` per (model, dataset) pair to evaluate, with
        languages deduplicated and sorted.
    """
    jobs: list[Job] = []
    official_pairs = official_dataset_language_pairs()
    official_by_dataset: dict[str, set[str]] = {}
    for dataset, language in official_pairs:
        official_by_dataset.setdefault(dataset, set()).add(language)

    target_dataset_codes: dict[str, set[str]] | None = None
    if target_datasets:
        target_dataset_codes = {
            d: dataset_language_codes(dataset_id=d) for d in target_datasets
        }

    for model in models:
        if model.api:
            provider = provider_for_model_id(model_id=model.model_id)
            if provider is None or provider.name not in selected_providers:
                continue
        model_languages = codes_for_model(model=model)
        if not model_languages:
            logger.info(
                f"{model.model_id}: skipping -- no resolvable language coverage."
            )
            continue

        if target_dataset_codes is not None:
            for dataset, dataset_langs in target_dataset_codes.items():
                langs = sorted(model_languages & dataset_langs)
                if not langs:
                    continue
                if not force:
                    langs = [
                        lang
                        for lang in langs
                        if (model.model_id, dataset, lang) not in observations
                    ]
                if not langs:
                    continue
                jobs.append(
                    Job(
                        model_id=model.model_id,
                        dataset=dataset,
                        languages=tuple(langs),
                        is_api=model.api,
                    )
                )
            continue

        for dataset, dataset_langs in official_by_dataset.items():
            langs = sorted(model_languages & dataset_langs)
            if not langs:
                continue
            if not force:
                langs = [
                    lang
                    for lang in langs
                    if (model.model_id, dataset, lang) not in observations
                ]
            if not langs:
                continue
            jobs.append(
                Job(
                    model_id=model.model_id,
                    dataset=dataset,
                    languages=tuple(langs),
                    is_api=model.api,
                )
            )
    return jobs


def apply_size_filter(jobs: list[Job]) -> list[Job]:
    """Drop open-weight jobs whose model can't fit on the local GPU.

    API jobs are passed through untouched (we cannot size-check them).
    Open-weight models whose safetensors footprint cannot be measured
    are also passed through; the queue script uses the same policy.

    Args:
        jobs:
            The full job list before the size check.

    Returns:
        The jobs that should still be scheduled.
    """
    gpu_bytes = gpu_total_memory_bytes()
    if gpu_bytes is None:
        logger.info("Local memory budget unknown; skipping size pre-check.")
        return jobs
    logger.info(f"Local memory budget: {gpu_bytes / (1024**3):.1f} GiB.")

    sized_cache: dict[str, bool] = {}
    kept: list[Job] = []
    for job in jobs:
        if job.is_api:
            kept.append(job)
            continue
        if job.model_id not in sized_cache:
            fits, needed = model_fits_locally(
                model_id=job.model_id, gpu_bytes=gpu_bytes
            )
            sized_cache[job.model_id] = fits
            if not fits and needed is not None:
                logger.info(
                    f"{job.model_id}: skipping -- needs "
                    f"~{needed / (1024**3):.1f} GiB of weights, exceeds local "
                    f"budget of {gpu_bytes / (1024**3):.1f} GiB."
                )
        if sized_cache[job.model_id]:
            kept.append(job)
    return kept


def execute_jobs(jobs: list[Job]) -> None:
    """Run each job in sequence via the shared euroeval runner.

    API jobs are run on the validation split with zero-shot prompting;
    open-weight jobs use the defaults (test split, few-shot).

    Args:
        jobs:
            The jobs to execute.
    """
    for index, job in enumerate(jobs, start=1):
        logger.info(
            f"[{index}/{len(jobs)}] euroeval --model {job.model_id} "
            f"--dataset {job.dataset} --language {' --language '.join(job.languages)}"
            f"{' (api, val, zero-shot)' if job.is_api else ''}"
        )
        returncode, _ = run_euroeval(
            model_id=job.model_id,
            languages=job.languages,
            datasets=[job.dataset],
            evaluate_test_split=not job.is_api,
            zero_shot=job.is_api,
        )
        if returncode != 0:
            logger.warning(
                f"{job.model_id} / {job.dataset}: euroeval exited with "
                f"code {returncode}; continuing."
            )


def codes_for_model(model: CoreModel) -> set[str]:
    """Return the ISO codes a model should be evaluated on.

    API models cover every known language. Open-weight models cover the
    ISO codes derived from their ``pareto_languages`` -- with one
    exception: when the Pareto list is empty (typical for EU/OSAI
    entries that haven't been benchmarked widely enough yet), fall back
    to "all known languages" so the backfill run is what generates the
    data the Pareto computation needs.

    Args:
        model:
            The core-model entry whose language coverage to resolve.

    Returns:
        The ISO codes the model should be evaluated on.
    """
    if model.api or not model.pareto_languages:
        return all_known_language_codes()
    codes: set[str] = set()
    for name in model.pareto_languages:
        code = language_name_to_code(name=name)
        if code is None:
            logger.debug(f"Could not map language name {name!r} to an ISO code.")
            continue
        codes.add(code)
    return codes


def dataset_language_codes(dataset_id: str) -> set[str]:
    """Return ISO codes for a dataset, or an empty set if the dataset is unknown.

    Args:
        dataset_id:
            The dataset id to look up.

    Returns:
        The ISO codes the dataset covers, or an empty set when the id is
        not a known dataset (logged at ERROR level).
    """
    configs = get_all_dataset_configs(
        custom_datasets_file=Path(""),
        dataset_ids=[dataset_id],
        api_key=None,
        cache_dir=Path(".cache"),
        trust_remote_code=False,
        run_with_cli=False,
    )
    config = configs.get(dataset_id)
    if config is None:
        logger.error(f"Unknown dataset id {dataset_id!r}; skipping.")
        return set()
    return {language.code for language in config.languages}


def provider_for_model_id(model_id: str) -> _Provider | None:
    """Return the API provider that owns a model id, or None.

    Args:
        model_id:
            The model id to classify.

    Returns:
        The matching :class:`_Provider`, or None when no provider claims
        the id (i.e. the model is not an API model).
    """
    for provider in PROVIDERS:
        if provider.matches(model_id):
            return provider
    return None


def language_name_to_code(name: str) -> str | None:
    """Map an English language name to its ISO code, or None.

    The names come from ``core_models.yaml::pareto_languages``; values are
    matched case-insensitively against
    :func:`euroeval.languages.get_all_languages`.

    Args:
        name:
            The English language name (case-insensitive).

    Returns:
        The ISO code for the language, or None when no match is found.
    """
    needle = name.strip().lower()
    if not needle:
        return None
    for code, language in get_all_languages().items():
        if language.name.lower() == needle:
            return code
    return None


def all_known_language_codes() -> set[str]:
    """Return every ISO code known to EuroEval.

    Returns:
        Every key from :func:`euroeval.languages.get_all_languages`.
    """
    return set(get_all_languages().keys())


def _strip_anchor(model_id: str) -> str:
    """Strip a leaderboard HTML anchor wrapper from a model id.

    ``BenchmarkResult.from_dict`` already removes the ``(val)`` /
    ``(few-shot)`` variant suffixes, so we only need to defend against
    anchor wrappers that older leaderboard exports may have left in
    the record's ``model`` field.

    Args:
        model_id:
            The model id from a parsed :class:`BenchmarkResult`.

    Returns:
        The canonical model id, ready to compare against ``core_models.yaml``.
    """
    m = _ANCHOR_RE.search(model_id)
    return (m.group("inner").strip() if m else model_id).strip()


def _prompt_api_providers() -> set[str]:
    """Prompt the user interactively for which API providers to run.

    Returns:
        The selected provider names. May be empty if the user
        deselects everything.

    Raises:
        click.ClickException:
            When stdin is not a TTY (so we cannot prompt safely).
    """
    if not sys.stdin.isatty():
        raise click.ClickException(
            "stdin is not a TTY; pass --api-providers to select providers "
            "non-interactively (or omit --include-api)."
        )
    click.echo("Select API providers to evaluate (default: all):")
    selected: set[str] = set()
    for provider in PROVIDERS:
        if click.confirm(f"  Include {provider.name}?", default=True):
            selected.add(provider.name)
    return selected


_ANCHOR_RE = re.compile(r"<a [^>]*>(?P<inner>[^<]+)</a>")


PROVIDERS: list[_Provider] = [
    _Provider(
        name="openai",
        env_var="OPENAI_API_KEY",
        matches=lambda m: m.startswith("openai/"),
    ),
    _Provider(
        name="anthropic",
        env_var="ANTHROPIC_API_KEY",
        matches=lambda m: m.startswith("claude-") or m.startswith("anthropic/"),
    ),
    _Provider(
        name="google",
        env_var="GEMINI_API_KEY",
        matches=lambda m: m.startswith("gemini/"),
    ),
    _Provider(
        name="xai", env_var="XAI_API_KEY", matches=lambda m: m.startswith("xai/")
    ),
]
PROVIDERS_BY_NAME: dict[str, _Provider] = {p.name: p for p in PROVIDERS}


if __name__ == "__main__":
    main()
