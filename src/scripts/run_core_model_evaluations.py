"""Run EuroEval for the core model list.

Three related workflows live in this script:

1. **Dataset-replacement mode** (``--dataset <id>`` [repeatable]) -- re-run
   every core model whose language coverage overlaps a dataset's languages
   on that dataset.
2. **Backfill mode** (default) -- for every core model, run every official
   ``(dataset, language)`` pair that the model has no result line for yet.
3. **Leaderboard re-eval mode** (``--reeval-old <ds> --reeval-new <ds>``) --
   when a new (still unofficial) dataset is going to replace an official one
   on a language's leaderboard, evaluate that candidate on every model that
   currently holds a full rank score there (i.e. holds every required
   dataset). Run this *before* flipping the official flags so the
   leaderboard stays intact while the data is gathered. Unlike the other two
   modes the model set is drawn from the recorded results, not from
   ``core_models.yaml``.

All modes share the size-check, results lookup, and the API-provider prompt.

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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import click
from update_core_models import refresh_core_models

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
from leaderboards.core_models import CoreModel
from leaderboards.evaluation_common import (
    gpu_total_memory_bytes,
    model_fits_locally,
    official_dataset_language_pairs,
    run_euroeval,
)
from leaderboards.task_metadata import (
    category_includes_task,
    official_datasets_for_language,
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
    "--reeval-old",
    "reeval_old",
    default=None,
    help="Leaderboard re-eval mode: the official dataset being replaced, whose "
    "ranked models define the set to evaluate. Requires --reeval-new.",
)
@click.option(
    "--reeval-new",
    "reeval_new",
    default=None,
    help="Leaderboard re-eval mode: the unofficial candidate dataset to evaluate on "
    "every model ranked under --reeval-old. Requires --reeval-old.",
)
@click.option(
    "--language",
    "languages",
    multiple=True,
    help="Leaderboard re-eval mode: restrict re-evaluation to these language codes. "
    "When omitted, defaults to the languages covered by both the old and new "
    "datasets.",
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
    "--gpu-memory-utilization",
    "gpu_memory_utilization",
    type=float,
    default=None,
    help="vLLM GPU memory utilization fraction (0.0-1.0) the fit pre-check should "
    f"budget against. When omitted, defaults to {DEFAULT_GPU_MEMORY_UTILIZATION}.",
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
    reeval_old: str | None,
    reeval_new: str | None,
    languages: tuple[str, ...],
    include_api: bool,
    api_providers: str | None,
    gpu_memory_utilization: float | None,
    dry_run: bool,
    force: bool,
) -> None:
    """Run EuroEval for the core model list.

    Args:
        datasets:
            Dataset ids passed via ``--dataset``. When non-empty, the
            script runs in dataset-replacement mode against exactly those
            datasets; otherwise it runs in backfill mode.
        reeval_old:
            Outgoing dataset id for leaderboard re-eval mode, or None.
        reeval_new:
            Incoming dataset id for leaderboard re-eval mode, or None.
        languages:
            Language codes to restrict re-eval to; empty means "infer from
            the old and new datasets". Only used in re-eval mode.
        include_api:
            Whether to consider ``api: true`` entries at all.
        api_providers:
            Optional comma-separated provider filter; bypasses the prompt.
        gpu_memory_utilization:
            vLLM GPU memory utilization fraction the fit pre-check budgets
            against, or None to use the default.
        dry_run:
            When True, print the planned jobs and return without running.
        force:
            When True, ignore previously recorded results.

    Raises:
        click.ClickException:
            When ``--api-providers`` is passed without ``--include-api``,
            when an unknown provider name is supplied, when a selected
            provider's env var is missing, or when the re-eval flags are
            combined incorrectly.
    """
    if api_providers and not include_api:
        raise click.ClickException(
            "--api-providers requires --include-api; pass both or neither."
        )
    if bool(reeval_old) != bool(reeval_new):
        raise click.ClickException(
            "--reeval-old and --reeval-new must be passed together."
        )
    reeval = bool(reeval_old and reeval_new)
    if reeval and datasets:
        raise click.ClickException(
            "--reeval-old/--reeval-new cannot be combined with --dataset."
        )
    if languages and not reeval:
        raise click.ClickException(
            "--language is only valid in leaderboard re-eval mode "
            "(--reeval-old/--reeval-new)."
        )

    target_datasets = list(datasets) or None
    mode = (
        "leaderboard-reeval"
        if reeval
        else "dataset-replacement"
        if target_datasets
        else "backfill"
    )
    logger.info(f"Mode: {mode}.")

    if reeval:
        validate_reeval_datasets(old_dataset=reeval_old, new_dataset=reeval_new)

    corpus = ObservedCorpus(observations=set(), api_model_ids=set(), eval_configs={})
    try:
        corpus = load_existing_observations()
    except FileNotFoundError as e:
        if reeval:
            raise click.ClickException(
                f"Leaderboard re-eval needs the recorded results to find ranked "
                f"models, but none could be loaded: {e}"
            ) from e
        logger.warning(
            f"Could not load existing results ({e}); proceeding as if none exist."
        )

    if reeval:
        jobs = plan_reeval_jobs(
            old_dataset=reeval_old,
            new_dataset=reeval_new,
            requested_languages=languages,
            corpus=corpus,
            include_api=include_api,
            api_providers_arg=api_providers,
            force=force,
        )
    else:
        if target_datasets:
            logger.info(f"Target dataset(s): {', '.join(target_datasets)}.")
        models = refresh_core_models()
        logger.info(f"Refreshed core-model list: {len(models)} entries.")
        selected_providers = select_api_providers(
            has_any_api=any(m.api for m in models),
            include_api=include_api,
            api_providers_arg=api_providers,
        )
        jobs = build_jobs(
            models=models,
            target_datasets=target_datasets,
            selected_providers=selected_providers,
            observations=corpus.observations if not force else set(),
            force=force,
        )
    logger.info(f"Planned {len(jobs)} job(s) before size check.")

    jobs = apply_size_filter(jobs=jobs, gpu_memory_utilization=gpu_memory_utilization)
    logger.info(f"{len(jobs)} job(s) survive the size check.")

    if dry_run:
        for job in jobs:
            tag = "api" if job.is_api else "open"
            split = "test" if job.evaluate_test_split else "val"
            shots = "zero-shot" if job.zero_shot else "few-shot"
            click.echo(
                f"[{tag}] {job.model_id} :: {job.dataset} :: "
                f"{', '.join(job.languages)} :: {split}, {shots}"
            )
        return

    execute_jobs(jobs=jobs, gpu_memory_utilization=gpu_memory_utilization)


@dataclass(frozen=True)
class Job:
    """A single (model, dataset, languages) evaluation job.

    ``evaluate_test_split`` and ``zero_shot`` capture how the model should
    be run, so :func:`execute_jobs` doesn't have to re-derive them: in
    re-eval mode they mirror how the model was evaluated on the outgoing
    dataset; in the core-model modes they follow the split/prompting default
    for the model type.
    """

    model_id: str
    dataset: str
    languages: tuple[str, ...]
    is_api: bool
    evaluate_test_split: bool
    zero_shot: bool


@dataclass(frozen=True)
class _ObsConfig:
    """How a model was evaluated on a given (dataset, language).

    Args:
        validation_split:
            True when the recorded result used the validation split, False
            for the test split, None when the record didn't say.
        few_shot:
            True when few-shot prompting was used, False for zero-shot, None
            when the record didn't say.
        generative:
            Whether the model is generative (few-shot/zero-shot only applies
            to generative models).
    """

    validation_split: bool | None
    few_shot: bool | None
    generative: bool


@dataclass(frozen=True)
class ObservedCorpus:
    """The recorded results, indexed for re-eval planning.

    Args:
        observations:
            Every ``(model_id, dataset, language)`` triple in the corpus.
        api_model_ids:
            Model ids the corpus shows were evaluated via an API.
        eval_configs:
            The evaluation config used for each ``(model, dataset, language)``
            triple, so a re-eval can mirror the split and prompting.
    """

    observations: set[tuple[str, str, str]]
    api_model_ids: set[str]
    eval_configs: dict[tuple[str, str, str], _ObsConfig]


@dataclass(frozen=True)
class _Provider:
    """An API provider: its short name, required env var, and id predicate."""

    name: str
    env_var: str
    matches: c.Callable[[str], bool]


def select_api_providers(
    has_any_api: bool, include_api: bool, api_providers_arg: str | None
) -> set[str]:
    """Resolve which API providers to run and verify their env vars.

    Args:
        has_any_api:
            Whether any API model is present in the candidate set; when
            False the ``--include-api`` opt-in is a no-op.
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


def load_existing_observations() -> ObservedCorpus:
    """Return the recorded results indexed for backfill and re-eval planning.

    Reads the merged result corpus the same way the leaderboard pipeline
    does -- the per-model files in ``RESULTS_DIR`` plus the optional
    ``new_results.jsonl`` -- but without the destructive unlink that
    :func:`leaderboards.result_loading.load_raw_results` performs.

    A model is treated as an API model when its record was produced by the
    ``litellm`` inference engine or is flagged as not open-weight. This is
    more reliable than id-prefix matching because historical API results are
    stored under bare ids (e.g. ``gpt-5.5``) that carry no ``openai/`` prefix.

    Each ``(model, dataset, language)`` triple also records how it was
    evaluated (split, few-shot, generative) so a re-eval can mirror it. When
    a triple appears more than once, a validation-split config is preferred,
    matching the completeness rule the queue uses.

    Returns:
        The parsed corpus. Model ids are unwrapped from any leaderboard HTML
        anchor so the keys line up with the canonical ids in
        ``core_models.yaml``.
    """
    lines: list[str] = []
    for model_file in sorted(RESULTS_DIR.glob("*.jsonl")):
        lines.extend(model_file.read_text(encoding="utf-8").splitlines())
    if NEW_RESULTS_PATH.exists():
        lines.extend(NEW_RESULTS_PATH.read_text(encoding="utf-8").splitlines())

    observations: set[tuple[str, str, str]] = set()
    api_model_ids: set[str] = set()
    eval_configs: dict[tuple[str, str, str], _ObsConfig] = {}
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

            # EEE: model_info.name, eval_library.additional_details.dataset/languages
            model_info = raw_record.get("model_info", {})
            model = model_info.get("name", "")
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

            model = _strip_anchor(model_id=str(model))
            if not model or not dataset:
                continue
            if _record_is_api(model_info=model_info):
                api_model_ids.add(model)
            config = _ObsConfig(
                validation_split=_as_bool(eval_additional.get("validation_split")),
                few_shot=_as_bool(eval_additional.get("few_shot")),
                generative=_as_bool(
                    model_info.get("additional_details", {}).get("generative")
                )
                is True,
            )
            for language in languages:
                key = (model, str(dataset), str(language))
                observations.add(key)
                existing = eval_configs.get(key)
                if existing is None or (
                    config.validation_split is True
                    and existing.validation_split is not True
                ):
                    eval_configs[key] = config
    logger.info(
        f"Loaded {len(observations):,} (model, dataset, language) observations "
        f"({len(api_model_ids):,} API model(s))."
    )
    if parse_failures:
        logger.warning(
            f"Skipped {parse_failures:,} unparseable result record(s); see "
            "earlier warnings for details."
        )
    return ObservedCorpus(
        observations=observations,
        api_model_ids=api_model_ids,
        eval_configs=eval_configs,
    )


def _as_bool(value: object) -> bool | None:
    """Coerce a JSON-ish truthy/falsy value to a tri-state bool.

    Args:
        value:
            A value that may be a bool, the strings ``"true"``/``"false"``
            (any case), or None/absent.

    Returns:
        True, False, or None when the value is missing or unrecognised.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _record_is_api(model_info: dict[str, object]) -> bool:
    """Return whether a result record's model was evaluated via an API.

    Args:
        model_info:
            The ``model_info`` object of an EEE result record.

    Returns:
        True when the record was produced by the ``litellm`` inference
        engine or is flagged as not open-weight.
    """
    engine = model_info.get("inference_engine", {})
    engine_name = engine.get("name", "") if isinstance(engine, dict) else ""
    if str(engine_name).lower() == "litellm":
        return True
    details = model_info.get("additional_details", {})
    open_flag = details.get("open") if isinstance(details, dict) else None
    return str(open_flag).lower() == "false"


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
                        evaluate_test_split=not model.api,
                        zero_shot=model.api,
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
                    evaluate_test_split=not model.api,
                    zero_shot=model.api,
                )
            )
    return jobs


def plan_reeval_jobs(
    old_dataset: str,
    new_dataset: str,
    requested_languages: tuple[str, ...],
    corpus: ObservedCorpus,
    include_api: bool,
    api_providers_arg: str | None,
    force: bool,
) -> list[Job]:
    """Plan the jobs for leaderboard re-eval mode.

    Selects the language(s) whose leaderboard is affected by the swap, finds
    every model that held a full rank score under the pre-swap config (in the
    generative *or* the NLU/all-models category, so encoder models are
    included when the task supports them), and schedules the new dataset for
    each such model in the languages it was ranked in. Every job mirrors the
    split and prompting the model used on the outgoing dataset. API models are
    excluded unless the user opts in with ``--include-api``, so a plain
    re-eval never spends money.

    Args:
        old_dataset:
            The outgoing dataset id being replaced.
        new_dataset:
            The incoming dataset id to run.
        requested_languages:
            Language codes to restrict to; empty means the codes covered by
            both the old and new datasets.
        corpus:
            The parsed results corpus (observations, API-model ids, and the
            per-triple evaluation configs to mirror).
        include_api:
            Whether the user opted in to running API models.
        api_providers_arg:
            Optional comma-separated provider filter.
        force:
            When True, re-run even models that already have a new-dataset
            result line.

    Returns:
        The re-eval jobs to schedule.

    Raises:
        click.ClickException:
            When the target languages can't be resolved, or a selected API
            provider's env var is missing.
    """
    old_codes = dataset_language_codes(dataset_id=old_dataset)
    new_codes = dataset_language_codes(dataset_id=new_dataset)
    if requested_languages:
        target_codes = set(requested_languages)
    else:
        target_codes = old_codes & new_codes
    if not target_codes:
        raise click.ClickException(
            "No target languages: the old and new datasets share no languages. "
            "Pass --language to specify the affected leaderboard(s) explicitly."
        )
    logger.info(
        f"Re-eval: {old_dataset!r} -> {new_dataset!r} on language(s): "
        f"{', '.join(sorted(target_codes))}."
    )

    old_config = dataset_config(dataset_id=old_dataset)
    swapped_task = old_config.task.name if old_config is not None else ""
    ranked_pairs = ranked_model_language_pairs(
        old_dataset=old_dataset,
        new_dataset=new_dataset,
        swapped_task=swapped_task,
        language_codes=target_codes,
        observations=corpus.observations,
    )
    logger.info(f"Found {len(ranked_pairs)} ranked (model, language) pair(s).")

    def is_api_model(model_id: str) -> bool:
        return (
            model_id in corpus.api_model_ids
            or provider_for_model_id(model_id) is not None
        )

    ranked_api = sorted({m for m, _ in ranked_pairs if is_api_model(m)})
    selected_providers = select_api_providers(
        has_any_api=bool(ranked_api),
        include_api=include_api,
        api_providers_arg=api_providers_arg,
    )
    if ranked_api and not include_api:
        logger.info(
            f"Excluding {len(ranked_api)} ranked API model(s); pass --include-api "
            "to evaluate them."
        )

    jobs: list[Job] = []
    for model_id, code in sorted(ranked_pairs):
        if code not in new_codes:
            logger.info(
                f"{model_id}: skipping {code} -- new dataset {new_dataset!r} does "
                f"not cover it."
            )
            continue
        is_api = is_api_model(model_id=model_id)
        if is_api:
            if not include_api:
                continue
            # Known-provider API models are gated on the provider selection; a
            # bare-id API model whose provider we can't classify is included
            # once the operator has opted in with --include-api.
            provider = provider_for_model_id(model_id=model_id)
            if provider is not None and provider.name not in selected_providers:
                continue
        if not force and (model_id, new_dataset, code) in corpus.observations:
            continue
        evaluate_test_split, zero_shot = _mirror_eval_config(
            config=corpus.eval_configs.get((model_id, old_dataset, code)), is_api=is_api
        )
        jobs.append(
            Job(
                model_id=model_id,
                dataset=new_dataset,
                languages=(code,),
                is_api=is_api,
                evaluate_test_split=evaluate_test_split,
                zero_shot=zero_shot,
            )
        )
    return jobs


def _mirror_eval_config(config: _ObsConfig | None, is_api: bool) -> tuple[bool, bool]:
    """Return ``(evaluate_test_split, zero_shot)`` mirroring the old dataset.

    The re-eval must reproduce how the model was evaluated on the outgoing
    dataset: the same split, and -- for generative models only -- the same
    zero-shot/few-shot prompting. Encoder models never get ``--zero-shot``.
    When no recorded config is available, fall back to the split/prompting
    default for the model type (API: validation split, zero-shot;
    open-weight: test split, few-shot).

    Args:
        config:
            The recorded evaluation config for the outgoing dataset, or None.
        is_api:
            Whether the model is an API model, used only for the fallback.

    Returns:
        The ``(evaluate_test_split, zero_shot)`` flags for the new job.
    """
    if config is None:
        return (not is_api), is_api
    # Validation split only when the record explicitly says so; otherwise test.
    evaluate_test_split = config.validation_split is not True
    # Zero-shot only for generative models whose record used zero-shot.
    zero_shot = config.generative and config.few_shot is False
    return evaluate_test_split, zero_shot


def ranked_model_language_pairs(
    old_dataset: str,
    new_dataset: str,
    swapped_task: str,
    language_codes: set[str],
    observations: set[tuple[str, str, str]],
) -> set[tuple[str, str]]:
    """Return ``(model_id, language_code)`` pairs ranked under the pre-swap config.

    A model is "ranked" in a language when it holds a result for every
    required (non-orthogonal) dataset of that language's single-language
    leaderboard -- the same eligibility rule the leaderboard generator uses
    to award a rank score. Eligibility is checked in every leaderboard
    category the swapped task belongs to: the ``generative`` category scores
    all tasks, while the ``all_models`` category scores only NLU tasks so
    encoder models can be compared. A model ranked in *either* category is
    selected, so encoder models are re-evaluated whenever the swapped task is
    an NLU task they can run.

    The required set is normalised so it always reflects the pre-swap
    leaderboard: the outgoing dataset is added and the incoming candidate
    removed. Because the re-eval runs before the flags are flipped this is
    normally a no-op (the outgoing dataset is already official and the
    incoming one already excluded), but it keeps the selection correct even
    if the configs have already been edited.

    Args:
        old_dataset:
            The outgoing dataset id being replaced (kept in the required set).
        new_dataset:
            The incoming candidate dataset id (kept out of the required set).
        swapped_task:
            The task both datasets belong to; determines which leaderboard
            categories are affected.
        language_codes:
            The language codes whose leaderboards are being re-evaluated.
        observations:
            Recorded ``(model, dataset, language)`` triples.

    Returns:
        The ``(model_id, language_code)`` pairs that were ranked pre-swap.
    """
    languages = get_all_languages()
    affected_categories = [
        category
        for category in LEADERBOARD_CATEGORIES
        if category_includes_task(category=category, task=swapped_task)
    ]
    if not affected_categories:
        logger.warning(
            f"Task {swapped_task!r} is in no leaderboard category; nothing to do."
        )
        return set()

    # Index observed datasets by (model, language) for O(1) subset checks.
    datasets_by_model_language: dict[tuple[str, str], set[str]] = defaultdict(set)
    for model_id, dataset, code in observations:
        datasets_by_model_language[(model_id, code)].add(dataset)

    ranked: set[tuple[str, str]] = set()
    for code in sorted(language_codes):
        language = languages.get(code)
        if language is None:
            logger.warning(f"Unknown language code {code!r}; skipping.")
            continue
        name = language.name.lower()
        if " " in name:
            logger.warning(
                f"{code!r} ({name!r}) is a dialect/multi-word language with no "
                f"standalone leaderboard; skipping."
            )
            continue
        try:
            by_task = official_datasets_for_language(name)
        except ValueError:
            logger.warning(f"No leaderboard datasets for {name!r}; skipping.")
            continue

        # A model is ranked in this language if it is eligible in any affected
        # category, i.e. holds every required dataset of that category.
        for category in affected_categories:
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
            for (model_id, obs_code), datasets in datasets_by_model_language.items():
                if obs_code == code and required <= datasets:
                    ranked.add((model_id, code))
    return ranked


def apply_size_filter(
    jobs: list[Job], gpu_memory_utilization: float | None
) -> list[Job]:
    """Drop open-weight jobs whose model can't fit on the local GPU.

    Mirrors the queue script's fit pre-check: vLLM can only allocate
    ``gpu_memory_utilization * total GPU memory``, so the budget is that
    fraction of the card rather than the whole card -- otherwise a model
    whose weights fit but whose KV cache does not still slips through and
    OOMs at runtime.

    API jobs are passed through untouched (we cannot size-check them).
    Open-weight models whose safetensors footprint cannot be measured
    are also passed through; the queue script uses the same policy.

    Args:
        jobs:
            The full job list before the size check.
        gpu_memory_utilization:
            The vLLM GPU memory utilization fraction the eval will run at,
            or None to use :data:`DEFAULT_GPU_MEMORY_UTILIZATION`.

    Returns:
        The jobs that should still be scheduled.
    """
    gpu_bytes = gpu_total_memory_bytes()
    if gpu_bytes is None:
        logger.info("Local memory budget unknown; skipping size pre-check.")
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

    sized_cache: dict[str, bool] = {}
    kept: list[Job] = []
    for job in jobs:
        if job.is_api:
            kept.append(job)
            continue
        if job.model_id not in sized_cache:
            fits, needed = model_fits_locally(
                model_id=job.model_id, gpu_bytes=usable_bytes
            )
            sized_cache[job.model_id] = fits
            if not fits and needed is not None:
                logger.info(
                    f"{job.model_id}: skipping -- needs "
                    f"~{needed / (1024**3):.1f} GiB of weights, exceeds the usable "
                    f"vLLM budget of {usable_bytes / (1024**3):.1f} GiB."
                )
        if sized_cache[job.model_id]:
            kept.append(job)
    return kept


def execute_jobs(jobs: list[Job], gpu_memory_utilization: float | None) -> None:
    """Run each job in sequence via the shared euroeval runner.

    Each job carries its own split and prompting flags (mirrored from the
    outgoing dataset in re-eval mode, or the model-type default otherwise),
    so they are forwarded to euroeval verbatim.

    Args:
        jobs:
            The jobs to execute.
        gpu_memory_utilization:
            vLLM GPU memory utilization fraction to pass to euroeval, or
            None to let the CLI default apply. Passing the same value the
            fit pre-check budgeted against keeps the two consistent.
    """
    for index, job in enumerate(jobs, start=1):
        split = "test" if job.evaluate_test_split else "val"
        shots = "zero-shot" if job.zero_shot else "few-shot"
        logger.info(
            f"[{index}/{len(jobs)}] euroeval --model {job.model_id} "
            f"--dataset {job.dataset} --language {' --language '.join(job.languages)} "
            f"({split}, {shots})"
        )
        returncode, _ = run_euroeval(
            model_id=job.model_id,
            languages=job.languages,
            datasets=[job.dataset],
            evaluate_test_split=job.evaluate_test_split,
            zero_shot=job.zero_shot,
            gpu_memory_utilization=gpu_memory_utilization,
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


def dataset_config(dataset_id: str) -> DatasetConfig | None:
    """Return the :class:`DatasetConfig` for a dataset id, or None if unknown.

    Args:
        dataset_id:
            The dataset id to look up.

    Returns:
        The matching config, or None when the id is not a known dataset.
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


def dataset_language_codes(dataset_id: str) -> set[str]:
    """Return ISO codes for a dataset, or an empty set if the dataset is unknown.

    Args:
        dataset_id:
            The dataset id to look up.

    Returns:
        The ISO codes the dataset covers, or an empty set when the id is
        not a known dataset (logged at ERROR level).
    """
    config = dataset_config(dataset_id=dataset_id)
    if config is None:
        logger.error(f"Unknown dataset id {dataset_id!r}; skipping.")
        return set()
    return {language.code for language in config.languages}


def validate_reeval_datasets(old_dataset: str, new_dataset: str) -> None:
    """Validate the old/new dataset pair for leaderboard re-eval mode.

    The re-eval runs *before* the config flags are flipped: the candidate
    replacement is still unofficial and the dataset it will replace is still
    official, so the leaderboard stays intact while the new dataset is
    evaluated on every ranked model. Enforces a genuine within-task swap:
    the outgoing dataset must be official, the incoming (candidate) dataset
    must be unofficial, both must have dataset configs, and both must belong
    to the same task.

    Args:
        old_dataset:
            The outgoing dataset id being replaced (expected to be official).
        new_dataset:
            The incoming candidate dataset id to evaluate (expected to be
            unofficial).

    Raises:
        click.ClickException:
            When either dataset is unknown, the official/unofficial flags
            are wrong, or the two datasets are not the same task.
    """
    old_config = dataset_config(dataset_id=old_dataset)
    new_config = dataset_config(dataset_id=new_dataset)
    if old_config is None:
        raise click.ClickException(
            f"--reeval-old {old_dataset!r} has no dataset config; both datasets "
            "must already be configured."
        )
    if new_config is None:
        raise click.ClickException(
            f"--reeval-new {new_dataset!r} has no dataset config; both datasets "
            "must already be configured."
        )
    if old_config.unofficial:
        raise click.ClickException(
            f"--reeval-old {old_dataset!r} must be official (the dataset being "
            "replaced); it is currently unofficial. Only an official dataset can "
            "be replaced."
        )
    if not new_config.unofficial:
        raise click.ClickException(
            f"--reeval-new {new_dataset!r} must be UNofficial (the candidate "
            "replacement to evaluate); it is currently official. Keep it "
            "unofficial until every ranked model has been evaluated on it, then "
            "flip the flags."
        )
    if old_config.task.name != new_config.task.name:
        raise click.ClickException(
            f"--reeval-old and --reeval-new must belong to the same task; "
            f"{old_dataset!r} is {old_config.task.name!r} but {new_dataset!r} is "
            f"{new_config.task.name!r}."
        )


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
