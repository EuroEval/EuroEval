"""Shot-mode policy and work planning for benchmark runs."""

import collections.abc as c
import typing as t

from .enums import GenerativeType, InferenceBackend, ModelType, ShotMode

if t.TYPE_CHECKING:
    from .data_models import BenchmarkResult, DatasetConfig, ModelConfig

ShotModeRequest = ShotMode | bool | None
ShotWork = tuple[ShotMode, "DatasetConfig"]


def cached_generative_type(
    records: c.Sequence["BenchmarkResult"],
) -> GenerativeType | None:
    """Infer one reliable generative type from cached benchmark records.

    Args:
        records:
            Cached records for a model.

    Returns:
        The shared generative type, or None when metadata is absent, mixed, or invalid.
    """
    if not records or any(not record.generative for record in records):
        return None
    try:
        types = {GenerativeType(record.generative_type) for record in records}
    except (TypeError, ValueError):
        return None
    return types.pop() if len(types) == 1 else None


def plan_shot_work(
    candidate_modes: c.Sequence[ShotMode], datasets: c.Sequence["DatasetConfig"]
) -> list[ShotWork]:
    """Create de-duplicated concrete mode/dataset work items.

    Args:
        candidate_modes:
            Candidate shot modes in execution order.
        datasets:
            Dataset configurations to pair with each mode.

    Returns:
        Concrete mode/dataset pairs, preserving input order.
    """
    work: list[ShotWork] = []
    seen_pairs: set[tuple[int, ShotMode]] = set()
    for mode in candidate_modes:
        for dataset_config in datasets:
            concrete_mode = effective_shot_mode(
                shot_mode=mode, dataset_config=dataset_config
            )
            if concrete_mode is None:
                continue
            pair_key = (id(dataset_config), concrete_mode)
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                work.append((concrete_mode, dataset_config))
    return work


def effective_shot_mode(
    shot_mode: ShotMode, dataset_config: "DatasetConfig"
) -> ShotMode | None:
    """Collapse a policy mode to the effective mode for one dataset.

    Args:
        shot_mode:
            A concrete or provisional shot mode.
        dataset_config:
            The dataset configuration.

    Returns:
        The mode to execute, or None when the requested work is redundant.
    """
    if dataset_config.task.requires_zero_shot:
        return (
            ShotMode.ZERO_SHOT
            if shot_mode in (ShotMode.AUTO, ShotMode.FEW_SHOT)
            else shot_mode
        )
    return shot_mode


def resolve_shot_modes(
    model_config: "ModelConfig",
    requested_mode: ShotModeRequest,
    generative_type: GenerativeType | None = None,
) -> list[ShotMode]:
    """Resolve a requested shot policy into concrete evaluation modes.

    Args:
        model_config:
            The model configuration.
        requested_mode:
            ``ShotMode.AUTO`` selects modes from model metadata. Booleans remain
            accepted for programmatic backwards compatibility, while ``None`` is
            treated as ``AUTO`` by this policy function.
        generative_type:
            The detected model type, when the model has already been loaded.
            Defaults to None.

    Returns:
        Concrete modes to evaluate, in execution order.
    """
    mode = coerce_shot_mode(requested_mode)
    if mode is not ShotMode.AUTO:
        return [mode]
    if model_config.model_type != ModelType.GENERATIVE:
        return [ShotMode.FEW_SHOT]
    if model_config.inference_backend == InferenceBackend.LITELLM:
        return [ShotMode.ZERO_SHOT]
    if generative_type == GenerativeType.BASE:
        return [ShotMode.FEW_SHOT]
    if generative_type in (GenerativeType.INSTRUCTION_TUNED, GenerativeType.REASONING):
        return [ShotMode.ZERO_SHOT, ShotMode.FEW_SHOT]
    # Before loading a local generative model, retain both possibilities so a cache
    # entry for one mode cannot suppress the other.
    return [ShotMode.ZERO_SHOT, ShotMode.FEW_SHOT]


def coerce_shot_mode(
    requested_mode: ShotModeRequest, *, none_mode: ShotMode = ShotMode.AUTO
) -> ShotMode:
    """Convert a public shot-mode value to a :class:`ShotMode`.

    Args:
        requested_mode:
            A shot mode, a legacy boolean, or ``None``.
        none_mode:
            The mode to use for ``None``. Defaults to :attr:`ShotMode.AUTO`.

    Returns:
        The equivalent concrete policy value. ``AUTO`` is a policy value and must be
        resolved before a benchmark result is created.

    Raises:
        TypeError:
            If ``requested_mode`` is not a supported shot-mode value.
    """
    if requested_mode is None:
        return none_mode
    if isinstance(requested_mode, bool):
        return ShotMode.FEW_SHOT if requested_mode else ShotMode.ZERO_SHOT
    if isinstance(requested_mode, ShotMode):
        return requested_mode
    raise TypeError(f"Unsupported shot mode: {requested_mode!r}")


def result_identity_values(
    shot_mode: ShotModeRequest,
    dataset_config: "DatasetConfig",
    evaluate_test_split: bool,
) -> tuple[bool | None, bool | None]:
    """Derive the identity fields emitted for a benchmark result.

    Args:
        shot_mode:
            Concrete shot mode used for the benchmark.
        dataset_config:
            Dataset configuration used for the benchmark.
        evaluate_test_split:
            Whether the test split was evaluated.

    Returns:
        The result's ``(few_shot, validation_split)`` identity values. Forced-zero-shot
        tasks and datasets without validation splits use ``None`` for the respective
        field.

    Raises:
        ValueError:
            If ``shot_mode`` is ``AUTO``, which must be resolved before a result is
            created.
    """
    mode = coerce_shot_mode(shot_mode)
    if mode is ShotMode.AUTO:
        raise ValueError("AUTO must be resolved before storing a benchmark result")
    few_shot = (
        None
        if dataset_config.task.requires_zero_shot
        else result_few_shot_value(requested_mode=mode)
    )
    validation_split = (
        None if dataset_config.val_split is None else not evaluate_test_split
    )
    return few_shot, validation_split


def result_few_shot_value(requested_mode: ShotModeRequest) -> bool:
    """Convert a concrete shot mode to the stored result boolean.

    Args:
        requested_mode:
            A concrete shot mode or a legacy boolean.

    Returns:
        Whether the result is few-shot.

    Raises:
        ValueError:
            If ``requested_mode`` is ``AUTO``, which must be resolved before a result
            is created.
    """
    mode = coerce_shot_mode(requested_mode)
    if mode is ShotMode.AUTO:
        raise ValueError("AUTO must be resolved before storing a benchmark result")
    return mode is ShotMode.FEW_SHOT
