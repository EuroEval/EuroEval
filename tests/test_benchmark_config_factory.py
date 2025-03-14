"""Unit tests for the `benchmark_config_factory` module."""

from typing import Generator

import pytest
import torch

from euroeval.benchmark_config_factory import (
    get_correct_language_codes,
    prepare_device,
    prepare_languages,
    prepare_tasks_and_datasets,
)
from euroeval.data_models import Language, Task
from euroeval.dataset_configs import get_all_dataset_configs
from euroeval.enums import Device
from euroeval.exceptions import InvalidBenchmark
from euroeval.languages import DA, EN, NB, NN, NO, get_all_languages
from euroeval.tasks import LA, NER, SENT, SPEED, get_all_tasks


@pytest.fixture(scope="module")
def all_official_dataset_names() -> Generator[list[str], None, None]:
    """Fixture for all linguistic acceptability dataset configurations."""
    yield [
        cfg.name
        for cfg in get_all_dataset_configs().values()
        if not cfg.unofficial and cfg.task != SPEED
    ]


@pytest.fixture(scope="module")
def all_official_la_dataset_names() -> Generator[list[str], None, None]:
    """Fixture for all linguistic acceptability dataset configurations."""
    yield [
        cfg.name
        for cfg in get_all_dataset_configs().values()
        if LA == cfg.task and not cfg.unofficial
    ]


@pytest.mark.parametrize(
    argnames=["input_language_codes", "expected_language_codes"],
    argvalues=[
        ("da", ["da"]),
        (["da"], ["da"]),
        (["da", "en"], ["da", "en"]),
        ("no", ["no", "nb", "nn"]),
        (["nb"], ["nb", "no"]),
        ("all", list(get_all_languages().keys())),
    ],
    ids=[
        "single language",
        "single language as list",
        "multiple languages",
        "no -> no + nb + nn",
        "nb -> nb + no",
        "all -> all languages",
    ],
)
def test_get_correct_language_codes(
    input_language_codes: str | list[str], expected_language_codes: list[str]
) -> None:
    """Test that the correct language codes are returned."""
    languages = get_correct_language_codes(language_codes=input_language_codes)
    assert set(languages) == set(expected_language_codes)


@pytest.mark.parametrize(
    argnames=["input_language_codes", "input_language", "expected_language"],
    argvalues=[
        ("da", None, [DA]),
        (["da"], None, [DA]),
        (["da", "no"], ["da"], [DA]),
        (["da", "en"], None, [DA, EN]),
        ("no", None, [NO, NB, NN]),
        (["nb"], None, [NB, NO]),
        ("all", None, list(get_all_languages().values())),
    ],
    ids=[
        "single language",
        "single language as list",
        "language takes precedence over model language",
        "multiple languages",
        "no -> no + nb + nn",
        "nb -> nb + no",
        "all -> all languages",
    ],
)
def test_prepare_languages(
    input_language_codes: str | list[str],
    input_language: list[str] | None,
    expected_language: list[Language],
) -> None:
    """Test the output of `prepare_languages`."""
    prepared_language_codes = get_correct_language_codes(
        language_codes=input_language_codes
    )
    model_languages = prepare_languages(
        language_codes=input_language, default_language_codes=prepared_language_codes
    )
    model_languages = sorted(model_languages, key=lambda x: x.code)
    expected_language = sorted(expected_language, key=lambda x: x.code)
    assert model_languages == expected_language


@pytest.mark.parametrize(
    argnames=[
        "input_task",
        "input_dataset",
        "input_languages",
        "expected_task",
        "expected_dataset",
    ],
    argvalues=[
        (
            None,
            None,
            list(get_all_languages().values()),
            [t for t in get_all_tasks().values() if t != SPEED],
            "all_official_dataset_names",
        ),
        (
            "linguistic-acceptability",
            None,
            list(get_all_languages().values()),
            [LA],
            "all_official_la_dataset_names",
        ),
        (
            None,
            "scala-da",
            list(get_all_languages().values()),
            [t for t in get_all_tasks().values() if t != SPEED],
            ["scala-da"],
        ),
        (
            "linguistic-acceptability",
            ["scala-da", "angry-tweets"],
            list(get_all_languages().values()),
            [LA],
            ["scala-da"],
        ),
        (
            ["linguistic-acceptability", "named-entity-recognition"],
            "scala-da",
            list(get_all_languages().values()),
            [LA, NER],
            ["scala-da"],
        ),
        (
            ["linguistic-acceptability", "sentiment-classification"],
            ["scala-da", "angry-tweets", "scandiqa-da"],
            list(get_all_languages().values()),
            [LA, SENT],
            ["scala-da", "angry-tweets"],
        ),
        (
            ["linguistic-acceptability", "sentiment-classification"],
            ["scala-da", "angry-tweets", "scandiqa-sv"],
            [DA],
            [LA, SENT],
            ["scala-da", "angry-tweets"],
        ),
        (
            ["linguistic-acceptability", "sentiment-classification"],
            None,
            [DA],
            [LA, SENT],
            ["scala-da", "angry-tweets"],
        ),
    ],
    ids=[
        "all tasks and datasets",
        "single task",
        "single dataset",
        "single task and multiple datasets",
        "multiple tasks and single dataset",
        "multiple tasks and datasets",
        "multiple tasks and datasets, filtered by language",
        "multiple tasks, filtered by language",
    ],
)
def test_prepare_tasks_and_datasets(
    input_task: str | list[str] | None,
    input_dataset: str | list[str] | None,
    input_languages: list[Language],
    expected_task: list[Task],
    expected_dataset: list[str] | str,
    request: pytest.FixtureRequest,
) -> None:
    """Test the output of `prepare_tasks_and_datasets`."""
    # This replaces the string with the actual fixture
    if isinstance(expected_dataset, str):
        expected_dataset = request.getfixturevalue(expected_dataset)

    prepared_tasks, prepared_datasets = prepare_tasks_and_datasets(
        task=input_task, dataset=input_dataset, dataset_languages=input_languages
    )
    assert set(prepared_tasks) == set(expected_task)
    assert set(prepared_datasets) == set(expected_dataset)


def test_prepare_tasks_and_datasets_invalid_task() -> None:
    """Test that an invalid task raises an error."""
    with pytest.raises(InvalidBenchmark):
        prepare_tasks_and_datasets(
            task="invalid-task", dataset=None, dataset_languages=[DA]
        )


def test_prepare_tasks_and_datasets_invalid_dataset() -> None:
    """Test that an invalid dataset raises an error."""
    with pytest.raises(InvalidBenchmark):
        prepare_tasks_and_datasets(
            task=None, dataset="invalid-dataset", dataset_languages=[DA]
        )


@pytest.mark.parametrize(
    argnames=["device", "expected_device"],
    argvalues=[
        (Device.CPU, torch.device("cpu")),
        (
            None,
            (
                torch.device("cuda")
                if torch.cuda.is_available()
                else (
                    torch.device("mps")
                    if torch.backends.mps.is_available()
                    else torch.device("cpu")
                )
            ),
        ),
    ],
    ids=["device provided", "device not provided"],
)
def test_prepare_device(device: Device, expected_device: torch.device) -> None:
    """Test the output of `prepare_device`."""
    prepared_device = prepare_device(device=device)
    assert prepared_device == expected_device
