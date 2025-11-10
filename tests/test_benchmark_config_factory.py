"""Tests for the `benchmark_config_factory` module."""

from typing import Generator

import pytest
import torch

from euroeval.benchmark_config_factory import (
    get_correct_language_codes,
    prepare_dataset_configs,
    prepare_device,
    prepare_languages,
)
from euroeval.data_models import DatasetConfig, Language
from euroeval.dataset_configs import get_all_dataset_configs
from euroeval.dataset_configs.danish import ANGRY_TWEETS_CONFIG, SCALA_DA_CONFIG
from euroeval.enums import Device
from euroeval.exceptions import InvalidBenchmark
from euroeval.languages import (
    DANISH,
    ENGLISH,
    NORWEGIAN,
    NORWEGIAN_BOKMÅL,
    NORWEGIAN_NYNORSK,
    get_all_languages,
)
from euroeval.tasks import LA


@pytest.fixture(scope="module")
def all_official_dataset_configs() -> Generator[list[DatasetConfig], None, None]:
    """Fixture for all official dataset configurations."""
    yield [cfg for cfg in get_all_dataset_configs().values() if not cfg.unofficial]


@pytest.fixture(scope="module")
def all_official_la_dataset_configs() -> Generator[list[DatasetConfig], None, None]:
    """Fixture for all linguistic acceptability dataset configurations."""
    yield [
        cfg
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
        ("da", None, [DANISH]),
        (["da"], None, [DANISH]),
        (["da", "no"], ["da"], [DANISH]),
        (["da", "en"], None, [DANISH, ENGLISH]),
        ("no", None, [NORWEGIAN, NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK]),
        (["nb"], None, [NORWEGIAN_BOKMÅL, NORWEGIAN]),
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
        "expected_dataset_configs",
    ],
    argvalues=[
        (
            None,
            None,
            list(get_all_languages().values()),
            "all_official_dataset_configs",
        ),
        (
            "linguistic-acceptability",
            None,
            list(get_all_languages().values()),
            "all_official_la_dataset_configs",
        ),
        (None, "scala-da", list(get_all_languages().values()), [SCALA_DA_CONFIG]),
        (
            "linguistic-acceptability",
            ["scala-da", "angry-tweets"],
            list(get_all_languages().values()),
            [SCALA_DA_CONFIG],
        ),
        (
            ["linguistic-acceptability", "named-entity-recognition"],
            "scala-da",
            list(get_all_languages().values()),
            [SCALA_DA_CONFIG],
        ),
        (
            ["linguistic-acceptability", "sentiment-classification"],
            ["scala-da", "angry-tweets", "scandiqa-da"],
            list(get_all_languages().values()),
            [SCALA_DA_CONFIG, ANGRY_TWEETS_CONFIG],
        ),
        (
            ["linguistic-acceptability", "sentiment-classification"],
            ["scala-da", "angry-tweets", "scandiqa-sv"],
            [DANISH],
            [SCALA_DA_CONFIG, ANGRY_TWEETS_CONFIG],
        ),
        (
            ["linguistic-acceptability", "sentiment-classification"],
            None,
            [DANISH],
            [SCALA_DA_CONFIG, ANGRY_TWEETS_CONFIG],
        ),
        (
            None,
            new_config := DatasetConfig(
                name="new-dataset",
                pretty_name="New Dataset",
                source="some/hf-dataset",
                task=LA,
                languages=[DANISH, ENGLISH],
            ),
            [DANISH],
            [new_config],
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
        "custom dataset config",
    ],
)
def test_prepare_dataset_configs(
    input_task: str | list[str] | None,
    input_dataset: str | list[str] | None,
    input_languages: list[Language],
    expected_dataset_configs: list[DatasetConfig] | str,
    request: pytest.FixtureRequest,
) -> None:
    """Test the output of `prepare_dataset_configs`."""
    # This replaces the string with the actual fixture
    if isinstance(expected_dataset_configs, str):
        expected_dataset_configs = request.getfixturevalue(expected_dataset_configs)

    prepared_dataset_configs = prepare_dataset_configs(
        task=input_task, dataset=input_dataset, languages=input_languages
    )
    assert set(prepared_dataset_configs) == set(expected_dataset_configs)


def test_prepare_dataset_configs_invalid_task() -> None:
    """Test that an invalid task raises an error."""
    with pytest.raises(InvalidBenchmark):
        prepare_dataset_configs(task="invalid-task", dataset=None, languages=[DANISH])


def test_prepare_dataset_configs_invalid_dataset() -> None:
    """Test that an invalid dataset raises an error."""
    with pytest.raises(InvalidBenchmark):
        prepare_dataset_configs(
            task=None, dataset="invalid-dataset", languages=[DANISH]
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
