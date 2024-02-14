"""Unit tests for the `benchmark_config_factory` module."""

import pytest
import torch
from scandeval.benchmark_config_factory import (
    get_correct_language_codes,
    prepare_device,
    prepare_languages,
    prepare_tasks_and_datasets,
)
from scandeval.dataset_configs import get_all_dataset_configs
from scandeval.enums import Device
from scandeval.languages import DA, EN, NB, NN, NO, get_all_languages
from scandeval.tasks import LA, get_all_tasks


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
def test_get_correct_language_codes(input_language_codes, expected_language_codes):
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
def test_prepare_languages(input_language_codes, input_language, expected_language):
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
    argnames=["input_task", "input_dataset", "expected_task", "expected_dataset"],
    argvalues=[
        (
            None,
            None,
            list(get_all_tasks().values()),
            list(get_all_dataset_configs().keys()),
        ),
        (
            "linguistic-acceptability",
            None,
            [LA],
            [cfg.name for cfg in get_all_dataset_configs().values() if LA == cfg.task],
        ),
    ],
    ids=["all tasks and datasets", "single task"],
)
def test_prepare_tasks_and_datasets(
    input_task, input_dataset, expected_task, expected_dataset
):
    """Test the output of `prepare_tasks_and_datasets`."""
    prepared_tasks, prepared_datasets = prepare_tasks_and_datasets(
        task=input_task, dataset=input_dataset
    )
    assert prepared_tasks == expected_task
    assert prepared_datasets == expected_dataset


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
def test_prepare_device(device, expected_device):
    """Test the output of `prepare_device`."""
    prepared_device = prepare_device(device=device)
    assert prepared_device == expected_device
